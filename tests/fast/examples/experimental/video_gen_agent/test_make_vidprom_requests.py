import csv
import json
import random
import subprocess
import sys

import pytest
from examples.experimental.video_gen_agent.make_vidprom_requests import (
    TOXICITY,
    NearDuplicates,
    clean,
    is_english,
    is_keyword_list,
    is_nsfw,
    splits,
)

SCRIPT = "examples/experimental/video_gen_agent/make_vidprom_requests.py"
_ANIMALS = "dog cat fox horse rabbit owl tiger bear wolf deer goat duck parrot otter panda lion zebra monkey".split()
_PEOPLE = "girl boy farmer sailor dancer chef knight pilot painter baker fisherman robot astronaut monk".split()
_ACTS = "chases follows watches greets carries pushes feeds circles leads joins".split()
_PLACES = "forest beach city street desert kitchen garden harbor meadow canyon library station".split()
_COLORS = "red blue green golden silver white black orange purple brown".split()
_EXTRAS = "softly slowly quietly happily wildly gently bravely calmly eagerly proudly".split()


def _request(rng: random.Random) -> str:
    """Thirty-odd words of English naming two subjects, a setting and an action, worded differently each time."""
    a, b = rng.sample(_ANIMALS, 2)
    person = rng.choice(_PEOPLE)
    c1, c2, c3 = rng.sample(_COLORS, 3)
    e1, e2 = rng.sample(_EXTRAS, 2)
    return (
        f"A {c1} {a} {rng.choice(_ACTS)} a {c2} {b} {e1} through the {rng.choice(_PLACES)} while a {person} in a "
        f"{c3} coat {rng.choice(_ACTS)} them {e2}, and the {b} then turns around as the light fades over the "
        f"{rng.choice(_PLACES)}"
    )


def _vidprom(path, prompts, scores=None):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uuid", "prompt", "time", *TOXICITY])
        for i, prompt in enumerate(prompts):
            writer.writerow([f"u{i:05d}", prompt, "Sun Sep  3 12:27:44 2023", *(scores or {}).get(i, [0.001] * 6)])
    return path


def _moviegen(path, prompts):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "concept", "motion_level"])
        for i, prompt in enumerate(prompts):
            writer.writerow([prompt, "animal, scene" if i % 2 else "human - activity", "low"])
    return path


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_clean_strips_pika_flags_entities_and_a_prompt_prefix():
    assert clean("prompt: A cat&#39;s  dance on a roof -gs 24 -ar 16:9 -motion 2") == "A cat's dance on a roof"
    assert clean("A dog runs, -neg &quot;blurry, ugly&quot; -camera zoom in") == "A dog runs"
    assert clean("Odin walks home --ar 9:16") == "Odin walks home"
    assert clean("a well-lit room with a high-backed chair") == "a well-lit room with a high-backed chair"


@pytest.mark.parametrize(
    "text",
    ["A cat <em>sits</em>", "see https://x.y for the cat", "Style: `cinematic`", "A cat.\n\nThe next scene: a dog."],
)
def test_markup_links_and_multi_paragraph_scripts_are_dropped(text):
    assert clean(text) is None


def test_keyword_lists_are_refused_but_prose_with_commas_is_kept():
    assert is_keyword_list("steam train in snow, polar express, scenic, dusk, ultra detailed, octane render, 8k")
    assert not is_keyword_list(
        "A fox runs through the snow, its tail low, while an owl circles above the pines, and the moon rises slowly"
    )


def test_english_needs_ascii_letters_and_function_words():
    assert is_english("A small boat drifts on the lake while the sun sets behind the hills")
    assert not is_english("一只猫在屋顶上跳舞，月亮升起来了 and the cat")
    assert not is_english("cyberpunk samurai neon katana rain hologram skyline drone chase sparks")


def test_nsfw_is_a_word_or_any_detoxify_score_at_the_bar():
    assert is_nsfw("a naked statue in a garden")
    assert is_nsfw("a man walks", {"toxicity": 0.12, "obscene": 0.0})
    assert not is_nsfw("a man walks his dog in the park", {"toxicity": 0.09})


def test_near_duplicates_are_refused_and_rewordings_kept():
    dedup = NearDuplicates()
    first = "a red fox chases a white rabbit through the snowy forest while an owl watches from a pine branch above"
    assert dedup.add(first)
    assert not dedup.add(first.replace("white rabbit", "grey rabbit"))
    assert dedup.add("two old fishermen mend their nets on a wooden pier as gulls circle the harbor at dawn")


def test_splits_bucket_length_subjects_and_setting():
    tags = {"seconds": 8, "subjects": ["a dog", "a cat", "a boy"], "setting": None}
    assert splits(tags) == {"seconds": "8-11s", "subjects": "3+", "setting": "no"}
    assert splits(tags | {"seconds": 4, "subjects": ["a dog"], "setting": "a park"})["seconds"] == "4-7s"


@pytest.fixture
def spacy_model():
    """The tagger and WordNet that ``prepare`` installs, which the fast-test image lacks."""
    spacy = pytest.importorskip("spacy")
    wordnet = pytest.importorskip("nltk.corpus").wordnet
    try:
        wordnet.ensure_loaded()
        return spacy.load("en_core_web_md")
    except (OSError, LookupError):
        pytest.skip("en_core_web_md or WordNet is not installed")


def test_tags_name_the_subjects_setting_actions_and_a_length(spacy_model):
    from examples.experimental.video_gen_agent.make_vidprom_requests import is_rich, tag

    tags = tag(spacy_model("A red fox chases a white rabbit through a snowy forest, then an owl swoops down at night"))

    assert tags["subjects"] == ["A red fox", "a white rabbit", "an owl"]
    assert tags["setting"] == "a snowy forest"
    assert tags["actions"] == ["chase", "swoop"]
    assert tags["seconds"] == 8 and is_rich(tags)
    lone = tag(spacy_model("A cinematic shot of a lonely lighthouse, ultra detailed, 4k, beautiful lighting"))
    assert not is_rich(lone)


def test_the_build_filters_dedups_and_splits_real_requests(spacy_model, tmp_path):
    rng = random.Random(0)
    good = [_request(rng) for _ in range(60)]
    bad = {
        "short": "A dog chases a cat through the park.",
        "person": "Taylor Swift " + good[0][2:],
        "lowercase person": "taylor swift " + good[1][2:],
        "nsfw": good[2].replace("coat", "bikini"),
        "keywords": "dog, cat, park, sunny, 4k, ultra detailed, octane render, " + ", ".join(_COLORS) + ", trending",
        "duplicate": good[3] + " again",
        "toxic": _request(rng),
    }
    template = [f"Make a cinematic video of this: {_request(rng)}" for _ in range(5)]
    prompts = good + list(bad.values()) + template
    vidprom = _vidprom(tmp_path / "vidprom.csv", prompts, {len(good) + 6: [0.5] * 6})
    moviegen = _moviegen(tmp_path / "moviegen.csv", [_request(rng) for _ in range(12)] + [bad["short"]])

    out = tmp_path / "out"
    args = [sys.executable, SCRIPT, str(vidprom), str(moviegen), str(out), "--train", "40", "--heldout", "10"]
    subprocess.run([*args, "--moviegen", "6"], check=True)

    train, heldout, movie = (_rows(out / f) for f in ("train.jsonl", "heldout.jsonl", "moviegen_heldout.jsonl"))
    assert (len(train), len(heldout), len(movie)) == (40, 10, 6)
    requests = [row["messages"][0]["content"] for row in train + heldout]
    assert len(set(requests)) == 50 and set(requests) <= set(good)
    assert {row["metadata"]["source"] for row in train + heldout} == {"vidprom"}
    first = heldout[0]["metadata"]
    assert set(first) == {"source", "id", "subjects", "setting", "actions", "seconds", "splits"}
    assert len(first["subjects"]) >= 2 and 4 <= first["seconds"] <= 15 and first["splits"]["setting"] in ("yes", "no")
    assert {row["metadata"]["category"] for row in movie} <= {"animal", "human - activity"}
    assert all(row["metadata"]["source"] == "moviegenbench" for row in movie)
    funnel = json.loads((out / "funnel.json").read_text())["vidprom"]
    assert funnel["loaded"] == len(prompts) and funnel["template"] == 5 and funnel["nsfw"] == 2
    assert funnel["person name"] == 2 and funnel["near-duplicate"] >= 1

    again = tmp_path / "again"
    subprocess.run([*args[:-5], str(again), "--train", "40", "--heldout", "10", "--moviegen", "6"], check=True)
    assert (again / "heldout.jsonl").read_text() == (out / "heldout.jsonl").read_text()
    reseeded = tmp_path / "reseeded"
    subprocess.run([*args[:-5], str(reseeded), "--train", "40", "--heldout", "10", "--seed", "1"], check=True)
    assert (reseeded / "heldout.jsonl").read_text() != (out / "heldout.jsonl").read_text()


def test_the_build_refuses_a_pool_short_of_the_requested_split(spacy_model, tmp_path):
    rng = random.Random(1)
    vidprom = _vidprom(tmp_path / "vidprom.csv", [_request(rng) for _ in range(5)])
    moviegen = _moviegen(tmp_path / "moviegen.csv", [_request(rng) for _ in range(5)])
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            str(vidprom),
            str(moviegen),
            str(tmp_path / "out"),
            "--train",
            "10",
            "--heldout",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "only 5 requests passed the filters, short of 12" in result.stderr
