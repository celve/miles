"""Turn VidProM's real text-to-video requests and MovieGenBench's prompts into the env's prompt data for long clips.

VidProM (Wang & Yang, NeurIPS 2024) holds 1.04M semantically distinct prompts that users sent Pika in 2023
(``VidProM_semantic_unique.csv``, CC-BY-NC-4.0); MovieGenBench (Meta, 2024) has 1,003 prompts tagged by concept and
motion (``MovieGenVideoBenchWithTag.csv``, CC-BY-NC-4.0). Both are fetched at prepare time, for non-commercial research
only, and never copied into this repository. A VidProM request is kept when it is rich enough for a 4-15 s clip:

- after Pika's flags (``-gs 12 -motion 2 -neg ...``) and HTML entities are stripped, 25 to 120 words of English prose
  on one paragraph, with no markup, links or keyword list (most comma-separated parts a few words long);
- below 0.1 on every Detoxify score VidProM ships, and free of the sexual and gore words in ``NSFW_WORDS``;
- naming no person, by spaCy's named-entity tagger, by a multi-word name it tagged elsewhere in the pool, written in
  any case, or by a capitalized proper noun WordNet knows as a particular person (this drops fictional names too);
- not sharing its first or last six words with four other kept prompts (templates);
- not opening as a writing task (``Write``, ``Compose``, ``Hello`` ...);
- naming at least two subjects, or a subject, a setting and an action, where a subject is a noun phrase WordNet reads
  as a person, animal, artifact, plant, food or natural object (or a word it lacks), and a setting a place noun;
- under 0.5 word-trigram Jaccard (estimated by MinHash) to every request kept before it.

Requests are taken in the order of a hash of ``--seed`` and VidProM's uuid: the first ``--heldout`` are held out, the
next ``--train`` train. MovieGenBench's prompts pass the same filters but for the template rule, the Detoxify scores it
lacks and a floor of 20 words, and ``--moviegen`` of them form a second held-out set, with their concept as ``category``
and their ``motion_level``. Rows are ``make_requests.py``'s, plus metadata:

  {"messages": [{"role": "user", "content": <request>}],
   "metadata": {"source": "vidprom" | "moviegenbench", "id": <uuid> | "moviegen/<line>", "subjects": [...],
                "setting": <phrase> | null, "actions": [...], "seconds": 4-15,
                "splits": {"seconds": "4-7s" | "8-11s" | "12-15s", "subjects": "1" | "2" | "3+",
                           "setting": "yes" | "no"}}}

Tagging needs spaCy's ``en_core_web_md`` and NLTK's WordNet (under ``NLTK_DATA``), which the keyframe launcher's
``prepare`` installs. ``seconds`` suggests a clip length from the request's actions, sequence words and length;
``subjects``, ``setting`` and ``seconds`` are a heuristic reading, and ``splits`` buckets them for
``episode_metrics.log_eval_episodes``.

  python examples/experimental/video_gen_agent/make_vidprom_requests.py \\
      /root/datasets/VidProM/VidProM_semantic_unique.csv /root/datasets/MovieGenBench/benchmark/MovieGenVideoBenchWithTag.csv /root/datasets/video_gen_requests_vidprom
"""

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

SPACY_MODEL = "en_core_web_md"
MIN_WORDS, MAX_WORDS = 25, 120
# MovieGenBench's prompts run shorter (median 16 words), and 25 leaves fewer than 100 of them.
MOVIEGEN_MIN_WORDS = 20
TOXICITY = ("toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit")
MAX_TOXICITY = 0.1
NSFW_WORDS = frozenset(
    "nsfw nude nudity naked topless nipple nipples breast breasts boob boobs busty cleavage lingerie underwear bikini "
    "thong sex sexy sexual seductive erotic porn porno orgasm genital genitals penis vagina butt buttocks hentai "
    "fetish horny stripper gore gory bloody corpse corpses dismembered decapitated beheaded mutilated suicide rape "
    "torture".split()
)
TEMPLATE_WORDS, MAX_TEMPLATE_SHARE = 6, 5
MAX_JACCARD = 0.5
# Pika's flags, which end a prompt: everything from the first one on is dropped.
_FLAGS = re.compile(r"(?:^|[\s.,;:])-{1,2}\s?(?:neg|gs|ar|motion|camera|fps|seed|size|hide|style|v)\b.*", re.I | re.S)
_LEAD = re.compile(r"^\s*(?:prompt\s*:\s*)+", re.I)
_MARKUP = re.compile(r"<[a-z/!]|https?://|www\.|[{}\[\]|`#@]|\\n", re.I)
_WORD = re.compile(r"[a-z0-9']+")
_ENGLISH = frozenset(
    "a an the and or of in on at to with from by for into onto through over under is are was were be his her their "
    "its as while then who that which this there".split()
)
# Nouns that describe the clip rather than what is in it, and verbs that ask for it rather than happen in it.
_NOT_SUBJECTS = frozenset(
    "shot view scene style image video picture photo photograph film movie clip animation footage frame camera "
    "angle lens resolution quality detail details lighting light color colour colors colours background foreground "
    "atmosphere mood vibe aesthetic look feel effect effects motion movement composition render rendering art "
    "artwork illustration painting drawing design 4k 8k hd uhd closeup close-up side front top bottom middle center "
    "centre left right distance moment time day night morning evening afternoon sunset sunrise way kind type sort "
    "lot number piece part shape size form sense feeling emotion expression perspective focus depth tone theme "
    "prompt sequence series end beginning start".split()
)
_NOT_ACTIONS = frozenset(
    "be have do make create generate show depict render animate feature use include let want need try give add "
    "capture convey evoke imagine seem look appear set keep write describe portray illustrate film shoot style".split()
)
_SEQUENCE_WORDS = frozenset(
    "then after afterwards later finally next suddenly until before meanwhile eventually".split()
)
SETTING_NOUNS = frozenset(
    "forest woods jungle desert beach shore coast ocean sea lake river stream waterfall pond mountain mountains hill "
    "hills valley canyon cliff cave field meadow farm garden park island volcano glacier tundra savanna swamp marsh "
    "prairie countryside landscape sky skies space galaxy universe planet moon orbit city town village street road "
    "highway alley avenue square plaza market bazaar harbor harbour port dock pier bridge tunnel station airport "
    "subway train bus building skyscraper tower castle palace temple church cathedral mosque shrine ruins house home "
    "mansion cottage cabin hut apartment room bedroom kitchen bathroom hallway corridor attic basement office "
    "classroom school library museum gallery theater theatre stage studio restaurant cafe bar pub shop store mall "
    "factory warehouse lab laboratory hospital prison arena stadium court gym pool spaceship spacecraft cockpit "
    "battlefield underwater seabed reef graveyard cemetery rooftop roof balcony courtyard yard backyard porch "
    "playground campsite camp tent kingdom realm world wasteland metropolis cityscape skyline downtown suburb "
    "neighborhood neighbourhood ship deck interior exterior snow rain fog storm".split()
)
_LOCATIVE = frozenset(
    "in at on inside within through across along over under beneath below above near beside by around amid amidst "
    "into onto toward towards outside throughout".split()
)
_PLACE_ENTITIES = frozenset(("GPE", "LOC", "FAC"))
_NOT_SUBJECT_ENTITIES = frozenset(("DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "NORP"))
# WordNet's lexicographer files for things a clip can show; a word WordNet lacks (hoodie, pirateship) counts too.
_CONCRETE = frozenset(("noun.person", "noun.animal", "noun.artifact", "noun.plant", "noun.food", "noun.object"))
_CONCRETE_GROUPS = frozenset("people crowd family couple group creature animal army herd flock team band".split())
# Requests for writing rather than for a clip.
_NOT_VIDEO_OPENINGS = frozenset("write compose tell explain list summarize translate hello hi dear".split())


def clean(text: str) -> str | None:
    """The request with Pika's flags, a leading ``prompt:`` and runs of spaces removed, or None if it has markup."""
    text = _LEAD.sub("", _FLAGS.sub("", html.unescape(html.unescape(text))))
    if _MARKUP.search(text) or re.search(r"\n\s*\n", text):
        return None
    text = " ".join(text.split()).strip(" ,;")
    return text or None


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    tokens = words(text)
    return (
        bool(letters)
        and sum(c.isascii() for c in letters) >= 0.98 * len(letters)
        and sum(t in _ENGLISH for t in tokens) >= 0.15 * len(tokens)
    )


def is_keyword_list(text: str) -> bool:
    parts = [p for p in re.split(r"[,;]", text) if p.strip()]
    return len(parts) >= 5 and sum(len(p.split()) <= 3 for p in parts) >= 0.5 * len(parts)


def is_nsfw(text: str, scores: dict[str, float] | None = None) -> bool:
    return any(word in NSFW_WORDS for word in words(text)) or any(v >= MAX_TOXICITY for v in (scores or {}).values())


def passes_text_filters(text: str, min_words: int = MIN_WORDS) -> bool:
    return min_words <= len(text.split()) <= MAX_WORDS and is_english(text) and not is_keyword_list(text)


def template_keys(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tokens = words(text)
    return ("head", *tokens[:TEMPLATE_WORDS]), ("tail", *tokens[-TEMPLATE_WORDS:])


def persons(doc) -> list[tuple[str, ...]]:
    """The person names the tagger found, lower-cased, and each capitalized proper noun WordNet knows as a particular
    person (Columbus)."""
    names = [tuple(words(ent.text)) for ent in doc.ents if ent.label_ == "PERSON"]
    return names + [(t.lower_,) for t in doc if t.pos_ == "PROPN" and t.is_title and _is_person_name(t.lower_)]


@cache
def _is_person_name(word: str) -> bool:
    from nltk.corpus import wordnet

    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    return any(s.lexname() == "noun.person" and s.instance_hypernyms() for s in synsets)


def names_person(tokens: list[str], names: set[tuple[str, ...]]) -> bool:
    """Whether ``tokens`` contain one of ``names``, the multi-word names found anywhere in the pool."""
    longest = max((len(name) for name in names), default=0)
    return any(tuple(tokens[i : i + n]) in names for n in range(2, longest + 1) for i in range(len(tokens) - n + 1))


@cache
def is_concrete(lemma: str) -> bool:
    from nltk.corpus import wordnet

    synsets = wordnet.synsets(lemma.replace(" ", "_"), pos=wordnet.NOUN)[:2]
    return not synsets or lemma in _CONCRETE_GROUPS or any(s.lexname() in _CONCRETE for s in synsets)


def asks_for_writing(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and tokens[0] in _NOT_VIDEO_OPENINGS


def tag(doc) -> dict:
    """Subjects, setting and actions as spaCy reads them, and a suggested clip length in seconds."""
    subjects, setting, seen = [], None, set()
    for chunk in doc.noun_chunks:
        root = chunk.root
        lemma = root.lemma_.lower()
        if root.pos_ == "PRON" or lemma in _NOT_SUBJECTS or root.ent_type_ in _NOT_SUBJECT_ENTITIES:
            continue
        locative = root.dep_ == "pobj" and root.head.lower_ in _LOCATIVE
        if lemma in SETTING_NOUNS or root.ent_type_ in _PLACE_ENTITIES and locative:
            setting = setting or chunk.text
            continue
        if lemma not in seen and is_concrete(lemma):
            seen.add(lemma)
            subjects.append(chunk.text)
    actions = list(
        dict.fromkeys(
            t.lemma_.lower()
            for t in doc
            if t.pos_ == "VERB"
            and t.dep_ not in ("aux", "auxpass", "amod", "compound")
            and t.lemma_.lower() not in _NOT_ACTIONS
        )
    )
    sequence = sum(t.lower_ in _SEQUENCE_WORDS for t in doc)
    seconds = 2 + 2 * len(actions) + 2 * sequence + (len(doc.text.split()) - MIN_WORDS) // 25
    return {"subjects": subjects, "setting": setting, "actions": actions, "seconds": max(4, min(15, seconds))}


def is_rich(tags: dict) -> bool:
    return len(tags["subjects"]) >= 2 or bool(tags["subjects"] and tags["setting"] and tags["actions"])


def splits(tags: dict) -> dict[str, str]:
    seconds = tags["seconds"]
    n = len(tags["subjects"])
    return {
        "seconds": "4-7s" if seconds <= 7 else "8-11s" if seconds <= 11 else "12-15s",
        "subjects": "3+" if n >= 3 else str(n),
        "setting": "yes" if tags["setting"] else "no",
    }


class NearDuplicates:
    """Greedy near-duplicate check: MinHash bands find the kept requests a new one might copy, Jaccard decides."""

    _PRIME = (1 << 61) - 1

    def __init__(self, bands: int = 16, rows: int = 2):
        self.bands, self.rows = bands, rows
        seeds = [hashlib.sha256(f"minhash:{i}".encode()).digest() for i in range(bands * rows)]
        self.coefficients = [(int.from_bytes(s[:8]) | 1, int.from_bytes(s[8:16])) for s in seeds]
        self.buckets: dict[tuple, list[frozenset]] = {}

    @staticmethod
    def shingles(text: str) -> frozenset:
        tokens = words(text)
        return frozenset(zip(tokens, tokens[1:], tokens[2:], strict=False)) or frozenset([tuple(tokens)])

    def _keys(self, shingles: frozenset) -> list[tuple]:
        hashes = [int.from_bytes(hashlib.blake2b(" ".join(s).encode(), digest_size=8).digest()) for s in shingles]
        signature = [min((a * h + b) % self._PRIME for h in hashes) for a, b in self.coefficients]
        return [(band, *signature[band * self.rows : (band + 1) * self.rows]) for band in range(self.bands)]

    def add(self, text: str) -> bool:
        """Keep ``text`` and return True unless it is a near-duplicate of a request kept before."""
        shingles = self.shingles(text)
        keys = self._keys(shingles)
        for key in keys:
            for kept in self.buckets.get(key, ()):
                if len(shingles & kept) / len(shingles | kept) >= MAX_JACCARD:
                    return False
        for key in keys:
            self.buckets.setdefault(key, []).append(shingles)
        return True


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def load_vidprom(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 24)
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"id": row["uuid"], "raw": row["prompt"], "scores": {k: float(row[k]) for k in TOXICITY}})
    return rows


def load_moviegen(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [
            {"id": f"moviegen/{line:04d}", "raw": row["prompt"], "category": row["concept"].split(",")[0].strip()}
            | {"motion_level": row["motion_level"]}
            for line, row in enumerate(csv.DictReader(f))
        ]


def select(
    rows: list[dict], nlp, counts: Counter, templates: bool = True, processes: int = 1, min_words: int = MIN_WORDS
) -> list[dict]:
    """The rows that pass every filter but the near-duplicate check, each with its tags; ``counts`` gets the funnel."""
    counts["loaded"] += len(rows)
    kept = []
    for row in rows:
        text = clean(row["raw"])
        if text is None:
            counts["markup"] += 1
        elif not passes_text_filters(text, min_words):
            counts["length, language or keyword list"] += 1
        elif is_nsfw(text, row.get("scores")):
            counts["nsfw"] += 1
        elif asks_for_writing(text):
            counts["writing task"] += 1
        else:
            kept.append(row | {"request": text})
    if templates:
        shared = Counter(key for row in kept for key in set(template_keys(row["request"])))
        before = len(kept)
        kept = [row for row in kept if all(shared[key] < MAX_TEMPLATE_SHARE for key in template_keys(row["request"]))]
        counts["template"] += before - len(kept)
    read = []
    for row, doc in zip(
        kept, nlp.pipe((row["request"] for row in kept), batch_size=256, n_process=processes), strict=True
    ):
        read.append((row, persons(doc), tag(doc)))
    names = {name for _, found, _ in read for name in found if len(name) > 1}
    rich = []
    for row, found, tags in read:
        if found or names_person(words(row["request"]), names):
            counts["person name"] += 1
            continue
        if not is_rich(tags):
            counts["not rich"] += 1
            continue
        rich.append(row | tags)
    counts["rich"] += len(rich)
    return rich


def take(rows: list[dict], n: int, seed: int, counts: Counter) -> list[dict]:
    """The first ``n`` rows by ``seed``'s hash order that are not near-duplicates of one taken before."""
    taken, dedup = [], NearDuplicates()
    for row in sorted(rows, key=lambda row: _rank(seed, row["id"])):
        if len(taken) == n:
            break
        if dedup.add(row["request"]):
            taken.append(row)
        else:
            counts["near-duplicate"] += 1
    assert len(taken) == n, f"only {len(taken)} requests passed the filters, short of {n}"
    return taken


def _write(path: Path, rows: list[dict], source: str) -> None:
    with path.open("w") as f:
        for row in rows:
            metadata = {"source": source, "id": row["id"]}
            metadata |= {key: row[key] for key in ("subjects", "setting", "actions", "seconds")}
            metadata |= {key: row[key] for key in ("category", "motion_level") if key in row}
            metadata["splits"] = splits(row)
            message = {"role": "user", "content": row["request"]}
            f.write(json.dumps({"messages": [message], "metadata": metadata}) + "\n")


def build(
    vidprom: Path,
    moviegen: Path,
    out_dir: Path,
    nlp,
    train: int,
    heldout: int,
    n_moviegen: int,
    seed: int,
    processes=1,
):
    counts, moviegen_counts = Counter(), Counter()
    rows = select(load_vidprom(vidprom), nlp, counts, processes=processes)
    taken = take(rows, heldout + train, seed, counts)
    held, trained = taken[:heldout], taken[heldout:]
    movie = select(load_moviegen(moviegen), nlp, moviegen_counts, templates=False, min_words=MOVIEGEN_MIN_WORDS)
    movie = take(movie, n_moviegen, seed, moviegen_counts)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "train.jsonl", trained, "vidprom")
    _write(out_dir / "heldout.jsonl", held, "vidprom")
    _write(out_dir / "moviegen_heldout.jsonl", movie, "moviegenbench")
    (out_dir / "funnel.json").write_text(json.dumps({"vidprom": counts, "moviegenbench": moviegen_counts}, indent=1))
    return counts, moviegen_counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vidprom_csv", type=Path, help="VidProM's VidProM_semantic_unique.csv")
    parser.add_argument("moviegen_csv", type=Path, help="MovieGenBench's benchmark/MovieGenVideoBenchWithTag.csv")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--heldout", type=int, default=200)
    parser.add_argument("--moviegen", type=int, default=100, help="MovieGenBench's held-out requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spacy-model", default=SPACY_MODEL)
    parser.add_argument("--processes", type=int, default=1, help="spaCy worker processes")
    args = parser.parse_args(argv)
    import spacy

    nlp = spacy.load(args.spacy_model)
    counts, moviegen = build(
        args.vidprom_csv,
        args.moviegen_csv,
        args.out_dir,
        nlp,
        args.train,
        args.heldout,
        args.moviegen,
        args.seed,
        args.processes,
    )
    print(f"VidProM: {dict(counts)}\nMovieGenBench: {dict(moviegen)}")
    print(f"{args.train} train, {args.heldout} held-out and {args.moviegen} MovieGenBench requests in {args.out_dir}")


if __name__ == "__main__":
    main()
