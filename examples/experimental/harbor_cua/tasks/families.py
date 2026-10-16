"""The narrow CUA task set: five families, each parameterised into instances.

The point of the parameters is a disclosed train/evaluation split that does not
share instances: a family's variant list is cut at a fixed index, so no training
instance appears in evaluation. The families themselves are shared across the
split, which is a real limitation and is why this set measures plumbing and
short-horizon control rather than generalisation.

Every task is one browser page on a KVM-free container desktop, graded from the
window title the page itself writes, so a reward is a state read rather than a
judge's opinion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PAGE_PLACEHOLDER = "page.html"
TASK_GROUP = "cua-basics"

_COLORS = [
    ("red", "#dc2626"),
    ("blue", "#2563eb"),
    ("green", "#16a34a"),
    ("orange", "#ea580c"),
    ("purple", "#7c3aed"),
    ("teal", "#0d9488"),
]
_SLOTS = [(60, 40), (420, 40), (780, 40), (60, 300), (420, 300), (780, 300), (60, 560), (420, 560), (780, 560)]
_WORDS = ["harbor", "anchor", "rudder", "beacon", "lantern", "compass", "ballast", "mooring"]
_LABEL_SETS = [
    ("Apply", ["Cancel", "Apply", "Reset", "Help"]),
    ("Publish", ["Draft", "Publish", "Archive", "Delete"]),
    ("Connect", ["Disconnect", "Retry", "Connect", "Settings"]),
    ("Confirm", ["Back", "Skip", "Confirm", "Abort"]),
    ("Upload", ["Download", "Upload", "Share", "Rename"]),
    ("Enable", ["Disable", "Inspect", "Enable", "Reload"]),
]
_STEP_PAIRS = [
    ("Start", "Finish"),
    ("Begin", "Complete"),
    ("Open", "Close"),
    ("Unlock", "Submit"),
    ("Prepare", "Send"),
    ("Select", "Apply"),
]
_CODES = ["K7QM", "R2XP", "T9BD", "N4HJ", "V6LW", "C3ZF", "M8KY", "P5RQ"]


@dataclass(frozen=True)
class Variant:
    key: str
    substitutions: dict[str, str]
    stages_total: int
    expected_answer: str | None = None


@dataclass(frozen=True)
class Family:
    name: str
    page: str
    instruction: str
    max_turns: int
    variants: tuple[Variant, ...]


def _click_target_variants() -> tuple[Variant, ...]:
    variants = []
    for index, (name, hex_code) in enumerate(_COLORS):
        target = _SLOTS[index % len(_SLOTS)]
        decoys = [
            [other_hex, *_SLOTS[(index + offset) % len(_SLOTS)]]
            for offset, (_, other_hex) in enumerate(_COLORS, start=1)
            if other_hex != hex_code
        ][:4]
        variants.append(
            Variant(
                key=name,
                substitutions={
                    "COLOR_NAME": name,
                    "COLOR_HEX": hex_code,
                    "TARGET_X": str(target[0]),
                    "TARGET_Y": str(target[1]),
                    "DECOYS": json.dumps(decoys),
                },
                stages_total=1,
            )
        )
    return tuple(variants)


def _type_text_variants() -> tuple[Variant, ...]:
    return tuple(Variant(key=word, substitutions={"WORD": word}, stages_total=1) for word in _WORDS)


def _pick_button_variants() -> tuple[Variant, ...]:
    return tuple(
        Variant(
            key=label.lower(),
            substitutions={"LABEL": label, "LABELS": json.dumps(labels)},
            stages_total=1,
        )
        for label, labels in _LABEL_SETS
    )


def _two_step_variants() -> tuple[Variant, ...]:
    return tuple(
        Variant(key=first.lower(), substitutions={"FIRST": first, "SECOND": second}, stages_total=2)
        for first, second in _STEP_PAIRS
    )


def _read_code_variants() -> tuple[Variant, ...]:
    return tuple(
        Variant(key=code.lower(), substitutions={"CODE": code}, stages_total=0, expected_answer=code)
        for code in _CODES
    )


FAMILIES: tuple[Family, ...] = (
    Family(
        name="click-target",
        page="click_target.html",
        instruction=(
            "Open the browser and navigate to `file:///app/page.html`.\n"
            "The page shows five coloured squares. Click the __COLOR_NAME__ one.\n"
            "The square turns green when you have clicked it. Then finish with the `done` action."
        ),
        max_turns=8,
        variants=_click_target_variants(),
    ),
    Family(
        name="type-text",
        page="type_text.html",
        instruction=(
            "Open the browser and navigate to `file:///app/page.html`.\n"
            "Click into the text box, type the word `__WORD__` exactly, and press Enter.\n"
            "The page reports `Done.` when it accepts the word. Then finish with the `done` action."
        ),
        max_turns=8,
        variants=_type_text_variants(),
    ),
    Family(
        name="pick-button",
        page="pick_button.html",
        instruction=(
            "Open the browser and navigate to `file:///app/page.html`.\n"
            "Four buttons are shown. Click the one labelled `__LABEL__` and no other.\n"
            "The page reports `Done.` when you click the right one. Then finish with the `done` action."
        ),
        max_turns=8,
        variants=_pick_button_variants(),
    ),
    Family(
        name="two-step",
        page="two_step.html",
        instruction=(
            "Open the browser and navigate to `file:///app/page.html`.\n"
            "Click the `__FIRST__` button. A `__SECOND__` button then appears below it; click that too.\n"
            "The page reports `Done.` after both. Then finish with the `done` action."
        ),
        max_turns=10,
        variants=_two_step_variants(),
    ),
    Family(
        name="read-code",
        page="read_code.html",
        instruction=(
            "Open the browser and navigate to `file:///app/page.html`.\n"
            "A four-character code is printed on the page. Read it, then finish with the `done` action\n"
            "and give the code as your result."
        ),
        max_turns=8,
        variants=_read_code_variants(),
    ),
)


def instance_id(family: Family, variant: Variant) -> str:
    return f"{TASK_GROUP}--{family.name}--{variant.key}"


def split_variants(family: Family, *, train: int, evaluation: int) -> tuple[list[Variant], list[Variant]]:
    """Disjoint instance lists, cut at a fixed index so a rerun splits the same way."""
    available = len(family.variants)
    if train + evaluation > available:
        raise ValueError(
            f"family {family.name} has {available} variants, fewer than the "
            f"{train} training plus {evaluation} evaluation instances asked for"
        )
    return list(family.variants[:train]), list(family.variants[train : train + evaluation])
