#!/bin/bash
#
# One verifier for every harbor_cua task: read the browser window's title for how many
# stages the page marked done, and compare the agent's final answer when the task has
# one. Both are state the page itself wrote, so no judge is involved.
set -u

mkdir -p /logs/verifier
export DISPLAY="${DISPLAY:-:1}"

STAGES_TOTAL="${STAGES_TOTAL:-0}"
EXPECTED_ANSWER="${EXPECTED_ANSWER:-}"

titles="$(wmctrl -l 2>/dev/null || true)"
stages_done="$(printf '%s\n' "$titles" | sed -n 's/.*CUA-PROGRESS \([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p' | sort -rn | head -1)"
stages_done="${stages_done:-0}"

answer=""
if [ -f /logs/agent/final_answer.txt ]; then
    answer="$(tr -d '\r' < /logs/agent/final_answer.txt)"
fi

answer_points=0
answer_total=0
if [ -n "$EXPECTED_ANSWER" ]; then
    answer_total=1
    if printf '%s' "$answer" | grep -qiF -- "$EXPECTED_ANSWER"; then
        answer_points=1
    fi
fi

STAGES_DONE="$stages_done" \
STAGES_TOTAL="$STAGES_TOTAL" \
ANSWER_POINTS="$answer_points" \
ANSWER_TOTAL="$answer_total" \
ANSWER="$answer" \
TITLES="$titles" \
python3 - <<'PY'
import json
import os

stages_done = min(int(os.environ["STAGES_DONE"]), int(os.environ["STAGES_TOTAL"]))
earned = stages_done + int(os.environ["ANSWER_POINTS"])
possible = int(os.environ["STAGES_TOTAL"]) + int(os.environ["ANSWER_TOTAL"])
reward = earned / possible if possible else 0.0

with open("/logs/verifier/reward.json", "w", encoding="utf-8") as handle:
    json.dump({"reward": reward}, handle, indent=2)
with open("/logs/verifier/report.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "reward": reward,
            "stages_done": stages_done,
            "stages_total": int(os.environ["STAGES_TOTAL"]),
            "answer_points": int(os.environ["ANSWER_POINTS"]),
            "answer_total": int(os.environ["ANSWER_TOTAL"]),
            "final_answer": os.environ["ANSWER"],
            "window_titles": os.environ["TITLES"].splitlines(),
        },
        handle,
        indent=2,
    )
print(f"reward={reward} stages={stages_done}/{os.environ['STAGES_TOTAL']}")
PY
