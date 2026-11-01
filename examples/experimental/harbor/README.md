# Harbor in-process on cloud sandboxes

This example runs [Harbor](https://github.com/harbor-framework/harbor) trials
**inside the rollout worker**: the agent function builds a `TrialConfig` and
calls `Trial.run()` directly, with the task sandbox on a cloud
[sandbox provider](../../../docs/user-guide/sandbox-providers.md) the worker
reaches over the network. There is no agent server.

Compared with [`examples/swe-agent-harbor-docker`](../../swe-agent-harbor-docker/README.md):

| | agent server (`swe-agent-harbor-docker`) | in-process (this example) |
| --- | --- | --- |
| Where `Trial.run()` runs | a separate host with a Docker daemon | the rollout worker |
| Sandbox backends | `docker` (local), `daytona` via the server's env | any cloud sandbox provider the worker can reach |
| Moving parts | trainer → HTTP → agent server → Harbor | trainer → Harbor |
| Use it when | tasks must run on the local Docker daemon | sandboxes are cloud-hosted |

Everything on the trainer side is the same: TITO, the session server, GRPO, the
reward hook (`generate.py` from the agent-server example).

## 1. Install Harbor in the rollout environment

Harbor now runs where the rollout workers run, so it goes into the Miles
image / environment. Use the `harbor-miles-v0.20.0` branch of
`harbor-framework/harbor` — the terminus-2 truncation policy it carries is
required for TITO (see the agent function's header for the full list) —
with the extra for your backend:

```bash
# uv, not pip: the branch carries a uv-workspace dependency pip cannot resolve
uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
# or harbor[daytona], harbor[modal], ...
```

## 2. Provision the sandbox backend

Credentials and endpoints for every provider are in
[Sandbox Providers](../../../docs/user-guide/sandbox-providers.md). The launch
command below uses E2B:

```bash
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key
```

Task directories: `HARBOR_TASKS_DIR` must contain one Harbor task dir per
`metadata.instance_id` in the training data (same as the agent-server example);
put it on a filesystem every worker can read.

**Network.** In-sandbox agents (mini-swe-agent, claude-code) call the model
from inside the sandbox, so the sandbox platform must reach the Miles session
server. `--session-server-external-host` is the address the sandboxes reach it
on: the driver builds the URL the agent gets from it and keeps the session
servers on the head node. On a multi-node job see the network note in
[the agent-server example](../../swe-agent-harbor-docker/README.md). Two port
ranges must route from the sandbox network: one
session-server port per worker starting at `--session-server-port` (30000-30031
for `run.py`'s 32 workers) and the SGLang router's 31000. Host-process agents
(terminus-2, computer-1) call the model from the worker and need no sandbox egress.

## 3. Prepare data

Same as the agent-server example:

```bash
python examples/swe-agent-harbor-docker/download_and_process_data.py \
    --input /path/to/terminal-bench.jsonl \
    --output /path/to/tb2_train.jsonl \
    --agent-name mini-swe-agent \
    --prompt-key instruction
```

## 4. Launch

```bash
HARBOR_ENV_TYPE=e2b python examples/experimental/harbor/run.py \
    --num-nodes 1 --num-gpus-per-node 8 --skip-prepare \
    --megatron-path /root/Megatron-LM \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --ref-load /path/to/GLM-4.7-Flash_torch_dist \
    --save-dir /path/to/checkpoints \
    --prompt-data /path/to/tb2_train.jsonl \
    --harbor-tasks-dir /path/to/harbor_tasks \
    --session-server-external-host <trainer-address-reachable-from-the-sandboxes> \
    --rollout-batch-size 4 --n-samples-per-prompt 8 --global-batch-size 32 \
    --num-rollout 200 --save-interval 10
```

`--save-dir` needs real headroom: a GLM-4.7-Flash torch_dist checkpoint with
optimizer state is several hundred GB, and the end-of-run save will fill
whatever is there.

`HARBOR_ENV_TYPE` has no default: the backend decides whose quota a run spends.
Backend-specific settings go in `HARBOR_ENV_KWARGS` as a JSON object (Harbor's
`EnvironmentConfig.kwargs`), e.g. `'{"auto_snapshot": true}'` for Daytona.

`HARBOR_OVERRIDE_CPUS`, `HARBOR_OVERRIDE_MEMORY_MB`, and
`HARBOR_OVERRIDE_STORAGE_MB` replace a task's declared resource values. They do
not change how Harbor enforces CPU or memory. Set
`HARBOR_CPU_ENFORCEMENT_POLICY` or `HARBOR_MEMORY_ENFORCEMENT_POLICY` to one of
Harbor's `auto`, `request`, `limit`, `guarantee`, or `ignore` modes when the
provider default is not the desired contract. For example, Modal CPU `auto` is
a hard request-and-limit pair; use `HARBOR_CPU_ENFORCEMENT_POLICY=request` to
keep the requested CPU guarantee while allowing burst capacity.

Every remaining knob — timeouts and their layering, failure semantics, the
full env-var reference — is documented in `harbor_agent_function.py`'s header,
next to the code that reads it.

Harbor's `computer-1` screenshot agent trains with Qwen3-VL through
`scripts/run_qwen3_vl_4b_harbor_cua.py --rollout-path session`, which passes
`--use-session-server v2 --use-session-images --tito-model qwen3_vl`.

## 5. Optional: run the trials on a CPU box

Host-process harnesses (`computer-1`, `terminus-2`) and their verifiers spend
the rollout host's CPUs, one Python process per trial. To move that work off the
GPU box, run Harbor's agent server (`miles_agent_server.py` on
`celve/harbor`'s `LIN-2510/run-the-cua-agent-harness-on-a` branch) on a CPU
machine and set `AGENT_SERVER_URL` for the rollout. The agent function still
builds each trial from its bindings and env vars here, sends it in `/run`'s
`trial_config`, and maps the returned `TrialResult` exactly as it would
in-process. Use `--custom-agent-function-mode inline`, since each call is now
one HTTP request.

On the CPU box, the server needs:

- `HARBOR_TASKS_DIR` with the same task dirs;
- the sandbox key (`E2B_API_KEY` or `E2B_API_KEY_FILE`, plus `E2B_API_URL`/`E2B_SANDBOX_URL` for a self-hosted service);
- every agent and verifier class the trial config names, importable on its `PYTHONPATH`.

The box must reach the session servers at the address the agents get:
`--session-server-external-host` (and `--session-server-port` for a fixed range).
The GPU box must reach `AGENT_SERVER_URL`. A `subproc` agent's dashboard reports
do not cross the HTTP hop.

## Validation

This README's command has been run end to end: 8×H200, real training mode with
the batch dials reduced, both trials scoring reward 1.0 and one GRPO step
completed. Which sandbox providers this path has been run on is the table in
[Sandbox Providers](../../../docs/user-guide/sandbox-providers.md).
