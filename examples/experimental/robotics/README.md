# Robotics: LIBERO episodes in PEAR's protocol

An agent function that plays [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) episodes the way [PEAR](https://github.com/mit-han-lab/pear) does, on the agentic stack Miles trains computer-use agents with: the session server records tokens, log-probs and frames, and each episode gets its own sandbox on the internal sandbox service (AgentENV).

PEAR (at `f32aa1f`) gives a vision-language model the robot's state as JSON and two 512×512 camera frames each turn; the model thinks, then ends its answer with one fenced json block of pose changes, which a closed-loop controller carries out on LIBERO's 20 Hz end-effector controller. The reward is 1 when the simulator ends the episode with the task done, else 0. This example reuses PEAR's simulator code (scene build, seed, initial states, stabilization, controller, success), its prompts, and its action checks; nothing in Miles core changes.

## Files

| File | What it is |
| --- | --- |
| `libero_agent_function.py` | `--custom-agent-function-path libero_agent_function.run` and `--custom-rm-path libero_agent_function.reward_func` |
| `libero_sandbox.py` | the `libero-sim` template recipe and build CLI, and the per-episode sandbox (TTL keepalive, max lifetime) |
| `libero_sim_server.py` | the simulator server inside the sandbox: `/reset`, `/act`, `/replay`, `/health` |
| `make_libero_data.py` | rows `{prompt, metadata: {suite, task_id, trial_id}}`, train trials 0–39 and held-out trials 40–49 |
| `libero_probe.py` | checks a server: reset reproducible across servers, stateful scene equal to PEAR's replay, timings |
| `libero_episodes.py` | `run()` on real sandboxes, with a scripted policy (no GPU) or a served model |

## How an episode runs

1. `run` starts a sandbox from the template (`LIBERO_TEMPLATE`, default the recipe's own alias) and waits until the server answers `/health`; the server imports LIBERO and builds one scene first, since a cold first reset takes longer than the sandbox gateway lets a request run.
2. `/reset` builds the row's episode: suite, task and trial (LIBERO's fixed initial state), PEAR's seed and its ten settling steps.
3. Each turn the policy gets PEAR's user message (state JSON, a reminder, `[agentview]` and `[wrist]` frames); its answer's last json block after `</think>` is checked as PEAR checks it and sent to `/act`. The scene stays open between actions; PEAR instead replays all native actions on every call, and `libero_probe.py` checks the two give identical state and frames.
4. The episode ends when the simulator ends it (task done, or the suite's native-step budget spent), an answer has no valid action (reward 0, as in PEAR), a reply hits its token cap, the next turn would not fit the context budget, the turn cap is reached, or the wall clock runs out. The sandbox is killed in every case.

A sandbox that does not start, or a simulator that cannot be reached or fails, raises `InfraAbort`, which discards the sample instead of training on it.

## Build the template

```bash
export E2B_API_URL=https://sandbox-service-control-plane.tail134ba0.ts.net
export E2B_SANDBOX_URL=http://sandbox-service-control-plane
export E2B_API_KEY_FILE=~/.config/e2b/api_key
python examples/experimental/robotics/libero_sandbox.py build   # prints libero-sim-<digest>
```

The build runs PEAR's own installer (`scripts/install.sh libero`), which ends with PEAR's check of one initial observation. The alias carries a digest of every build input, so editing the server or the recipe builds a new template.

## Check it without a GPU

```bash
python examples/experimental/robotics/libero_episodes.py --episodes 4
```

## Train

Use a CUA launcher's arguments with these changes: `--custom-agent-function-path libero_agent_function.run`, `--custom-rm-path libero_agent_function.reward_func`, rows from `make_libero_data.py` with `--input-key prompt --metadata-key metadata`, this directory on the rollout's and the agent worker's `PYTHONPATH`, and thinking on (`--apply-chat-template-kwargs '{"enable_thinking": true}'`) as in PEAR. A group is the `--n-samples-per-prompt` episodes of one row, so it shares task and initial state.

| Env var | Default | Meaning |
| --- | --- | --- |
| `LIBERO_TEMPLATE` | the recipe's alias | the template episodes start from |
| `LIBERO_MAX_TURNS` | 64 | actions per episode |
| `LIBERO_MAX_EPISODE_S` | 3600 | wall clock per episode |
| `LIBERO_CONTEXT_TOKENS` | `--max-seq-len`, else 32768 | context budget of an episode |
| `LIBERO_TURN_RESERVE_TOKENS` | 1024 | room kept for the next observation |
| `LIBERO_SIM_TIMEOUT_S` | 300 | one reset or action |
| `LIBERO_SANDBOX_TTL_S` | 900 | sandbox timeout the keepalive re-arms |
| `LIBERO_READY_TIMEOUT_S` | 300 | server start, imports and the first scene included |
| `LIBERO_CREATE_CONCURRENCY` | 8 | sandbox creates in flight per rollout process |
| `LIBERO_E2B_URL_SCHEME` | http | the sandbox gateway's scheme |

## Measured (2026-11-14, template at 2 vCPU / 4 GiB)

- On a 16-vCPU VM: reset 2.8–5.5 s, action 0.5 s, PEAR's replay of the same episode 3.1 s rising to 7.1 s at 300 native steps.
- In sandboxes, from an AWS VM, four at once: sandbox start 47–98 s, reset 8–13 s, action 2.1–2.3 s.

## Limits

- No episode here has succeeded: the scripted smoke policy does not solve a task, so reward 1 has been exercised only by the fast tests, on PEAR's own success rule.
- LIBERO renders on the CPU (llvmpipe); a sandbox service node's CPUs bound how many episodes run at once.
- PEAR's other environments (LIBERO-plus, LIBERO-PRO, RoboDojo, RoboLab) are not wired in; RoboDojo needs GPUs a CPU sandbox does not have.
