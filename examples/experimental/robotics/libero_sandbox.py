"""The ``libero-sim`` sandbox: its template on an E2B-compatible service (AgentENV), and one sandbox per episode.

The template is Ubuntu plus PEAR's own LIBERO installer at a pinned commit
(``scripts/install.sh libero``: LIBERO, robosuite and MuJoCo at PEAR's pins, a CPU-only
torch, conda-forge Mesa for llvmpipe rendering, and its own initial-observation check),
with ``libero_sim_server.py`` and ``START_SCRIPT`` written in by ``RUN`` (the service has
no ``COPY``). Its alias carries a digest of every build input, so an edit re-bakes.

    export E2B_API_URL=https://sandbox-service-control-plane.tail134ba0.ts.net
    export E2B_SANDBOX_URL=http://sandbox-service-control-plane
    python libero_sandbox.py build         # prints the alias; LIBERO_TEMPLATE selects another

Env vars:
  LIBERO_TEMPLATE                the template alias to start episodes from (default: this recipe's alias)
  LIBERO_SANDBOX_TTL_S           the sandbox timeout the keepalive re-arms (default 900)
  LIBERO_READY_TIMEOUT_S         how long the server may take to answer /health (default 300)
  LIBERO_CREATE_CONCURRENCY      creates in flight per rollout process (default 8)
  LIBERO_E2B_URL_SCHEME          the gateway's scheme (default http, AgentENV's data plane)
  LIBERO_E2B_PROXY               a proxy to the service, for hosts off its network (socks5h://... needs httpx[socks])
  E2B_API_URL / E2B_SANDBOX_URL / E2B_API_KEY[_FILE]   the service, read by the SDK and resolve_api_key
"""

import argparse
import base64
import hashlib
import os
import shlex
import sys
import threading
import time
from pathlib import Path

import httpx

from miles.rollout.agentic.credentials import resolve_provider_api_key

HERE = Path(__file__).resolve().parent

PEAR_URL = "https://github.com/mit-han-lab/pear.git"
PEAR_COMMIT = "f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087"
BASE_IMAGE = "ubuntu:24.04"
PEAR_ROOT = "/opt/pear"
PEAR_WORK = "/opt/pear-work"
SIM_DIR = "/opt/libero-sim"
SERVER_PORT = 8000
# built before the server answers /health: the imports and the first scene take longer than the gateway lets
# a request run (it answers 504), so a sandbox pays them while starting
WARM_EPISODE = '{"suite": "libero_goal", "task_id": 0, "trial_id": 0}'
CPU_COUNT = 2
MEMORY_MB = 4096

# The variables PEAR's LiberoEnv.setup_env gives a simulator process (Mesa's EGL software device, llvmpipe,
# one BLAS thread, no CUDA); LD_LIBRARY_PATH has to be set before Python starts, hence a script.
START_SCRIPT = r"""#!/bin/bash
set -euo pipefail
PEAR_ROOT=${PEAR_ROOT:-/opt/pear}
PEAR_WORK=${PEAR_WORK:-/opt/pear-work}
MESA=$PEAR_WORK/envs/mesa
export LD_LIBRARY_PATH=$MESA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=$MESA/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_DRIVERS_PATH=$MESA/lib/dri GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=${LP_NUM_THREADS:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export USER=${USER:-$(id -un)} LIBERO_CONFIG_PATH=$PEAR_WORK/envs/libero/etc/libero
export XDG_RUNTIME_DIR=${TMPDIR:-/tmp}/xdg_$USER
exec "$PEAR_WORK/envs/libero/bin/python" "$(dirname "$0")/libero_sim_server.py" --pear-root "$PEAR_ROOT" "$@"
"""


def _write_file_command(path: str, data: bytes, mode: str = "644") -> str:
    encoded = base64.b64encode(data).decode()
    return (
        f"mkdir -p {shlex.quote(str(Path(path).parent))} && echo {encoded} | base64 -d > {path} && chmod {mode} {path}"
    )


def build_commands() -> list[str]:
    server_source = (HERE / "libero_sim_server.py").read_bytes()
    return [
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "ca-certificates curl git patch tar coreutils libc-bin "
        # what the opencv-python LIBERO pulls in loads at import
        "libglib2.0-0t64 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*",
        f"git clone -q {PEAR_URL} {PEAR_ROOT} && git -C {PEAR_ROOT} checkout -q {PEAR_COMMIT}",
        f"USER=root WORK_DIR={PEAR_WORK} bash {PEAR_ROOT}/scripts/install.sh libero",
        _write_file_command(f"{SIM_DIR}/libero_sim_server.py", server_source),
        _write_file_command(f"{SIM_DIR}/start.sh", START_SCRIPT.encode(), mode="755"),
    ]


def template_alias() -> str:
    inputs = [BASE_IMAGE, *build_commands(), f"cpu={CPU_COUNT} memory_mb={MEMORY_MB}"]
    return f"libero-sim-{hashlib.sha256(chr(10).join(inputs).encode()).hexdigest()[:10]}"


def selected_template() -> str:
    return os.environ.get("LIBERO_TEMPLATE") or template_alias()


def resolve_api_key() -> str:
    return resolve_provider_api_key("E2B_API_KEY", "E2B_API_KEY_FILE", "~/.config/e2b/api_key")


def proxy() -> str | None:
    return os.environ.get("LIBERO_E2B_PROXY") or None


def connection_opts() -> dict:
    """What every SDK call takes: the key, and the proxy; the SDK reads neither from the proxy env vars."""
    return {"api_key": resolve_api_key(), "proxy": proxy()}


def build_template(*, force: bool = False, on_logs=None) -> str:
    """Build the template unless its alias exists; return the alias."""
    from e2b import Template  # the SDK is needed only where a sandbox is made

    alias = template_alias()
    if not force and Template.alias_exists(alias, **connection_opts()):
        return alias
    template = Template().from_image(BASE_IMAGE).set_user("root")
    for command in build_commands():
        template = template.run_cmd(command)
    Template.build(
        template,
        alias,
        cpu_count=CPU_COUNT,
        memory_mb=MEMORY_MB,
        skip_cache=force,
        on_build_logs=on_logs,
        **connection_opts(),
    )
    return alias


class Sandbox:
    """One episode's sandbox running the simulator server; ``kill`` it when the episode ends."""

    def __init__(self, *, template: str, max_lifetime_s: int, labels: dict[str, str]):
        from e2b import Sandbox as E2BSandbox

        self._opts = connection_opts()
        ttl_s = int(os.environ.get("LIBERO_SANDBOX_TTL_S", "900"))
        self._handle = E2BSandbox.create(
            template=template,
            # a server enforcing max_lifetime_secs refuses a longer timeout
            timeout=min(ttl_s, max_lifetime_s),
            metadata={**labels, "purpose": "libero-sim", "max_lifetime_secs": str(max_lifetime_s)},
            **self._opts,
        )
        self.sandbox_id = self._handle.sandbox_id
        self._alive = True
        try:
            self._handle.commands.run(
                f"{SIM_DIR}/start.sh --port {SERVER_PORT} --warm {shlex.quote(WARM_EPISODE)} > /tmp/libero-sim.log 2>&1",
                background=True,
                user="root",
            )
            scheme = os.environ.get("LIBERO_E2B_URL_SCHEME", "http")
            self.url = f"{scheme}://{self._handle.get_host(SERVER_PORT)}"
            wait_ready(self.url, timeout_s=float(os.environ.get("LIBERO_READY_TIMEOUT_S", "300")))
        except BaseException:
            self.kill()
            raise
        threading.Thread(
            target=self._keepalive, args=(ttl_s,), name=f"libero-keepalive-{self.sandbox_id}", daemon=True
        ).start()

    def log_tail(self, lines: int = 40) -> str:
        """The server's log, for an error report; empty when the sandbox cannot be reached."""
        try:
            return self._handle.commands.run(f"tail -n {lines} /tmp/libero-sim.log", user="root").stdout
        except Exception:
            return ""

    def kill(self) -> None:
        self._alive = False
        try:
            self._handle.kill(**self._opts)
        except Exception:
            pass  # the timeout and max lifetime reclaim it

    def _keepalive(self, ttl_s: int) -> None:
        """Re-arm the timeout while this process lives, so a killed caller's sandbox expires within ``ttl_s``."""
        failures = 0
        while self._alive and failures < 3:
            time.sleep(ttl_s / 6)
            try:
                if self._alive:
                    self._handle.set_timeout(ttl_s, **self._opts)
                failures = 0
            except Exception:
                failures += 1


def wait_ready(url: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=10, proxy=proxy()).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"the simulator server at {url} did not answer /health in {timeout_s:.0f}s: {last_error!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the template unless its alias exists")
    build.add_argument("--force", action="store_true", help="rebuild even when the alias exists")
    sub.add_parser("alias", help="print this recipe's alias")
    args = parser.parse_args()

    if args.command == "alias":
        print(template_alias())
        return 0
    started = time.monotonic()
    alias = build_template(force=args.force, on_logs=lambda entry: print(f"  | {entry}", flush=True))
    print(f"{alias} ready in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
