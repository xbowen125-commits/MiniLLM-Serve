"""Explicit correctness-only host with request/owner identity collection."""
import json
from pathlib import Path
import sys


def main():
    from . import server
    from .artifacts import git_snapshot
    from .config import load_config
    from .runtime_trace import RuntimeTrace
    from .identity_trace import IdentityObserver
    from .trace_check import seal
    from .preflight_server import install_observers
    directory = Path(sys.argv[sys.argv.index("--trial-dir") + 1])
    config = load_config(Path(sys.argv[sys.argv.index("--config") + 1]))
    source = {k: v for k, v in git_snapshot().items() if k != "files"}
    # Geometry is explicitly unobserved here; no guessed dimensions are used.
    identity = {k: config.data[k] for k in ("model_revision", "dtype", "device")}
    identity.update(run_id=directory.parent.name, config_sha256=config.sha256, git=source,
                    geometry={"status": "unobserved_by_identity_host"})
    trace = RuntimeTrace(directory, identity).install()
    IdentityObserver(trace).install()
    journal = install_observers(directory)
    try:
        server.main()
    finally:
        journal.close()
        trace.close()
        seal(directory)


if __name__ == "__main__":
    main()
