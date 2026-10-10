"""Static deployment contract tests; no Docker daemon or GPU required."""

from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_pins_compatible_runtime_and_runs_non_root():
    dockerfile = read("Dockerfile")
    assert "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:" in dockerfile
    assert "python -m venv --system-site-packages /opt/venv" in dockerfile
    assert 'PATH="/opt/venv/bin:${PATH}"' in dockerfile
    assert "--break-system-packages" not in dockerfile
    assert "--constraint constraints-container.txt accelerate" in dockerfile
    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "Path(torch.__file__).is_relative_to('/opt/venv')" in dockerfile
    assert "torch.version.cuda == '12.8'" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert '"--lifespan", "on"' in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["/opt/venv/bin/python", "-c"' in dockerfile
    assert 'CMD ["/opt/venv/bin/python", "-m", "uvicorn"' in dockerfile
    assert "/health" in dockerfile
    assert "/v1/chat/completions" not in dockerfile


def test_dockerfile_installs_matching_venv_package_as_root_before_creation():
    dockerfile = read("Dockerfile")
    root_index = dockerfile.index("USER root")
    update_index = dockerfile.index("apt-get update")
    install_index = dockerfile.index(
        "apt-get install -y --no-install-recommends python3.12-venv"
    )
    cleanup_index = dockerfile.index("rm -rf /var/lib/apt/lists/*")
    venv_index = dockerfile.index("python -m venv --system-site-packages /opt/venv")
    runtime_user_index = dockerfile.index("USER 10001:10001")

    assert root_index < update_index < install_index < cleanup_index < venv_index
    assert venv_index < runtime_user_index


def test_compose_requests_one_gpu_cache_volume_and_graceful_stop():
    compose = read("compose.yaml")
    assert '"127.0.0.1:${MINILLM_HOST_PORT:-8000}:8000"' in compose
    assert '"${MINILLM_HOST_PORT:-8000}:8000"' not in compose
    assert "driver: nvidia" in compose
    assert "count: 1" in compose
    assert "capabilities: [gpu]" in compose
    assert "hf-cache:/home/minillm/.cache/huggingface" in compose
    assert "stop_signal: SIGTERM" in compose
    assert "stop_grace_period: 5m" in compose
    assert 'MINILLM_MAX_CONCURRENT_GENERATIONS: "${MINILLM_MAX_CONCURRENT_GENERATIONS:-1}"' in compose


def test_build_context_excludes_secrets_caches_and_model_weights():
    ignore = read(".dockerignore")
    for pattern in (
        ".git",
        ".venv",
        ".env",
        ".huggingface",
        "benchmarks",
        "**/*.safetensors",
        "**/*.bin",
        "**/*.gguf",
    ):
        assert pattern in ignore
    assert "!.env.example" in ignore
    gitignore = read(".gitignore")
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_container_constraints_match_verified_core_versions():
    constraints = read("constraints-container.txt")
    for requirement in (
        "torch==2.11.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "fastapi==0.141.1",
        "uvicorn==0.52.4",
        "pydantic==2.13.5",
        "click>=8,<8.4,!=8.3.0",
    ):
        assert requirement in constraints


def test_container_click_constraint_satisfies_spin_and_uvicorn():
    constraints = read("constraints-container.txt").splitlines()
    click_line = next(line for line in constraints if line.lower().startswith("click"))
    specifier = Requirement(click_line).specifier

    assert "8.0.0" in specifier
    assert "8.2.1" in specifier
    assert "8.3.0" not in specifier
    assert "8.4.0" not in specifier
    assert "8.5.0" not in specifier

    dockerfile = read("Dockerfile")
    assert "--constraint constraints-container.txt" in dockerfile
    assert "python -m pip check" in dockerfile


def test_ci_is_cpu_only_and_does_not_run_slow_model_tests():
    workflow = read(".github/workflows/ci.yml")
    assert "whl/cpu" in workflow
    assert 'pytest -m "not slow"' in workflow
    assert "--gpus" not in workflow
