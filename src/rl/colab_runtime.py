"""Colab/runtime helpers for bounded PPO artifact and evaluation execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path
import resource
import sys
from typing import Any, Iterator, Mapping

from core.logging import get_logger
from rl.colab_staging_closure import (
    CLOSURE_REPORT_FILENAME,
    RUNTIME_DEPENDENCY_REPORT_FILENAME,
    STAGING_MANIFEST_FILENAME,
    stage_dependency_closure,
    validate_existing_stage,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Machine-readable runtime environment snapshot."""

    colab_detected: bool
    notebook_detected: bool
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_memory_bytes: int | None
    gpu_total_memory_gib: float | None
    system_ram_bytes: int | None
    system_ram_gib: float | None
    process_rss_bytes: int | None
    process_rss_mib: float | None
    torch_compile_supported: bool


def collect_runtime_environment() -> RuntimeEnvironment:
    """Collect a narrow runtime snapshot for Colab/local execution reporting."""

    torch_module, _ = _optional_import("torch")
    torch_available = torch_module is not None
    torch_version = getattr(torch_module, "__version__", None) if torch_module is not None else None
    cuda_available = bool(torch_available and bool(torch_module.cuda.is_available()))
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    gpu_total_memory_gib: float | None = None
    if torch_module is not None and cuda_available:
        try:
            current_device = int(torch_module.cuda.current_device())
            properties = torch_module.cuda.get_device_properties(current_device)
            gpu_name = str(properties.name)
            gpu_total_memory_bytes = int(properties.total_memory)
            gpu_total_memory_gib = round(float(gpu_total_memory_bytes) / float(1024**3), 3)
        except Exception:  # noqa: BLE001
            gpu_name = None
            gpu_total_memory_bytes = None
            gpu_total_memory_gib = None

    system_ram_bytes = _resolve_system_ram_bytes()
    system_ram_gib = round(float(system_ram_bytes) / float(1024**3), 3) if system_ram_bytes is not None else None
    process_rss_bytes = _resolve_process_rss_bytes()
    process_rss_mib = round(float(process_rss_bytes) / float(1024**2), 3) if process_rss_bytes is not None else None
    colab_detected = _module_exists("google.colab")
    notebook_detected = _notebook_detected()
    torch_compile_supported = bool(torch_module is not None and callable(getattr(torch_module, "compile", None)))

    return RuntimeEnvironment(
        colab_detected=colab_detected,
        notebook_detected=notebook_detected,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
        gpu_total_memory_gib=gpu_total_memory_gib,
        system_ram_bytes=system_ram_bytes,
        system_ram_gib=system_ram_gib,
        process_rss_bytes=process_rss_bytes,
        process_rss_mib=process_rss_mib,
        torch_compile_supported=torch_compile_supported,
    )


def capture_memory_snapshot(*, label: str, step: int | None = None) -> dict[str, Any]:
    """Capture one bounded memory snapshot for machine-readable reporting."""

    environment = collect_runtime_environment()
    torch_module, _ = _optional_import("torch")
    gpu_allocated_bytes: int | None = None
    gpu_reserved_bytes: int | None = None
    if torch_module is not None and bool(environment.cuda_available):
        try:
            gpu_allocated_bytes = int(torch_module.cuda.memory_allocated())
            gpu_reserved_bytes = int(torch_module.cuda.memory_reserved())
        except Exception:  # noqa: BLE001
            gpu_allocated_bytes = None
            gpu_reserved_bytes = None

    return {
        "label": label,
        "step": int(step) if step is not None else None,
        "captured_at_utc": _generated_at(),
        "process_rss_bytes": environment.process_rss_bytes,
        "process_rss_mib": environment.process_rss_mib,
        "gpu_allocated_bytes": gpu_allocated_bytes,
        "gpu_allocated_mib": _bytes_to_mib(gpu_allocated_bytes),
        "gpu_reserved_bytes": gpu_reserved_bytes,
        "gpu_reserved_mib": _bytes_to_mib(gpu_reserved_bytes),
    }


@contextmanager
def numpy_fail_fast_context() -> Iterator[dict[str, Any]]:
    """Apply numpy fail-fast floating point behavior and restore it afterward."""

    numpy_module, numpy_error = _optional_import("numpy")
    metadata: dict[str, Any] = {
        "numpy_available": numpy_module is not None,
        "numpy_import_error": numpy_error,
        "applied": False,
        "requested_policy": {"all": "raise"},
        "previous_policy": None,
    }
    if numpy_module is None:
        yield metadata
        return

    previous_policy = dict(numpy_module.geterr())
    numpy_module.seterr(all="raise")
    metadata["applied"] = True
    metadata["previous_policy"] = previous_policy
    try:
        yield metadata
    finally:
        numpy_module.seterr(**previous_policy)


def ensure_model_parameters_finite(model: Any) -> None:
    """Fail fast when the trained PPO model contains non-finite parameters."""

    torch_module, _ = _optional_import("torch")
    if torch_module is None:
        return

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "parameters"):
        return

    for index, parameter in enumerate(policy.parameters()):
        try:
            finite_mask = torch_module.isfinite(parameter.detach())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to validate model parameter finiteness at index {index}") from exc
        if not bool(finite_mask.all().item()):
            raise ValueError(f"Non-finite model parameter detected at index {index}")


def validate_finite_scalar(*, name: str, value: Any) -> float:
    """Return a float value only when it is finite; otherwise raise clearly."""

    numeric_value = float(value)
    if not _is_finite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def stage_explicit_inputs(*, staging_root: Path, source_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Stage a closure-complete explicit input surface into a fresh Colab root."""

    LOGGER.info(
        "Colab staging start | staging_root=%s staged_labels=%s",
        staging_root.resolve(),
        sorted(str(key) for key in source_paths),
    )
    manifest_payload = stage_dependency_closure(
        staging_root=staging_root.resolve(),
        source_paths={str(key): Path(value).resolve() for key, value in source_paths.items()},
    )
    LOGGER.info(
        "Colab staging completed | staging_root=%s manifest_path=%s closure_report_path=%s runtime_dependency_report_path=%s",
        staging_root.resolve(),
        staging_root.resolve() / STAGING_MANIFEST_FILENAME,
        staging_root.resolve() / CLOSURE_REPORT_FILENAME,
        staging_root.resolve() / RUNTIME_DEPENDENCY_REPORT_FILENAME,
    )
    return manifest_payload


def validate_staged_preflight(*, staging_root: Path) -> dict[str, Any]:
    """Revalidate the shared staged preflight reports for a staged Colab root."""

    return validate_existing_stage(staging_root=staging_root.resolve())


def _resolve_system_ram_bytes() -> int | None:
    """Return total system RAM bytes when available."""

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None
    return page_size * phys_pages


def _resolve_process_rss_bytes() -> int | None:
    """Return current process RSS bytes when available."""

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kib = int(usage.ru_maxrss)
    except (ValueError, OSError):
        return None
    if sys.platform == "darwin":
        return rss_kib
    return rss_kib * 1024


def _optional_import(module_name: str) -> tuple[Any | None, str | None]:
    """Import one optional module without hard-failing the caller."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return module, None


def _module_exists(module_name: str) -> bool:
    """Return whether a module import spec is available."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _notebook_detected() -> bool:
    """Return whether the current runtime looks like an IPython notebook kernel."""

    ipython_module, _ = _optional_import("IPython")
    if ipython_module is None:
        return False
    try:
        shell = ipython_module.get_ipython()
    except Exception:  # noqa: BLE001
        return False
    return shell is not None and getattr(shell, "kernel", None) is not None


def _sha256_file(path: Path) -> str:
    """Return the SHA256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite(value: float) -> bool:
    """Return whether a float is finite without importing math globally."""

    return not (value != value or value in {float("inf"), float("-inf")})


def _bytes_to_mib(value: int | None) -> float | None:
    """Convert bytes into MiB when available."""

    if value is None:
        return None
    return round(float(value) / float(1024**2), 3)


def _generated_at() -> str:
    """Return a UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()
