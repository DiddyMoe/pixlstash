"""VRAM budget utilities for GPU memory-aware batch sizing.

"VRAM" is a discrete card's word. On Apple Silicon there is no card and no
separate pool: the GPU reads the same RAM the OS and every other application
is using. So on that hardware every figure here means **how much of the single
shared pool PixlStash may claim**, not how much of a dedicated device it owns,
and the numbers are deliberately conservative for that reason - overshooting on
a discrete card wastes the card, overshooting on unified memory swaps the whole
machine.
"""

import subprocess

from pixlstash.pixl_logging import get_logger
from pixlstash.utils.accelerator import (
    MPS,
    ALL_DEVICE_ERROR_WORDS,
    accelerator_total_memory_mb,
    empty_accelerator_cache,
    is_accelerated,
    is_apple_silicon,
)
from pixlstash.utils.model_utils import loaded_module

logger = get_logger(__name__)

#: Installed VRAM in MiB once ``nvidia-smi`` has reported a card; 0 until then.
#: See :func:`query_total_vram_mb`.
_card_total_mb: int = 0

#: How long ``nvidia-smi`` may take to report the card's memory. A healthy
#: driver answers in well under a second, even cold. A wedged one can block
#: indefinitely, and this read now sits on the settings request path.
NVIDIA_SMI_TIMEOUT_S = 10


def query_total_vram_mb() -> int:
    """Return the GPU memory this host can offer, in MiB.

    Two very different machines answer this:

    * **NVIDIA**: the sum of installed VRAM across all cards, from
      ``nvidia-smi``. Memory the GPU owns outright.
    * **Apple Silicon**: Metal's recommended working set, which is a *share of
      system RAM* (see
      :func:`~pixlstash.utils.accelerator.accelerator_total_memory_mb`).
      Nothing owns it; the OS, the browser and PixlStash all spend one pool.

    Returning 0 on a Mac - which is what the nvidia-smi-only version did - is
    not harmless. It sends ``system_utils.default_max_vram_gb`` to its 6 GB
    fallback and ``max_vram_budget_gb`` to 12 GB, figures that were invisible
    only while no Apple GPU was ever selected.

    The card's answer is kept once it is positive. ``nvidia-smi`` is a
    subprocess that brings the driver up on every run (54 ms measured on a WSL2
    development box), and ``GET /users/me/config`` asks on every request for
    the VRAM slider's bound; installed VRAM does not change under a running
    process. A 0 is not kept, because it can mean the driver was not up yet,
    and keeping it would hold the slider at its floor until a restart. The
    Apple answer is not kept either: it is a cheap read, and it sharpens from a
    share of RAM to Metal's own figure once torch is loaded.

    ``nvidia-smi`` gets :data:`NVIDIA_SMI_TIMEOUT_S` to answer. A driver that
    does not answer in that time counts as no card for this call and is asked
    again on the next one.

    Returns:
        Total GPU-usable memory in MiB, or 0 when there is no GPU to ask.
    """
    global _card_total_mb
    if is_apple_silicon():
        return accelerator_total_memory_mb(MPS)
    if _card_total_mb > 0:
        return _card_total_mb
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
        totals = []
        for line in output.splitlines():
            value = line.strip()
            if not value:
                continue
            totals.append(int(float(value)))
        _card_total_mb = sum(totals)
        return _card_total_mb
    except subprocess.TimeoutExpired:
        # Not the routine "no card" case below: a card whose driver is not
        # answering is worth a line in the log.
        logger.warning(
            "nvidia-smi did not report the card's memory within %d s; treating "
            "this host as having no card for now, and asking again next time.",
            NVIDIA_SMI_TIMEOUT_S,
        )
        return 0
    except Exception:
        # nvidia-smi absent/failing is normal on CPU-only hosts; 0 (no VRAM) IS
        # the documented answer, so logging it would be routine noise.
        return 0


def vram_limited_batch_cap(
    budget_mb: int | None,
    device: str,
    base_mb: int,
    per_item_mb: int,
) -> int:
    """Return the maximum batch size that fits within a VRAM budget.

    Args:
        budget_mb: Configured VRAM budget in MiB, or ``None`` for unlimited.
        device: Inference device string. Any accelerator (``"cuda"``, ``"mps"``)
            enables the cap; the CPU has no device memory to run out of.
        base_mb: Fixed model footprint in MiB (loaded once).
        per_item_mb: Incremental VRAM per image/item in MiB.

    Returns:
        Maximum item count that fits, or ``10_000`` when the cap is inactive.
    """
    if not is_accelerated(device) or not budget_mb:
        return 10_000
    reserve_mb = max(256, int(budget_mb * 0.20))
    task_budget_mb = max(1, budget_mb - reserve_mb)
    if task_budget_mb <= base_mb:
        return 1
    return max(1, int((task_budget_mb - base_mb) / max(1, per_item_mb)))


#: Metal's command-buffer failure, which is the *driver* running out rather
#: than torch's allocator. It says "OutOfMemory" as one word and never says
#: "out of memory", so the phrase check below cannot see it.
_METAL_COMMAND_BUFFER_OOM = "kiogpucommandbuffercallbackerroroutofmemory"

#: How far up the ``__cause__``/``__context__`` chain to look. A plugin that
#: wraps the driver's error in its own class is the common case; a chain deeper
#: than this is not.
_CAUSE_DEPTH = 5


def is_vram_oom(error: BaseException) -> bool:
    """True when *error* is an out-of-GPU-memory failure.

    Type identity is the reliable signal (``torch.OutOfMemoryError``), but a
    plugin may run its model through a runtime that raises its own exception
    type for the same condition, so the message is checked as well - and the
    wrapped-cause chain with it, because ``raise RuntimeError(...) from oom``
    is exactly how a plugin reports one. ``torch`` is read from
    :data:`sys.modules` for the same reason as in :func:`empty_device_cache`: a
    process that never imported it cannot have raised its OOM.

    Args:
        error: The exception to classify.

    Returns:
        ``True`` for a GPU OOM, which callers treat as transient and retry.
    """
    torch = loaded_module("torch")
    oom_type = getattr(torch, "OutOfMemoryError", None) if torch else None
    seen = set()
    current: BaseException | None = error
    for _ in range(_CAUSE_DEPTH):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        if isinstance(oom_type, type) and isinstance(current, oom_type):
            return True
        message = str(current).lower()
        if "cuda_error_out_of_memory" in message:
            return True
        if _METAL_COMMAND_BUFFER_OOM in message:
            return True
        # ONNX Runtime's BFC arena says neither "out of memory" nor a device
        # word when the card is full: "Failed to allocate memory for requested
        # buffer of size N" from bfc_arena.cc. Another process holding the
        # card (a local LLM, a ComfyUI graph) produces exactly this, and it is
        # as transient as torch's.
        if "failed to allocate memory for requested buffer" in message:
            return True
        # A device word, or SQLite's SQLITE_NOMEM "out of memory" would read as
        # transient GPU pressure. Apple's OOMs say "mps"/"metal", never "gpu".
        if "out of memory" in message and any(
            w in message for w in ALL_DEVICE_ERROR_WORDS
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def empty_device_cache(device=None) -> bool:
    """Flush the accelerator allocator's cache back to the driver.

    Kept as the name every recovery and teardown path in the codebase already
    imports; the accelerator-specific work is
    :func:`~pixlstash.utils.accelerator.empty_accelerator_cache`. It was called
    ``empty_cuda_cache`` and released CUDA's cache only, so on any other
    accelerator every caller's "free memory and retry" did nothing at all.

    Args:
        device: Flush only this accelerator's cache. ``None`` flushes whichever
            one this host has, which is what every teardown path wants. A
            spill-to-CPU path passes the device it is spilling *from*, because
            by the time it flushes it has already reassigned its own device to
            ``"cpu"`` and asking the host would flush the wrong allocator - or,
            on a host with two, the wrong one of the two.

    Returns:
        ``True`` if a cache was flushed, ``False`` when torch is not loaded or
        the host has no accelerator (callers use this to skip their own cache
        bookkeeping).
    """
    return empty_accelerator_cache(device)
