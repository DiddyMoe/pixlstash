"""A PixlStash tagger that cannot load waits before it is loaded again.

Its load failure used to log "disabling" and disable nothing: every tagging
batch, quality pass and anomaly-region request loaded it again from disk.
"""

import logging
from types import SimpleNamespace

from pixlstash.inference.model_lifecycle import ModelLifecycleManager
from pixlstash.tagger_plugins import pixlstash_tagger as tagger_module
from pixlstash.tagger_plugins.pixlstash_tagger import (
    PixlStashTaggerService,
    TaggerLoadFailure,
    TaggerUnavailableError,
)


def _service(tmp_path, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(
        tagger_module,
        "time",
        SimpleNamespace(monotonic=lambda: now["t"]),
        raising=False,
    )
    # An empty model folder: the load fails on the missing checkpoint.
    service = PixlStashTaggerService(
        device="cpu", model_dir=str(tmp_path), batch_size_fn=lambda: 2
    )
    loads = []
    real_load = service._init_locked

    def counted():
        loads.append(now["t"])
        real_load()

    monkeypatch.setattr(service, "_init_locked", counted)
    return service, now, loads


def test_a_tagger_that_cannot_load_is_not_loaded_again_per_batch(
    tmp_path, monkeypatch, caplog
):
    service, now, loads = _service(tmp_path, monkeypatch)
    base = tagger_module.LOAD_RETRY_BASE_S

    with caplog.at_level(logging.WARNING, logger=tagger_module.logger.name):
        assert service.init_or_cpu_fallback() is False
    assert len(loads) == 1
    failure = service.load_failure()
    assert failure is not None
    assert failure.failures == 1
    assert failure.retry_in_s == base
    assert failure.error == "FileNotFoundError"
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "FileNotFoundError" in warned[0].getMessage()
    assert f"{base:.0f} s" in warned[0].getMessage(), warned[0].getMessage()

    # Every batch until then is turned away without touching the disk.
    for _ in range(5):
        assert service.init_or_cpu_fallback() is False
    assert len(loads) == 1

    now["t"] += base - 1
    assert service.init_or_cpu_fallback() is False
    assert len(loads) == 1
    assert service.load_failure().retry_in_s == 1

    now["t"] += 1
    assert service.init_or_cpu_fallback() is False
    assert len(loads) == 2, "tried again once the wait is over"
    assert service.load_failure().failures == 2
    assert service.load_failure().retry_in_s == 2 * base


def test_the_wait_between_loads_is_capped(tmp_path, monkeypatch):
    service, now, loads = _service(tmp_path, monkeypatch)
    waits = []
    for _ in range(12):
        assert service.init_or_cpu_fallback() is False
        wait = service.load_failure().retry_in_s
        waits.append(wait)
        now["t"] += wait
    assert len(loads) == 12
    assert waits == sorted(waits)
    assert max(waits) == tagger_module.LOAD_RETRY_MAX_S


def test_a_load_that_succeeds_clears_the_failure(tmp_path, monkeypatch):
    service, now, loads = _service(tmp_path, monkeypatch)
    assert service.init_or_cpu_fallback() is False
    now["t"] += service.load_failure().retry_in_s

    def loaded():
        loads.append(now["t"])
        service._model = object()
        service._labels = ["a"]
        service._transform = object()

    monkeypatch.setattr(service, "_init_locked", loaded)
    assert service.init_or_cpu_fallback() is True
    assert service.load_failure() is None

    # A later failure starts from the first wait again.
    service.unload()
    monkeypatch.setattr(
        service, "_init_locked", lambda: (_ for _ in ()).throw(OSError("gone"))
    )
    assert service.init_or_cpu_fallback() is False
    assert service.load_failure().failures == 1
    assert service.load_failure().retry_in_s == tagger_module.LOAD_RETRY_BASE_S


def test_every_tagging_call_is_refused_while_the_tagger_waits(tmp_path, monkeypatch):
    # The path every tagging batch, quality pass and anomaly request takes. It
    # must answer "not loaded" on every call (TaggingWorkflow then raises), not
    # once and then report a tagger that tags nothing.
    service, now, loads = _service(tmp_path, monkeypatch)
    lifecycle = ModelLifecycleManager("cpu")

    def ready():
        return lifecycle.ensure_tagging_ready(
            None, service, use_wd14=False, use_pixlstash_tagger=True
        )

    assert [ready() for _ in range(4)] == [False] * 4
    assert len(loads) == 1


def test_the_unavailable_error_says_why_and_when():
    failure = TaggerLoadFailure(failures=3, retry_in_s=119.6, error="OSError")
    error = TaggerUnavailableError(failure)
    assert isinstance(error, RuntimeError)
    assert error.failure is failure
    assert str(error) == (
        "The PixlStash tagger could not be loaded (OSError, 3 attempts in a "
        "row); it is tried again in 120 s."
    )
    first = TaggerUnavailableError(
        TaggerLoadFailure(failures=1, retry_in_s=30.0, error="OSError")
    )
    assert str(first) == (
        "The PixlStash tagger could not be loaded (OSError, 1 failed attempt); "
        "it is tried again in 30 s."
    )
    assert str(TaggerUnavailableError(None)) == "The PixlStash tagger is not loaded."
