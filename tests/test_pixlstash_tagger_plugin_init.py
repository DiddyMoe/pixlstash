"""``PixlStashTaggerPlugin.init`` loads the way the tagging workflow does:
it honours the wait after a failed load, and any load that
succeeds clears the failure ``load_failure()`` reports.
"""

from types import SimpleNamespace

import pytest

from pixlstash.tagger_plugins import pixlstash_tagger as tagger_module
from pixlstash.tagger_plugins.pixlstash_tagger import (
    PixlStashTaggerPlugin,
    PixlStashTaggerService,
    TaggerUnavailableError,
)


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(
        tagger_module, "time", SimpleNamespace(monotonic=lambda: now["t"])
    )
    return now


def _service(tmp_path, monkeypatch):
    # An empty model folder: a real load fails on the missing checkpoint.
    service = PixlStashTaggerService(
        device="cpu", model_dir=str(tmp_path), batch_size_fn=lambda: 2
    )
    loads = []
    real_load = service._init_locked

    def counted():
        loads.append(1)
        real_load()

    monkeypatch.setattr(service, "_init_locked", counted)
    return service, loads


def _loads_fine(service, loads):
    def loaded():
        loads.append(1)
        service._model = object()
        service._labels = ["a"]
        service._transform = object()

    return loaded


def test_the_plugin_init_waits_out_a_failed_load(tmp_path, monkeypatch, clock):
    service, loads = _service(tmp_path, monkeypatch)
    plugin = PixlStashTaggerPlugin()
    plugin.bind_service(service)

    with pytest.raises(TaggerUnavailableError, match="FileNotFoundError"):
        plugin.init({})
    assert len(loads) == 1
    assert service.load_failure().failures == 1

    # Within the wait the plugin check is turned away without touching disk.
    with pytest.raises(TaggerUnavailableError):
        plugin.init({})
    assert len(loads) == 1, "the plugin's init loaded again inside the wait"


def test_a_plugin_init_that_loads_clears_the_failure(tmp_path, monkeypatch, clock):
    service, loads = _service(tmp_path, monkeypatch)
    plugin = PixlStashTaggerPlugin()
    plugin.bind_service(service)
    assert service.init_or_cpu_fallback() is False
    clock["t"] += service.load_failure().retry_in_s

    monkeypatch.setattr(service, "_init_locked", _loads_fine(service, loads))
    plugin.init({})
    assert service.is_loaded()
    assert service.load_failure() is None, "loaded, yet still reported failing"


def test_a_direct_service_load_clears_the_failure(tmp_path, monkeypatch, clock):
    service, loads = _service(tmp_path, monkeypatch)
    assert service.init_or_cpu_fallback() is False
    assert service.load_failure() is not None

    monkeypatch.setattr(service, "_init_locked", _loads_fine(service, loads))
    service.init()
    assert service.is_loaded()
    assert service.load_failure() is None, "loaded, yet still reported failing"
