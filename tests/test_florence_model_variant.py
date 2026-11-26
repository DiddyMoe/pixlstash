#!/usr/bin/env python3
"""Tests for the configurable Florence-2 checkpoint (issue #512).

These tests never load a model: they exercise the variant plumbing only -
the service's variant switch, the VRAM figure the gate charges, the plugin
parameter schema, and the engine property that reads the user's setting.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from pixlstash.inference.engine import InferenceEngine
from pixlstash.tagger_plugins.florence2 import (
    DEFAULT_FLORENCE_VARIANT,
    FLORENCE_BASE_VRAM_MB,
    FLORENCE_LARGE_FT_VRAM_MB,
    FLORENCE_MODEL_VARIANTS,
    Florence2Plugin,
    Florence2Service,
)
from tests.test_exif_orientation_loads import _unwritable_exif


def _service(**kwargs) -> Florence2Service:
    return Florence2Service(device="cpu", **kwargs)


class TestVariantRegistry:
    def test_default_is_base(self):
        assert DEFAULT_FLORENCE_VARIANT == "base"
        assert FLORENCE_MODEL_VARIANTS[DEFAULT_FLORENCE_VARIANT]["model"].endswith(
            "Florence-2-base"
        )

    def test_every_variant_pins_a_revision(self):
        # An unpinned HuggingFace ref is a silent supply-chain change.
        for key, spec in FLORENCE_MODEL_VARIANTS.items():
            assert spec.get("revision"), f"variant {key!r} has no pinned revision"
            assert spec.get("model")
            assert isinstance(spec.get("vram_mb"), int)
            assert spec.get("label")


class TestServiceVariantSwitch:
    def test_defaults_to_base(self):
        svc = _service()
        assert svc.model_variant == "base"
        assert svc.base_vram_mb == FLORENCE_BASE_VRAM_MB
        assert svc._model_name == "florence-community/Florence-2-base"

    def test_switch_updates_model_revision_and_vram(self):
        svc = _service()
        svc.set_model_variant("large-ft")
        assert svc.model_variant == "large-ft"
        assert svc._model_name == "florence-community/Florence-2-large-ft"
        assert svc._model_revision == FLORENCE_MODEL_VARIANTS["large-ft"]["revision"]
        assert svc.base_vram_mb == FLORENCE_LARGE_FT_VRAM_MB

    def test_switch_unloads_a_resident_checkpoint(self):
        svc = _service()
        svc._model = object()
        svc._processor = object()
        assert svc.is_loaded()
        svc.set_model_variant("large-ft")
        assert not svc.is_loaded()
        assert svc._model is None
        assert svc._processor is None

    def test_switching_to_the_same_variant_keeps_the_model_loaded(self):
        svc = _service()
        svc._model = object()
        svc._processor = object()
        svc.set_model_variant("base")
        assert svc.is_loaded()

    def test_unknown_variant_is_ignored_not_applied(self):
        svc = _service()
        svc.set_model_variant("does-not-exist")
        assert svc.model_variant == "base"
        assert svc._model_name == "florence-community/Florence-2-base"

    def test_empty_variant_falls_back_to_default(self):
        svc = _service()
        svc.set_model_variant("large-ft")
        svc.set_model_variant("")
        assert svc.model_variant == DEFAULT_FLORENCE_VARIANT

    def test_state_info_reports_the_variant(self):
        svc = _service()
        svc.set_model_variant("large-ft")
        info = svc.state_info()
        assert info["florence_variant"] == "large-ft"
        assert info["florence_model"] == "florence-community/Florence-2-large-ft"


class TestVramGateFollowsTheVariant:
    @pytest.mark.parametrize(
        "variant,expected",
        [("base", FLORENCE_BASE_VRAM_MB), ("large-ft", FLORENCE_LARGE_FT_VRAM_MB)],
    )
    def test_description_batch_size_charges_the_selected_footprint(
        self, variant, expected
    ):
        seen: list[int] = []

        def vram_cap(base_mb, per_item_mb):
            seen.append(base_mb)
            return 8

        svc = Florence2Service(device="cuda", vram_cap_fn=vram_cap)
        svc.set_model_variant(variant)
        svc.description_batch_size()
        assert seen == [expected]


class TestDeviceFallback:
    def test_explicit_cuda_unavailable_loads_cpu_and_records_reason(self, monkeypatch):
        svc = Florence2Service(device="cuda")
        loaded = []

        def fake_load(device, dtype):
            loaded.append((device.type, dtype))
            svc._model = object()
            svc._processor = object()
            svc._model_device = device
            svc._dtype = dtype

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(svc, "_load_model", fake_load)

        svc.ensure_ready()

        assert loaded == [("cpu", torch.float32)]
        assert svc._model_device.type == "cpu"
        assert svc._last_fallback_reason.startswith("cuda_unavailable:")

    def test_typed_oom_retries_caption_on_cpu(self, monkeypatch, tmp_path):
        path = tmp_path / "sample.png"
        Image.new("RGB", (8, 8), "white").save(path)
        svc = Florence2Service(device="cuda")
        svc._model = object()
        svc._processor = object()
        svc._model_device = torch.device("cuda")
        svc._dtype = torch.float16
        calls = 0
        fallback_causes = []

        def infer(_image):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise torch.OutOfMemoryError("allocation failed")
            return "Recovered on CPU."

        def reload_cpu(cause=None):
            fallback_causes.append(cause)
            svc._model_device = torch.device("cpu")
            return True

        monkeypatch.setattr(svc, "_infer_single", infer)
        monkeypatch.setattr(svc, "_reload_on_cpu", reload_cpu)

        assert svc.generate_caption(str(path)) == "Recovered on CPU."
        assert calls == 2
        assert len(fallback_causes) == 1
        assert isinstance(fallback_causes[0], torch.OutOfMemoryError)

    def test_mps_loads_on_metal_in_fp16(self, monkeypatch):
        svc = Florence2Service(device="mps")
        loaded = []

        def fake_load(device, dtype):
            loaded.append((device.type, dtype))
            svc._model = object()
            svc._processor = object()

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(svc, "_load_model", fake_load)

        svc.ensure_ready()

        assert loaded == [("mps", torch.float16)]
        assert svc._last_fallback_reason is None

    def test_mps_unavailable_loads_cpu_and_records_reason(self, monkeypatch):
        svc = Florence2Service(device="mps")
        loaded = []

        def fake_load(device, dtype):
            loaded.append((device.type, dtype))
            svc._model = object()
            svc._processor = object()

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(svc, "_load_model", fake_load)

        svc.ensure_ready()

        assert loaded == [("cpu", torch.float32)]
        assert svc._last_fallback_reason.startswith("mps_unavailable:")

    def test_metal_oom_retries_caption_on_cpu(self, monkeypatch, tmp_path):
        path = tmp_path / "sample.png"
        Image.new("RGB", (8, 8), "white").save(path)
        svc = Florence2Service(device="mps")
        svc._model = object()
        svc._processor = object()
        svc._model_device = torch.device("mps")
        svc._dtype = torch.float16
        calls = 0
        fallback_causes = []

        def infer(_image):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("MPS backend out of memory")
            return "Recovered on CPU."

        def reload_cpu(cause=None):
            fallback_causes.append(cause)
            svc._model_device = torch.device("cpu")
            return True

        monkeypatch.setattr(svc, "_infer_single", infer)
        monkeypatch.setattr(svc, "_reload_on_cpu", reload_cpu)

        assert svc.generate_caption(str(path)) == "Recovered on CPU."
        assert calls == 2
        assert len(fallback_causes) == 1

    def test_a_model_bug_on_the_cpu_is_not_retried(self):
        svc = Florence2Service(device="mps")
        svc._model_device = torch.device("cpu")
        assert not svc._is_device_error(RuntimeError("MPS backend out of memory"))

    def test_a_model_bug_on_metal_is_not_retried(self):
        svc = Florence2Service(device="mps")
        svc._model_device = torch.device("mps")
        assert not svc._is_device_error(KeyError("pixel_values"))

    def test_mps_batches_are_charged_against_the_memory_budget(self):
        seen = []

        def vram_cap(base_mb, per_item_mb):
            seen.append(base_mb)
            return 3

        svc = Florence2Service(device="mps", vram_cap_fn=vram_cap)
        assert svc.description_batch_size() == 3
        assert seen == [FLORENCE_BASE_VRAM_MB]


class TestTheTextReadingLoad:
    def test_a_photo_whose_exif_cannot_be_written_back_is_read_upright(self, tmp_path):
        """EXIF Pillow reads but cannot write back no longer costs the text.

        A LONG8 tag beside orientation 6: `ImageOps.exif_transpose` raised on
        it, so the load failed and the picture was left out of the result. The
        turn writes no EXIF back now, so the reader gets the picture upright.
        """
        path = tmp_path / "unwritable-exif.png"
        Image.new("RGB", (40, 30), "blue").save(
            path, "PNG", exif=b"Exif\x00\x00" + _unwritable_exif(6)
        )
        fed_sizes = []

        class Processor:
            tokenizer = SimpleNamespace(pad_token_id=0)

            def __call__(self, text, images, return_tensors, padding):
                fed_sizes.extend(image.size for image in images)
                return {
                    "input_ids": torch.zeros((len(images), 1), dtype=torch.long),
                    "pixel_values": torch.zeros((len(images), 3, 1, 1)),
                }

            def batch_decode(self, generated_ids, skip_special_tokens):
                return ["no text"] * len(generated_ids)

            def post_process_generation(self, text, task, image_size):
                return {task: {"quad_boxes": [], "labels": []}}

        class Model:
            def generate(self, **kwargs):
                return kwargs["input_ids"]

        svc = _service()
        svc._processor = Processor()
        svc._model = Model()
        svc._model_device = torch.device("cpu")
        svc._dtype = torch.float32

        assert svc.read_text([str(path)]) == {str(path): []}, "the picture was left out"
        assert fed_sizes == [(30, 40)], "orientation 6, upright"


class TestPluginSchema:
    def test_model_variant_is_a_select_defaulting_to_base(self):
        plugin = Florence2Plugin()
        field = next(
            f for f in plugin.parameter_schema() if f["name"] == "model_variant"
        )
        assert field["type"] == "select"
        assert field["default"] == DEFAULT_FLORENCE_VARIANT
        values = [opt["value"] for opt in field["options"]]
        assert values == list(FLORENCE_MODEL_VARIANTS)

    def test_default_params_include_the_variant(self):
        assert Florence2Plugin().default_params()["model_variant"] == (
            DEFAULT_FLORENCE_VARIANT
        )


class TestEngineReadsTheSetting:
    @staticmethod
    def _variant_for(settings: dict) -> str:
        return InferenceEngine.florence_model_variant.fget(
            SimpleNamespace(_tagger_settings=settings)
        )

    def test_reads_the_configured_variant(self):
        settings = {"plugins": {"florence2": {"params": {"model_variant": "large-ft"}}}}
        assert self._variant_for(settings) == "large-ft"

    def test_defaults_when_unset(self):
        assert self._variant_for({}) == DEFAULT_FLORENCE_VARIANT
        assert self._variant_for({"plugins": {}}) == DEFAULT_FLORENCE_VARIANT
        assert (
            self._variant_for({"plugins": {"florence2": {"params": {}}}})
            == DEFAULT_FLORENCE_VARIANT
        )

    def test_defaults_on_a_malformed_settings_blob(self):
        assert self._variant_for({"plugins": {"florence2": None}}) == (
            DEFAULT_FLORENCE_VARIANT
        )
        assert self._variant_for({"plugins": {"florence2": {"params": None}}}) == (
            DEFAULT_FLORENCE_VARIANT
        )
        assert (
            self._variant_for(
                {"plugins": {"florence2": {"params": {"model_variant": None}}}}
            )
            == DEFAULT_FLORENCE_VARIANT
        )
