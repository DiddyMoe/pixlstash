"""Handling of graphs that end in a ComfyUI-PixlStash saver node.

``PixlStashPictureSaver`` does not write a file for PixlStash to collect: it
uploads straight into the vault over the API and reports the picture ids it
created in its history entry, alongside ``type: "temp"`` previews of images
that are, by then, already imported.

Treating it as "no save node" made Generate Variants refuse the whole class of
workflows built around the node pack. Treating its previews as ordinary outputs
would re-download and re-import pictures the node had just imported, which
dedups to nothing and so loses the stack placement, the source lineage and the
import event. Both are covered here.
"""

import ast
import inspect
import json
import logging
import os
import tempfile
import threading
import tracemalloc
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy.orm import configure_mappers

import pixlstash.services.comfyui_service as comfyui_service
from pixlstash.database import DatabaseUnavailableError
from pixlstash.services import comfyui_recipe_service
from pixlstash.services import workflow_run_service
from pixlstash.utils.bounded_http import BoundedResponse
from pixlstash.event_types import EventType

CKPT = "PixlStashCheckpointLoader"

SAVER_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 1}},
    "9": {"class_type": "PixlStashPictureSaver", "inputs": {"filename_prefix": "v"}},
}


def _history(outputs: dict) -> dict:
    return {"prompt-1": {"outputs": outputs, "status": {"status_str": "success"}}}


# ComfyUI's queue as ``_read_comfyui_queue`` answers it: (running, pending).
RUNNING = (frozenset({"prompt-1"}), frozenset())
WAITING = (frozenset(), frozenset({"prompt-1"}))
GONE = (frozenset(), frozenset())


def _raising(exc: BaseException):
    """A stand-in for a call that raises *exc*, whatever it is passed."""

    def raises(*_args, **_kwargs):
        raise exc

    return raises


def _answered(status_code: int, body: bytes) -> BoundedResponse:
    """ComfyUI's answer of *status_code* carrying *body*, as ``_ask_comfyui``
    returns it once read."""
    return BoundedResponse("http://comfy", status_code, body)


def _comfyui_answers(monkeypatch, histories: list, queues: list) -> list[str]:
    """ComfyUI answers a run's wait with *queues* and *histories*, in order,
    and its sleeps return at once. The log of what the wait did, in order:
    ``sleep``, ``queue`` (a read of the queue) or ``history``."""
    log = []
    sleeps = len(queues)

    def sleep(_seconds):
        # Each sleep comes before a read of the queue, so one past the answers
        # given is a wait that stopped asking: it fails rather than spins.
        assert log.count("sleep") < sleeps, "the wait slept on without asking"
        log.append("sleep")

    def read_queue(_base_url):
        log.append("queue")
        return queues.pop(0)

    def fetch_history(_base_url, _prompt_id):
        log.append("history")
        return histories.pop(0)

    monkeypatch.setattr(comfyui_service, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(comfyui_service, "_queue_reads", {})
    monkeypatch.setattr(comfyui_service, "_queue_waits", {})
    monkeypatch.setattr(comfyui_service, "_read_comfyui_queue", read_queue)
    monkeypatch.setattr(comfyui_service, "_fetch_comfyui_history", fetch_history)
    return log


class TestGraphInspection:
    def test_the_saver_counts_as_an_output_node(self):
        assert comfyui_service._extract_output_node_ids(SAVER_GRAPH, {}) == ["9"]


def _node(class_type: str, **inputs) -> dict:
    return {"class_type": class_type, "inputs": inputs}


def _why(graph: dict, **run) -> dict:
    """``{node_id: why}`` for every node the policy refuses."""
    return {
        entry["node_id"]: entry["why"]
        for entry in comfyui_service.pixlstash_node_refusals(graph, **run)
    }


# The strictest run and the most permissive one. A node allowed under STRICT is
# allowed everywhere; one refused under OPEN is refused everywhere.
STRICT: dict = {}
OPEN = {
    "library_ids": {
        "project": {3: "Holiday"},
        "set": {4: "Holiday"},
        "character": {5: "Holiday"},
    },
    "picture_loader": True,
    "from_file": True,
}


class TestNodePolicy:
    """Each ComfyUI-PixlStash class has its own answer (#1521), both ways."""

    def test_digest_loaders_searches_and_gates_run_everywhere(self):
        for cls in (
            "PixlStashAdapterLoader",
            "PixlStashVAELoader",
            "PixlStashCLIPLoader",
            "PixlStashLikenessSearch",
            "PixlStashSemanticSearch",
            "PixlStashFaceLikenessGate",
            "PixlStashPictureLikenessGate",
        ):
            assert _why({"9": _node(cls)}, **STRICT) == {}, cls

    def test_a_library_loader_runs_when_its_id_is_in_this_library(self):
        for cls, field, kind, library_id in (
            ("PixlStashProjectLoader", "pixlstash_project", "project", 3),
            ("PixlStashSetLoader", "pixlstash_set", "set", 4),
            ("PixlStashCharacterLoader", "pixlstash_character", "character", 5),
        ):
            graph = {"9": _node(cls, **{field: f"Holiday #{library_id}"})}
            assert comfyui_service.library_ids_named(graph) == {kind: {library_id}}
            assert _why(graph, library_ids={kind: {library_id: "Holiday"}}) == {}
            # The same id under another kind is no answer, and nor is the same
            # id under another name: ids start at 1 in every library.
            other = "set" if kind != "set" else "project"
            for present in (
                {other: {library_id: "Holiday"}},
                {kind: {library_id: "Portraits"}},
            ):
                refused = comfyui_service.pixlstash_node_refusals(
                    graph, library_ids=present
                )
                assert [(e["why"], e["kind"], e["id"]) for e in refused] == [
                    ("not_in_library", kind, library_id)
                ], (cls, present)

    def test_a_library_loader_with_no_choice_names_nothing_to_check(self):
        graph = {"9": _node("PixlStashSetLoader", pixlstash_set="(loading…)")}
        assert comfyui_service.library_ids_named(graph) == {}
        assert _why(graph, **STRICT) == {}

    def test_a_wired_library_choice_cannot_be_checked_so_is_refused(self):
        graph = {"9": _node("PixlStashProjectLoader", pixlstash_project=["1", 0])}
        assert _why(graph, **OPEN) == {"9": "unreadable_id"}

    def test_the_picture_loader_runs_only_where_the_run_feeds_it(self):
        graph = {"9": _node("PixlStashPictureLoader", picture_ids="1,2")}
        assert _why(graph, picture_loader=True) == {}
        assert _why(graph, **{**OPEN, "picture_loader": False}) == {
            "9": "picks_its_own_picture"
        }
        # The run's half: a loader it did not write ids into is refused.
        assert [
            e["why"] for e in comfyui_service.unfed_picture_loaders(graph, set())
        ] == ["picks_its_own_picture"]
        assert comfyui_service.unfed_picture_loaders(graph, {"9"}) == []

    def test_the_checkpoint_loader_runs_only_from_a_stored_file(self):
        graph = {"9": _node("PixlStashCheckpointLoader", checkpoint_id="12")}
        assert _why(graph, from_file=True) == {}
        assert _why(graph, **{**OPEN, "from_file": False}) == {
            "9": "per_hub_checkpoint"
        }

    def test_a_picture_loader_runs_on_ids_that_say_they_are_this_librarys(self):
        # Unfed, from any source: what Open in ComfyUI and the node's picker
        # write, read back out of a picture (#1912).
        def loader(**inputs):
            return {"9": _node("PixlStashPictureLoader", **inputs)}

        own = loader(picture_ids="7, 12", library="lib-here")
        assert comfyui_service.own_picture_ids(own["9"], "lib-here") == [7, 12]
        assert _why(own, library_uuid="lib-here") == {}
        refused = {"9": "picks_its_own_picture"}
        # Another library's ids, ids that name no library, and no library to
        # be: an id alone starts at 1 everywhere.
        assert _why(own, library_uuid="lib-other") == refused
        assert _why(loader(picture_ids="7"), library_uuid="lib-here") == refused
        assert _why(loader(picture_ids="7", library=""), library_uuid="") == refused
        assert _why(own, library_uuid=None) == refused
        # No id at all picks pictures by the loader's own sort, whoever's.
        for ids in ("", " , ", "7,x", "-3", ["4", 0], None):
            graph = loader(picture_ids=ids, library="lib-here")
            assert comfyui_service.own_picture_ids(graph["9"], "lib-here") == [], ids
            assert _why(graph, library_uuid="lib-here") == refused, ids

    def test_a_checkpoint_loader_runs_on_an_id_that_says_it_is_this_shelfs(self):
        def loader(**inputs):
            return {"9": _node("PixlStashCheckpointLoader", **inputs)}

        own = loader(checkpoint_id="12", shelf="shelf-here")
        assert comfyui_service.own_checkpoint_id(own["9"], "shelf-here") == "12"
        assert _why(own, shelf_uuid="shelf-here") == {}
        refused = {"9": "per_hub_checkpoint"}
        assert _why(own, shelf_uuid="shelf-other") == refused
        assert _why(own) == refused
        assert _why(loader(checkpoint_id="12"), shelf_uuid="shelf-here") == refused
        assert _why(loader(checkpoint_id="12", shelf=""), shelf_uuid="") == refused
        for row in ("", "x", "1 2", "007", "0", ["4", 0]):
            graph = loader(checkpoint_id=row, shelf="shelf-here")
            assert _why(graph, shelf_uuid="shelf-here") == refused, row

    def test_the_saver_is_refused_until_swapped_and_runs_as_save_image(self):
        graph = {
            "9": {
                **_node(
                    "PixlStashPictureSaver",
                    images=["3", 0],
                    filename_prefix="v",
                    save_workflow=True,
                    pixlstash_set=["8", 1],
                ),
                "_meta": {"title": "Keep"},
            }
        }
        assert _why(graph, **OPEN) == {"9": "imports_itself"}
        assert comfyui_service.swap_pixlstash_savers(graph) == ["9"]
        assert graph["9"] == {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "v"},
            "_meta": {"title": "Keep"},
        }
        assert _why(graph, **STRICT) == {}

    def test_a_saver_whose_output_is_read_is_not_swapped(self):
        # SaveImage has no output to hand on, so the swap would break the link.
        graph = {
            "9": _node("PixlStashPictureSaver", images=["3", 0]),
            "10": _node("ShowText", text=["9", 0]),
        }
        assert comfyui_service.swap_pixlstash_savers(graph) == []
        assert _why(graph, **OPEN) == {"9": "imports_itself"}

    def test_a_saver_with_no_images_is_not_swapped(self):
        graph = {"9": _node("PixlStashPictureSaver", filename_prefix="v")}
        assert comfyui_service.swap_pixlstash_savers(graph) == []
        assert _why(graph, **OPEN) == {"9": "imports_itself"}

    def test_a_pack_node_without_an_entry_stays_refused(self):
        graph = {"9": _node("PixlStashSomethingNew")}
        assert _why(graph, **OPEN) == {"9": "no_policy"}

    def test_an_ordinary_graph_and_a_lookalike_are_not_pack_nodes(self):
        graph = {
            "3": _node("KSampler"),
            "8": _node("NotPixlStashSaver"),
            "9": _node("SaveImage"),
        }
        assert _why(graph, **STRICT) == {}
        assert comfyui_service.swap_pixlstash_savers(graph) == []


class TestOpeningWithPackNodes:
    """``swap_in_pack_nodes``: what Open in ComfyUI puts in a graph."""

    INFO = {
        "SaveImage": {"display_name": "Save Image"},
        "PixlStashPictureSaver": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "comfyui"}],
                    "save_workflow": ["BOOLEAN", {"default": True}],
                },
                "optional": {"pixlstash_set": ["PIXLSTASH_SET", {}]},
            },
            "display_name": "PixlStash Picture Saver",
        },
        "PixlStashVideoSaver": {
            "input": {
                "required": {
                    "video": ["VIDEO", {}],
                    "filename_prefix": ["STRING", {"default": "video/comfyui"}],
                    "format": ["COMFY_DYNAMICCOMBO_V3", {}],
                    "save_workflow": ["BOOLEAN", {"default": True}],
                }
            },
        },
        "PixlStashAdapterLoader": {
            "input": {
                "required": {"model": ["MODEL", {}], "adapter_sha256": ["STRING", {}]},
                "optional": {"strength_model": ["FLOAT", {"default": 1.0}]},
            },
            "output": ["MODEL", "CLIP", "STRING"],
        },
    }

    @staticmethod
    def _swap(graph, info=None, digests=None):
        return comfyui_recipe_service.swap_in_pack_nodes(
            graph,
            TestOpeningWithPackNodes.INFO if info is None else info,
            (digests or {}).get,
        )

    def test_a_save_image_becomes_the_picture_saver_on_the_same_inputs(self):
        graph = {
            "9": {
                **_node("SaveImage", images=["3", 0], filename_prefix="v"),
                "_meta": {"title": "Save Image"},
            }
        }
        assert self._swap(graph) == [
            {"node_id": "9", "was": "SaveImage", "now": "PixlStashPictureSaver"}
        ]
        assert graph["9"] == {
            "class_type": "PixlStashPictureSaver",
            "inputs": {
                "filename_prefix": "v",
                "save_workflow": True,
                "images": ["3", 0],
            },
            # The stock node's own name gives way; a title the owner wrote stays.
            "_meta": {"title": "PixlStash Picture Saver"},
        }
        # And a run of it submits the stock saver again: one import, not two.
        assert comfyui_service.swap_pixlstash_savers(graph) == ["9"]
        assert graph["9"]["class_type"] == "SaveImage"
        assert graph["9"]["inputs"] == {"filename_prefix": "v", "images": ["3", 0]}

    def test_a_title_the_owner_wrote_is_kept(self):
        graph = {
            "9": {**_node("SaveImage", images=["3", 0]), "_meta": {"title": "Keep"}}
        }
        self._swap(graph)
        assert graph["9"]["_meta"] == {"title": "Keep"}

    def test_a_save_video_becomes_the_video_saver_and_runs_as_save_video(self):
        inputs = {
            "video": ["7", 0],
            "filename_prefix": "video/v",
            "format": "mp4",
            "format.codec": "h264",
        }
        graph = {"9": _node("SaveVideo", **inputs)}
        assert [s["now"] for s in self._swap(graph)] == ["PixlStashVideoSaver"]
        assert graph["9"]["inputs"] == {**inputs, "save_workflow": True}
        assert _why(graph, **OPEN) == {"9": "imports_itself"}
        assert comfyui_service.swap_pixlstash_savers(graph) == ["9"]
        assert graph["9"] == _node("SaveVideo", **inputs)
        assert _why(graph, **STRICT) == {}

    def test_a_setting_the_video_saver_lacks_takes_its_default(self):
        # A current ComfyUI's Save Video has a `codec` the pack's saver does
        # not: the saver encodes with its own default rather than staying out.
        graph = {
            "9": _node(
                "SaveVideo",
                video=["7", 0],
                format="auto",
                **{"format.codec": "auto", "codec": "auto", "codec.crf": 23},
            )
        }
        assert [s["now"] for s in self._swap(graph)] == ["PixlStashVideoSaver"]
        assert graph["9"]["inputs"] == {
            "video": ["7", 0],
            "filename_prefix": "video/comfyui",
            "format": "auto",
            "format.codec": "auto",
            "save_workflow": True,
        }

    def test_a_saver_is_left_where_the_pack_cannot_stand_in_for_it(self):
        # A wire the pack's saver has no place for, nothing wired in to save,
        # and a ComfyUI without the pack.
        old = _node("SaveVideo", video=["7", 0], codec=["6", 0])
        unwired = _node("SaveImage", filename_prefix="v")
        graph = {"8": dict(old), "9": dict(unwired)}
        assert self._swap(graph) == []
        assert graph == {"8": old, "9": unwired}
        wired = {"9": _node("SaveImage", images=["3", 0])}
        assert self._swap(wired, info={"SaveImage": {}}) == []
        assert wired["9"]["class_type"] == "SaveImage"

    def test_a_lora_loader_loads_by_digest_only_a_file_the_shelf_can_name(self):
        graph = {
            "2": _node(
                "LoraLoaderModelOnly", lora_name="a.safetensors", model=["1", 0]
            ),
            "5": _node(
                "LoraLoaderModelOnly", lora_name="b.safetensors", model=["2", 0]
            ),
            "6": _node("LoraLoaderModelOnly", lora_name=["4", 0], model=["5", 0]),
        }
        untouched = {key: dict(graph[key]) for key in ("5", "6")}
        swapped = self._swap(graph, digests={"a.safetensors": "d" * 64})
        assert [(s["node_id"], s["now"]) for s in swapped] == [
            ("2", "PixlStashAdapterLoader")
        ]
        assert graph["2"]["inputs"] == {
            "strength_model": 1.0,
            "adapter_sha256": "d" * 64,
            "model": ["1", 0],
        }
        assert {key: graph[key] for key in ("5", "6")} == untouched

    ID_LOADERS = {
        "LoadImage": {"display_name": "Load Image"},
        "PixlStashPictureLoader": {
            "input": {
                "required": {"picture_ids": ["STRING", {"default": ""}]},
                "optional": {
                    "library": ["STRING", {"default": ""}],
                    "pixlstash_set": ["PIXLSTASH_SET", {"forceInput": True}],
                },
            },
            "output": ["IMAGE", "MASK", "PIXLSTASH_PROJECT"],
            "display_name": "PixlStash Picture Loader",
        },
        "PixlStashCheckpointLoader": {
            "input": {
                "required": {"checkpoint_id": ["STRING", {"default": ""}]},
                "optional": {"shelf": ["STRING", {"default": ""}]},
            },
            "output": ["MODEL", "CLIP", "VAE"],
        },
    }

    def test_a_diffusion_model_loads_off_the_shelf_as_a_checkpoint(self):
        """The pack's checkpoint loader hands on MODEL first, so it stands in
        for Load Diffusion Model on the shelf's row, but not where the stock
        loader casts the weights and this pack's loader has no cast."""
        info = {**self.INFO, **self.ID_LOADERS}
        known = {"checkpoint_id": {"u.safetensors": "8"}.get, "shelf_uuid": "s"}
        unet = _node("UNETLoader", unet_name="u.safetensors", weight_dtype="default")
        graph = {"1": dict(unet), "5": _node("KSampler", model=["1", 0])}
        swapped = comfyui_recipe_service.swap_in_pack_nodes(
            graph, info, {}.get, **known
        )
        assert [(s["was"], s["now"]) for s in swapped] == [("UNETLoader", CKPT)]
        assert graph["1"] == _node(CKPT, checkpoint_id="8", shelf="s")
        assert graph["5"]["inputs"] == {"model": ["1", 0]}
        cast = {"1": _node("UNETLoader", unet_name="u.safetensors", weight_dtype="fp8")}
        assert (
            comfyui_recipe_service.swap_in_pack_nodes(cast, info, {}.get, **known) == []
        )

    def test_a_cast_goes_with_the_file_onto_a_pack_loader_that_can_cast(self):
        """Wrong if the owner's fp8 choice is dropped, or if an uncast loader
        gains a cast it never had."""
        loader = json.loads(json.dumps(self.ID_LOADERS[CKPT]))
        loader["input"]["optional"]["weight_dtype"] = [
            ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
            {"default": "default"},
        ]
        info = {**self.INFO, **self.ID_LOADERS, CKPT: loader}
        known = {"checkpoint_id": {"u.safetensors": "8"}.get, "shelf_uuid": "s"}
        cast = {
            "1": _node(
                "UNETLoader", unet_name="u.safetensors", weight_dtype="fp8_e4m3fn_fast"
            )
        }
        swapped = comfyui_recipe_service.swap_in_pack_nodes(cast, info, {}.get, **known)
        assert [(s["was"], s["now"]) for s in swapped] == [("UNETLoader", CKPT)]
        assert cast["1"]["inputs"] == {
            "checkpoint_id": "8",
            "shelf": "s",
            "weight_dtype": "fp8_e4m3fn_fast",
        }
        plain = {
            "1": _node("UNETLoader", unet_name="u.safetensors", weight_dtype="default")
        }
        comfyui_recipe_service.swap_in_pack_nodes(plain, info, {}.get, **known)
        assert plain["1"]["inputs"]["weight_dtype"] == "default"
        # A cast wired in from another node has nowhere to go: stock it stays.
        wired = {
            "1": _node("UNETLoader", unet_name="u.safetensors", weight_dtype=["5", 0])
        }
        assert (
            comfyui_recipe_service.swap_in_pack_nodes(wired, info, {}.get, **known)
            == []
        )

    @staticmethod
    def _stock():
        return {
            "1": _node("CheckpointLoaderSimple", ckpt_name="a.safetensors"),
            "2": {
                **_node("LoadImage", image="a.png", upload="image"),
                "_meta": {"title": "Load Image"},
            },
            "3": _node("VAEEncode", pixels=["2", 0], vae=["1", 2]),
        }

    def _swap_ids(self, graph, info=None, **known):
        known = {
            "pictures": {"2": 41},
            "library_uuid": "lib-here",
            "checkpoint_id": {"a.safetensors": "7"}.get,
            "shelf_uuid": "shelf-here",
            **known,
        }
        return comfyui_recipe_service.swap_in_pack_nodes(
            graph,
            {**self.INFO, **self.ID_LOADERS} if info is None else info,
            {}.get,
            **known,
        )

    def test_the_id_loaders_are_swapped_in_with_whose_the_id_is(self):
        graph = self._stock()
        assert self._swap_ids(graph) == [
            {"node_id": "1", "was": "CheckpointLoaderSimple", "now": CKPT},
            {"node_id": "2", "was": "LoadImage", "now": "PixlStashPictureLoader"},
        ]
        assert graph["1"] == _node(CKPT, checkpoint_id="7", shelf="shelf-here")
        # The file it named and its upload widget go; the wired filters the
        # pack's loader also takes are not made up.
        assert graph["2"] == {
            **_node("PixlStashPictureLoader", picture_ids="41", library="lib-here"),
            "_meta": {"title": "PixlStash Picture Loader"},
        }
        # What read them reads them where it did.
        assert graph["3"] == self._stock()["3"]
        # And what was put in is what a run accepts back.
        assert (
            comfyui_service.pixlstash_node_refusals(
                graph, library_uuid="lib-here", shelf_uuid="shelf-here"
            )
            == []
        )

    def test_an_id_loader_is_never_swapped_in_without_what_it_needs(self):
        stock = self._stock()
        # Not asked to swap either, or with no library or shelf to say whose
        # an id is.
        for known in (
            {"pictures": None, "checkpoint_id": None},
            {"library_uuid": None, "shelf_uuid": None},
            {"library_uuid": "", "shelf_uuid": ""},
        ):
            graph = self._stock()
            assert self._swap_ids(graph, **known) == [], known
            assert graph == stock
        # A pack from before the widget that says whose the id is: the id
        # would not be trusted again, so the stock node stays.
        old_pack = json.loads(json.dumps({**self.INFO, **self.ID_LOADERS}))
        del old_pack["PixlStashPictureLoader"]["input"]["optional"]["library"]
        del old_pack["PixlStashCheckpointLoader"]["input"]["optional"]
        # One whose outputs are not the stock node's, and a ComfyUI without.
        other = json.loads(json.dumps({**self.INFO, **self.ID_LOADERS}))
        other["PixlStashPictureLoader"]["output"] = ["MASK", "IMAGE"]
        other["PixlStashCheckpointLoader"]["output"] = ["MODEL"]
        for info in (old_pack, other, self.INFO):
            graph = self._stock()
            assert self._swap_ids(graph, info=info) == []
            assert graph == stock

    def test_a_load_image_the_library_holds_no_picture_of_opens_empty(self):
        """A Load Image whose picture the library is not known to hold opens
        as an empty Picture Loader to browse in, whether or not ComfyUI holds
        its file. Wrong if it stays a Load Image."""
        info = json.loads(json.dumps({**self.INFO, **self.ID_LOADERS}))
        for listed in ([["a.png"], {}], [["b.png"], {}], None):
            if listed is None:
                info["LoadImage"].pop("input", None)
            else:
                info["LoadImage"]["input"] = {"required": {"image": listed}}
            for image in ("a.png", "a.png [output]", "a.png [temp]"):
                graph = self._stock()
                graph["2"]["inputs"]["image"] = image
                assert self._swap_ids(graph, info=info, pictures={})[1] == {
                    "node_id": "2",
                    "was": "LoadImage",
                    "now": "PixlStashPictureLoader",
                }, (listed, image)
                assert graph["2"]["inputs"] == {
                    "picture_ids": "",
                    "library": "lib-here",
                }
        # Not asked to swap pictures at all: no Load Image is.
        graph = self._stock()
        assert len(self._swap_ids(graph, info=info, pictures=None)) == 1
        assert graph["2"] == self._stock()["2"]
        # A file name wired in from another node would be dropped: it stays,
        # unless the library's picture is known.
        wired = self._stock()
        wired["2"]["inputs"]["image"] = ["9", 0]
        graph = json.loads(json.dumps(wired))
        assert len(self._swap_ids(graph, info=info, pictures={})) == 1
        assert graph["2"] == wired["2"]
        self._swap_ids(graph, info=info)
        assert graph["2"]["inputs"]["picture_ids"] == "41"

    def test_every_swap_reads_as_the_stock_node_for_a_layout(self):
        # A layout is matched on the stock classes, so which of a pair the
        # graph holds on the day does not drop it.
        stock = {
            **self._stock(),
            "4": _node("LoraLoader", lora_name="x.safetensors", model=["1", 0]),
            "9": _node("SaveImage", images=["3", 0], filename_prefix="v"),
            "m1": _node("UNETLoader", unet_name="u.safetensors"),
        }
        swapped = json.loads(json.dumps(stock))
        info = {**self.INFO, **self.ID_LOADERS}
        done = comfyui_recipe_service.swap_in_pack_nodes(
            swapped,
            info,
            {"x.safetensors": "d" * 64}.get,
            pictures={"2": 41},
            library_uuid="lib-here",
            checkpoint_id={"a.safetensors": "7", "u.safetensors": "8"}.get,
            shelf_uuid="shelf-here",
        )
        assert {s["node_id"] for s in done} == {"1", "2", "4", "9", "m1"}
        classes = comfyui_recipe_service.SWAPPED_CLASSES
        structure = workflow_run_service.graph_structure
        assert structure(swapped) != structure(stock)
        assert structure(swapped, classes) == structure(stock, classes)
        # Wired another way it is another layout, whatever the classes.
        swapped["3"]["inputs"]["vae"] = ["1", 0]
        assert structure(swapped, classes) != structure(stock, classes)


class TestReadingPackLoadersAsStock:
    """A graph saved from ComfyUI holds the pack's model loaders; the screens
    that change models read each as the stock loader of its file."""

    ROWS = {("130", "this-shelf"): "unet.safetensors"}
    DIGESTS = {
        "a" * 64: "vae.safetensors",
        "b" * 64: "t5.safetensors",
        "c" * 64: "clip_l.safetensors",
    }

    def _read(self, graph):
        return comfyui_recipe_service.as_stock_loaders(
            graph,
            lambda row, shelf: self.ROWS.get((row, shelf)),
            self.DIGESTS.get,
        )

    def test_each_model_loader_reads_as_the_stock_one_of_its_file(self):
        graph = {
            "1": {
                "class_type": CKPT,
                "inputs": {"checkpoint_id": "130", "shelf": "this-shelf"},
            },
            "2": {
                "class_type": "PixlStashVAELoader",
                "inputs": {"vae_sha256": "A" * 64},
            },
            "3": {
                "class_type": "PixlStashCLIPLoader",
                "inputs": {
                    "clip_sha256": "b" * 64,
                    "clip_sha256_2": "c" * 64,
                    "type": "flux",
                },
            },
            "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        read, done = self._read(graph)
        # Only its MODEL is read: a diffusion model's loader.
        assert read["1"] == {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "unet.safetensors", "weight_dtype": "default"},
        }
        assert read["2"]["inputs"] == {"vae_name": "vae.safetensors"}
        assert read["3"] == {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
        }
        assert read["4"] is graph["4"]
        assert {d["node_id"] for d in done} == {"1", "2", "3"}
        # The graph it was handed is left as it was.
        assert graph["1"]["class_type"] == CKPT

        # Its CLIP read too: the whole checkpoint's loader.
        graph["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}}
        assert self._read(graph)[0]["1"] == {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "unet.safetensors"},
        }

    def test_a_cast_the_swap_carried_reads_back_onto_the_stock_loader(self):
        """Wrong if an fp8 pack loader reads back uncast: every later run and
        save of the graph would load the model at full size."""
        graph = {
            "1": {
                "class_type": CKPT,
                "inputs": {
                    "checkpoint_id": "130",
                    "shelf": "this-shelf",
                    "weight_dtype": "fp8_e4m3fn_fast",
                },
            },
            "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        assert self._read(graph)[0]["1"]["inputs"] == {
            "unet_name": "unet.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        }

    def test_a_loader_naming_no_file_of_this_shelf_stays(self):
        graph = {
            # Another shelf's row, and one with no shelf: not this hub's ids.
            "1": {
                "class_type": CKPT,
                "inputs": {"checkpoint_id": "130", "shelf": "another-shelf"},
            },
            "2": {"class_type": CKPT, "inputs": {"checkpoint_id": "130"}},
            "3": {
                "class_type": "PixlStashVAELoader",
                "inputs": {"vae_sha256": "d" * 64},
            },
        }
        read, done = self._read(graph)
        assert read == graph and done == []


class TestHistoryExtraction:
    def test_picture_ids_are_parsed_from_the_comma_joined_string(self):
        payload = _history(
            {
                "9": {
                    "images": [{"filename": "v_00001.png", "type": "temp"}],
                    "picture_ids": ["41,42"],
                }
            }
        )
        assert comfyui_service._extract_pixlstash_picture_ids(
            payload, "prompt-1", ["9"]
        ) == [41, 42]

    def test_no_saver_node_reports_none_not_an_empty_list(self):
        # None means "no PixlStash saver ran, import the images normally".
        payload = _history({"9": {"images": [{"filename": "out_00001.png"}]}})
        assert (
            comfyui_service._extract_pixlstash_picture_ids(payload, "prompt-1", None)
            is None
        )

    def test_a_saver_that_imported_nothing_new_reports_an_empty_list(self):
        # Every image was a duplicate of one already in the vault. Distinct from
        # None: there is still nothing to download.
        payload = _history(
            {
                "9": {
                    "images": [{"filename": "v_00001.png", "type": "temp"}],
                    "picture_ids": [""],
                }
            }
        )
        assert (
            comfyui_service._extract_pixlstash_picture_ids(payload, "prompt-1", None)
            == []
        )

    def test_saver_previews_are_not_offered_for_import(self):
        payload = _history(
            {
                "9": {
                    "images": [{"filename": "v_00001.png", "type": "temp"}],
                    "picture_ids": ["41"],
                }
            }
        )
        assert (
            comfyui_service._extract_comfyui_output_images(payload, "prompt-1", None)
            == []
        )

    def test_a_sibling_save_image_is_still_imported(self):
        # Mixed graph: the SaveImage output is ours to collect, the saver's is
        # not, and both have to end up in the same new_ids set downstream.
        payload = _history(
            {
                "8": {"images": [{"filename": "out_00001.png"}]},
                "9": {
                    "images": [{"filename": "v_00001.png", "type": "temp"}],
                    "picture_ids": ["41"],
                },
            }
        )
        images = comfyui_service._extract_comfyui_output_images(
            payload, "prompt-1", None
        )
        assert [img["filename"] for img in images] == ["out_00001.png"]


class _FakeVault:
    def __init__(self):
        self.events = []

    def notify(self, event_type, payload):
        self.events.append((event_type, payload))


class _FakeServer:
    def __init__(self):
        self.vault = _FakeVault()


_MIB = 1024 * 1024


class _StreamedView:
    """ComfyUI's ``GET /view`` as ``_ask_comfyui`` hands it over with a sink.

    The body goes to the sink in three pieces, and nothing comes back in the
    answer: holding the whole body is what kept every output of a run in
    memory until its import ended. With *cut_after*, the body ends after that
    many pieces with the error a dropped connection raises.
    """

    def __init__(self, body: bytes, cut_after: int | None = None):
        self.pieces = 0
        self._body = body
        self._cut_after = cut_after

    def send(self, url: str, sink) -> BoundedResponse:
        piece = max(1, -(-len(self._body) // 3))
        for n, start in enumerate(range(0, len(self._body), piece)):
            if n == self._cut_after:
                raise comfyui_service.requests.exceptions.ChunkedEncodingError(
                    "test-cut-off"
                )
            sink(self._body[start : start + piece])
            self.pieces += 1
        return BoundedResponse(url, 200, b"")


def _serve_views(monkeypatch, bodies: dict, cut_after: int | None = None) -> list:
    """Stand in for ComfyUI's ``/view`` over *bodies*, by file name; return
    ``(url, keyword arguments, response)`` for each request."""
    asked = []

    def request(method, url, **kwargs):
        assert method == "GET"
        view = _StreamedView(bodies[kwargs["params"]["filename"]], cut_after)
        asked.append((url, kwargs, view))
        return view.send(url, kwargs["sink"])

    monkeypatch.setattr(comfyui_service, "bounded_request", request)
    return asked


def _traced_peak(call):
    """Run *call* under ``tracemalloc``; return its result and the heap peak.

    The ORM's mappers are configured first: the first ``Picture`` a process
    builds does that, a one-time ~3 MiB that is not the file being measured.
    """
    configure_mappers()
    tracemalloc.start()
    try:
        result = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


class TestFetchingOutputs:
    """An output is fetched to a file and imported from it, never held whole."""

    def test_an_output_is_written_to_disk_as_it_arrives(self, monkeypatch, tmp_path):
        body = b"test-output-" * 30000
        asked = _serve_views(monkeypatch, {"out_00001.png": body})

        path, ext = comfyui_service._download_comfyui_image(
            "http://comfy",
            {"filename": "out_00001.png", "subfolder": "test-sub", "type": "output"},
            str(tmp_path),
        )

        assert (os.path.dirname(path), os.path.splitext(path)[1], ext) == (
            str(tmp_path),
            ".png",
            ".png",
        )
        with open(path, "rb") as handle:
            assert handle.read() == body
        ((url, kwargs, view),) = asked
        assert url == "http://comfy/view"
        sink = kwargs.pop("sink")
        assert callable(sink)
        assert kwargs == {
            "params": {
                "filename": "out_00001.png",
                "subfolder": "test-sub",
                "type": "output",
            },
            "deadline_s": comfyui_service.COMFYUI_FILE_DEADLINE_S,
            "max_bytes": comfyui_service.COMFYUI_FILE_MAX_BYTES,
            "timeout": comfyui_service.COMFYUI_ASK_TIMEOUT_S,
            "proxies": comfyui_service.NO_PROXY,
        }
        assert view.pieces == 3

    @pytest.mark.parametrize("kind", ["relative", "absolute"])
    def test_a_fetched_file_stays_in_the_runs_folder(self, monkeypatch, tmp_path, kind):
        """ComfyUI names the file, but only its extension is taken: a name
        that climbs out of the run's folder, or an absolute path, still lands
        inside it. Both escapes point inside this test's own folder."""
        run = tmp_path / "run" / "fetch"
        run.mkdir(parents=True)
        escape = tmp_path / "test-escape.png"
        name = "../../test-escape.png" if kind == "relative" else str(escape)
        _serve_views(monkeypatch, {name: b"test-output-" * 1000})

        path, ext = comfyui_service._download_comfyui_image(
            "http://comfy", {"filename": name, "type": "output"}, str(run)
        )

        assert (os.path.dirname(path), ext) == (str(run), ".png")
        assert os.listdir(run) == [os.path.basename(path)]
        assert not escape.exists()

    def test_a_fetch_cut_off_midway_is_a_failed_fetch(self, monkeypatch, tmp_path):
        asked = _serve_views(
            monkeypatch, {"out_00001.png": b"test-output-" * 30000}, cut_after=1
        )

        with pytest.raises(comfyui_service.HTTPException) as failed:
            comfyui_service._download_comfyui_image(
                "http://comfy", {"filename": "out_00001.png"}, str(tmp_path)
            )

        assert (failed.value.status_code, failed.value.detail) == (
            502,
            "ComfyUI image fetch failed",
        )
        assert asked[0][2].pieces == 1

    def test_an_output_is_imported_without_being_read_whole(self, tmp_path):
        """A ~14 MiB output is fingerprinted and copied into the library with
        a heap peak under a quarter of its size, and arrives byte for byte.

        Decoded pixels live in Pillow's own allocations, outside the traced
        heap, so what this measures is the file held in memory.
        """
        downloads = tmp_path / "test-downloads"
        library = tmp_path / "library"
        downloads.mkdir()
        library.mkdir()
        source = downloads / "test-output.bmp"
        Image.new("RGB", (2200, 2200), (40, 50, 60)).save(source, format="BMP")
        size = os.path.getsize(source)
        server = SimpleNamespace(
            vault=SimpleNamespace(
                image_root=str(library),
                db=SimpleNamespace(
                    # The library holds nothing with this content, and the
                    # rows are not what this test is about.
                    run_immediate_read_task=lambda fn, *args: {},
                    run_task=lambda fn, *args: [],
                ),
            )
        )

        _, peak = _traced_peak(
            lambda: comfyui_service._import_comfyui_outputs(
                server, [(str(source), ".bmp")]
            )
        )

        assert peak < size / 4, f"heap peak {peak / _MIB:.1f} MiB of {size / _MIB:.1f}"
        (published,) = [p for p in library.iterdir() if p.suffix == ".bmp"]
        assert published.read_bytes() == source.read_bytes()


class TestOutputProcessing:
    def _run(
        self,
        monkeypatch,
        images,
        pixlstash_ids,
        imported=(),
        duplicates=(),
        fetch=False,
        import_outputs=None,
        **kwargs,
    ):
        """Process a run's outputs with everything past ComfyUI stood in for.

        With *fetch*, the real download runs (the test stands in for
        ``bounded_request``); *import_outputs* replaces the stand-in import.
        """
        server = _FakeServer()
        calls = {
            "downloads": 0,
            "stacked": None,
            "sourced": None,
            "filed": None,
            "person": None,
        }

        monkeypatch.setattr(
            comfyui_service,
            "_wait_for_comfyui_outputs",
            lambda *a, **kw: (images, pixlstash_ids),
        )

        def fake_download(base_url, entry, dest_dir):
            calls["downloads"] += 1
            path = os.path.join(dest_dir, f"test-{calls['downloads']}.png")
            with open(path, "wb") as handle:
                handle.write(b"png-bytes")
            return path, ".png"

        if not fetch:
            monkeypatch.setattr(
                comfyui_service, "_download_comfyui_image", fake_download
            )
        monkeypatch.setattr(
            comfyui_service,
            "_import_comfyui_outputs",
            import_outputs or (lambda *a, **kw: (list(imported), list(duplicates))),
        )
        monkeypatch.setattr(
            comfyui_service, "_held_before", lambda srv, ids, started: list(ids)
        )
        monkeypatch.setattr(
            comfyui_service,
            "_assign_outputs_to_lora_people",
            lambda srv, people, ids: calls.__setitem__("person", (people, ids)),
        )
        monkeypatch.setattr(
            comfyui_service,
            "_assign_outputs_to_stack_top",
            lambda srv, stack_id, ids: calls.__setitem__("stacked", (stack_id, ids)),
        )
        monkeypatch.setattr(
            comfyui_service,
            "_set_source_picture_id_on_pictures",
            lambda srv, src, ids: calls.__setitem__("sourced", (src, ids)),
        )
        monkeypatch.setattr(
            comfyui_service, "_copy_set_and_project_assignments", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            comfyui_service,
            "_set_run_workflow_id",
            lambda srv, workflow_id, ids, version=None: calls.__setitem__(
                "filed", (workflow_id, ids, version)
            ),
        )

        comfyui_service._process_comfyui_outputs(
            server, "http://comfy", "prompt-1", ["9"], 7, None, **kwargs
        )
        return server, calls

    @staticmethod
    def _ending(event):
        """``(run_id, status)`` of a ``PLUGIN_PROGRESS`` event."""
        event_type, payload = event
        assert event_type == EventType.PLUGIN_PROGRESS
        return payload["run_id"], payload["status"]

    def test_a_manual_workflows_run_files_what_it_made_on_it(self, monkeypatch):
        """Imported and saver-reported pictures both; a plain run files none."""
        _server, calls = self._run(
            monkeypatch,
            images=[{"filename": "out_00001.png", "subfolder": "", "type": "output"}],
            pixlstash_ids=[41],
            imported=[40],
            run_workflow_id="manual:" + "a" * 32,
            run_workflow_version=3,
        )
        # With the version of its document the run submitted.
        assert calls["filed"] == ("manual:" + "a" * 32, [40, 41], 3)
        _server, calls = self._run(
            monkeypatch, images=[], pixlstash_ids=[41], imported=[]
        )
        assert calls["filed"] is None

    def test_what_the_run_made_is_of_the_person_whose_lora_it_loaded(self, monkeypatch):
        """Imported and saver-reported both; a duplicate keeps the people it has."""
        _server, calls = self._run(
            monkeypatch,
            images=[{"filename": "out_00001.png", "subfolder": "", "type": "output"}],
            pixlstash_ids=[41],
            imported=[40],
            duplicates=[39],
            lora_character_ids=[5],
        )
        assert calls["person"] == ([5], [40, 41])
        # Some new, one the library had: the run completed and says so.
        assert "also made a picture" in _server.vault.events[-1][1]["notice"]
        _server, calls = self._run(
            monkeypatch, images=[], pixlstash_ids=[41], imported=[]
        )
        assert calls["person"] == (None, [41])

    def test_ids_reported_by_the_saver_are_stacked_and_announced(self, monkeypatch):
        server, calls = self._run(
            monkeypatch,
            images=[],
            pixlstash_ids=[41, 42],
        )
        assert calls["downloads"] == 0
        assert calls["stacked"] == (7, [41, 42])
        assert calls["sourced"] == (None, [41, 42])
        # The import, then the end of the run: the order a tab following the
        # run needs, so its row leaves once the pictures are there.
        assert server.vault.events[0] == (
            EventType.PICTURE_IMPORTED,
            {"ids": [41, 42], "source": "ui", "change_kind": "added"},
        )
        assert [self._ending(event) for event in server.vault.events[1:]] == [
            ("comfyui-prompt-1", "completed")
        ]

    def test_a_mixed_graph_merges_both_sets_of_ids(self, monkeypatch):
        _server, calls = self._run(
            monkeypatch,
            images=[{"filename": "out_00001.png", "subfolder": "", "type": "output"}],
            pixlstash_ids=[41],
            imported=[40],
        )
        assert calls["downloads"] == 1
        assert calls["stacked"] == (7, [40, 41])

    def test_a_saver_run_with_no_new_pictures_emits_no_import_event(self, monkeypatch):
        # All duplicates. Nothing to stack, nothing to announce, and crucially
        # not reported as "ComfyUI finished without outputs" either.
        server, calls = self._run(monkeypatch, images=[], pixlstash_ids=[])
        assert calls["downloads"] == 0
        assert calls["stacked"] is None
        # The run still ended, and its row must not wait for a picture that
        # is never coming.
        assert [self._ending(event) for event in server.vault.events] == [
            ("comfyui-prompt-1", "completed")
        ]

    def test_a_genuinely_empty_run_still_reports_failure(self, monkeypatch):
        # No saver ran and nothing was written: the pre-existing failure path
        # must not be swallowed by the new "ids is not None" branch.
        server, _calls = self._run(monkeypatch, images=[], pixlstash_ids=None)
        assert [event for event, _payload in server.vault.events] == [
            EventType.PLUGIN_PROGRESS
        ]
        assert self._ending(server.vault.events[0]) == ("comfyui-prompt-1", "failed")

    def test_an_empty_run_reports_the_outputs_comfyui_dropped(self, monkeypatch):
        # ComfyUI accepted the prompt but dropped the save output at validation;
        # its reason, not "finished without outputs", is what the owner sees.
        server, _calls = self._run(
            monkeypatch,
            images=[],
            pixlstash_ids=None,
            rejected="KSamplerAdvanced (node 94): sampler_name: 'res_2s' not in []",
        )
        assert server.vault.events[0][1]["message"] == (
            "KSamplerAdvanced (node 94): sampler_name: 'res_2s' not in []"
        )

    def test_a_finished_run_with_no_outputs_stops_waiting(self, monkeypatch):
        log = _comfyui_answers(monkeypatch, [_history({})], [GONE])
        assert comfyui_service._wait_for_comfyui_outputs(
            "http://comfy", "prompt-1", ["9"]
        ) == ([], None)
        assert log == ["sleep", "queue", "history"]

    def test_outputs_are_imported_from_files_that_go_with_the_run(
        self, monkeypatch, tmp_path
    ):
        """Each output is fetched to a file in a folder of the run's own, the
        import is handed those files, and the folder is gone once the run
        ends. An output with no bytes is not imported."""
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        bodies = {
            "out_00001.png": b"test-first-" * 20000,
            "out_00002.png": b"test-second-" * 20000,
            "out_00003.png": b"",
        }
        _serve_views(monkeypatch, bodies)
        handed = []

        def import_outputs(srv, entries, **kwargs):
            for path, ext in entries:
                with open(path, "rb") as handle:
                    handed.append((os.path.dirname(path), handle.read(), ext))
            return [40, 41], []

        server, _calls = self._run(
            monkeypatch,
            images=[{"filename": name} for name in bodies],
            pixlstash_ids=None,
            fetch=True,
            import_outputs=import_outputs,
        )

        assert [(body, ext) for _folder, body, ext in handed] == [
            (bodies["out_00001.png"], ".png"),
            (bodies["out_00002.png"], ".png"),
        ]
        (folder,) = {folder for folder, _body, _ext in handed}
        assert os.path.dirname(folder) == str(tmp_path)
        assert os.path.basename(folder).startswith("pixlstash-comfyui-")
        assert list(tmp_path.iterdir()) == []
        assert self._ending(server.vault.events[-1]) == (
            "comfyui-prompt-1",
            "completed",
        )

    @pytest.mark.parametrize("fails", ["the import", "the fetch"])
    def test_the_fetched_files_go_with_a_run_that_fails(
        self, monkeypatch, tmp_path, caplog, fails
    ):
        """The run's folder is gone whichever step failed, and a failed import
        ends the run with the server's sentence: the OS's text, which can name
        paths, goes to the log with its traceback."""
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        _serve_views(
            monkeypatch,
            {"out_00001.png": b"test-output-" * 20000},
            cut_after=1 if fails == "the fetch" else None,
        )
        cause = OSError(28, "test-no-space-left")

        def import_outputs(srv, entries, **kwargs):
            raise cause

        server, _calls = self._run(
            monkeypatch,
            images=[{"filename": "out_00001.png"}],
            pixlstash_ids=None,
            fetch=True,
            import_outputs=import_outputs,
        )

        assert list(tmp_path.iterdir()) == []
        payload = server.vault.events[-1][1]
        assert (payload["status"], payload["message"]) == (
            "failed",
            comfyui_service.OUTPUT_IMPORT_FAILED
            if fails == "the import"
            else "ComfyUI image fetch failed",
        )
        assert "test-no-space-left" not in repr(server.vault.events)
        logged = any(
            record.exc_info is not None and record.exc_info[1] is cause
            for record in caplog.records
        )
        assert logged is (fails == "the import")


class TestVideoSavers:
    """A video saver's file is collected as a ``SaveImage``'s is."""

    def test_a_video_saver_is_the_output_node_and_a_preview_is_not(self):
        graph = {
            "7": _node("SaveVideo", video=["6", 0], filename_prefix="video/out"),
            "8": _node("PreviewImage", images=["5", 0]),
        }
        assert comfyui_service._extract_output_node_ids(graph, {}) == ["7"]

    def test_a_saved_video_is_listed_for_download(self):
        # What ComfyUI's history holds for SaveVideo: the file under `images`.
        payload = _history(
            {
                "7": {
                    "images": [
                        {
                            "filename": "out_00001_.mp4",
                            "subfolder": "video",
                            "type": "output",
                        }
                    ],
                    "animated": [True],
                }
            }
        )
        assert comfyui_service._extract_comfyui_output_images(
            payload, "prompt-1", ["7"]
        ) == [{"filename": "out_00001_.mp4", "subfolder": "video", "type": "output"}]

    def test_a_video_helper_suite_file_is_listed_for_download(self):
        # VHS_VideoCombine reports under `gifs`, with fields of its own.
        payload = _history(
            {
                "7": {
                    "gifs": [
                        {
                            "filename": "AnimateDiff_00001.mp4",
                            "subfolder": "",
                            "type": "output",
                            "format": "video/h264-mp4",
                        }
                    ]
                }
            }
        )
        assert comfyui_service._extract_comfyui_output_images(
            payload, "prompt-1", ["7"]
        ) == [{"filename": "AnimateDiff_00001.mp4", "subfolder": "", "type": "output"}]

    def test_a_video_helper_suite_preview_is_not_collected(self):
        # `save_output` off: a `temp` file its author chose not to keep. The
        # SaveImage beside it is still the run's output.
        payload = _history(
            {
                "7": {"gifs": [{"filename": "AnimateDiff_00001.mp4", "type": "temp"}]},
                "8": {"images": [{"filename": "out_00001_.png"}]},
            }
        )
        images = comfyui_service._extract_comfyui_output_images(
            payload, "prompt-1", ["7", "8"]
        )
        assert [image["filename"] for image in images] == ["out_00001_.png"]

    def test_an_animation_keeps_its_frames_and_a_video_latent_is_one_video(self):
        # AnimateDiff's frames are a batch of pictures; pinned to 1, the run
        # would save a one-frame video. A video latent counts frames in
        # `length`, so its batch is still how many videos.
        graph = {
            "1": _node("EmptyLatentImage", width=512, height=512, batch_size=16),
            "2": _node("EmptyHunyuanLatentVideo", length=33, batch_size=2),
            "7": _node("VHS_VideoCombine", images=["1", 0], filename_prefix="out"),
        }
        assert workflow_run_service.pin_batch_size(graph) == ["2"]
        assert graph["1"]["inputs"]["batch_size"] == 16
        assert graph["2"]["inputs"]["batch_size"] == 1
        # Saving pictures, the same batch is how many pictures: pinned.
        graph["7"] = _node("SaveImage", images=["1", 0], filename_prefix="out")
        assert workflow_run_service.pin_batch_size(graph) == ["1"]


class TestWaitingOutARun:
    """The wait lasts while ComfyUI's queue holds the prompt, however long."""

    DONE = _history({"7": {"images": [{"filename": "out_00001_.mp4"}]}})

    def test_a_prompt_still_in_the_queue_is_waited_for_however_long(self, monkeypatch):
        log = _comfyui_answers(
            monkeypatch, [self.DONE], [RUNNING, WAITING, RUNNING, GONE]
        )
        images, _ids = comfyui_service._wait_for_comfyui_outputs(
            "http://comfy", "prompt-1", ["7"]
        )
        assert [image["filename"] for image in images] == ["out_00001_.mp4"]
        # Its history is read once, when the queue no longer holds it.
        assert log == ["sleep", "queue"] * 4 + ["history"]
        # And nothing is kept for the address once no run waits on it.
        assert comfyui_service._queue_reads == {}
        assert comfyui_service._queue_waits == {}

    def test_a_prompt_comfyui_no_longer_has_is_a_stopped_run(self, monkeypatch):
        """In neither the queue nor the history: taken off the queue before
        its turn, or lost to a restart. Said so, not "finished"."""
        log = _comfyui_answers(monkeypatch, [{}], [GONE])
        with pytest.raises(comfyui_service.ComfyUIRunStopped, match="no longer has"):
            comfyui_service._wait_for_comfyui_outputs("http://comfy", "prompt-1", ["7"])
        assert log == ["sleep", "queue", "history"]

    def test_the_apps_own_abort_is_noticed_at_once(self, monkeypatch):
        """Abort clears ComfyUI's queue, and a prompt that was waiting in it
        never reaches the history: its waiter goes by a read of the queue sent
        after the clear, never one sent before it, so the run is said to have
        stopped before the owner has started the next one."""
        for registry in ("_run_outcomes", "_run_ids", "_live_runs", "_held_cancels"):
            monkeypatch.setattr(comfyui_service, registry, {})
        log = _comfyui_answers(monkeypatch, [{}], [RUNNING, GONE])
        monkeypatch.setattr(
            comfyui_service,
            "bounded_request",
            lambda *a, **k: _answered(200, b""),
        )

        # While the wait sleeps, the listener reads the queue (it still holds
        # the prompt) and the owner aborts. A second sleep means the wait went
        # by the read from before the abort.
        def abort_while_asleep(_s):
            assert "sleep" not in log, "kept waiting after the queue was cleared"
            log.append("sleep")
            comfyui_service._note_running_prompts("http://comfy")
            comfyui_service._comfyui_abort("http://comfy")

        monkeypatch.setattr(
            comfyui_service, "time", SimpleNamespace(sleep=abort_while_asleep)
        )
        with pytest.raises(comfyui_service.ComfyUIRunStopped):
            comfyui_service._wait_for_comfyui_outputs("http://comfy", "prompt-1", None)
        assert log == ["sleep", "queue", "queue", "history"]

    @pytest.mark.parametrize("status, cleared", [(200, True), (500, False)])
    def test_only_a_queue_that_was_cleared_sends_the_pollers_to_ask(
        self, monkeypatch, status, cleared
    ):
        """A clear ComfyUI refused emptied nothing, so there is nothing new
        for a poller to learn from the queue."""
        monkeypatch.setattr(
            comfyui_service,
            "bounded_request",
            lambda *a, **k: _answered(status, b""),
        )
        before = comfyui_service._queue_cleared
        assert (
            comfyui_service._comfyui_abort("http://comfy")["queue_cleared"] is cleared
        )
        assert (comfyui_service._queue_cleared != before) is cleared

    def test_the_cleared_count_changes_only_under_its_lock(self):
        """``_queue_cleared`` is what tells a waiter a read is stale; a bump
        racing a read's ``answers`` check could let a waiter go by a queue
        read from before a clear. Every write to it is inside
        ``with _run_outcomes_lock``."""
        tree = ast.parse(inspect.getsource(comfyui_service))
        unlocked = []

        def visit(node, locked):
            if isinstance(node, ast.With):
                locked = locked or any(
                    isinstance(item.context_expr, ast.Name)
                    and item.context_expr.id == "_run_outcomes_lock"
                    for item in node.items
                )
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            if not isinstance(node, ast.Module) and any(
                isinstance(t, ast.Name) and t.id == "_queue_cleared" for t in targets
            ):
                if not locked:
                    unlocked.append(node.lineno)
            for child in ast.iter_child_nodes(node):
                # The module-level initial value is not a change.
                if isinstance(node, ast.Module) and isinstance(child, ast.Assign):
                    continue
                visit(child, locked)

        visit(tree, False)
        writes = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_queue_cleared"
        ]
        assert writes, "the sweep found no bump of _queue_cleared at all"
        assert unlocked == [], f"_queue_cleared written outside its lock: {unlocked}"

    def test_a_clear_that_never_reached_comfyui_sends_nobody(self, monkeypatch):
        def refuse(*a, **k):
            raise comfyui_service.requests.ConnectionError("refused")

        monkeypatch.setattr(comfyui_service, "bounded_request", refuse)
        before = comfyui_service._queue_cleared
        comfyui_service._comfyui_abort("http://comfy")
        assert comfyui_service._queue_cleared == before

    def test_a_prompt_in_the_history_with_no_ending_finishes_empty(self, monkeypatch):
        # ComfyUI still has it on record, so it ran: nothing to import, and
        # the caller says "finished without outputs".
        unfinished = {"prompt-1": {"outputs": {}}}
        log = _comfyui_answers(monkeypatch, [unfinished], [GONE])
        assert comfyui_service._wait_for_comfyui_outputs(
            "http://comfy", "prompt-1", ["7"]
        ) == ([], None)
        assert log == ["sleep", "queue", "history"]

    def test_a_prompt_that_finishes_as_it_leaves_the_queue_is_collected(
        self, monkeypatch
    ):
        """The queue is asked BEFORE the history is read. The other way round,
        a prompt that finishes between the two reads is in neither answer and
        its video is never collected."""
        log = _comfyui_answers(monkeypatch, [], [GONE])
        # The prompt finishes at the moment the queue is asked about it.
        monkeypatch.setattr(
            comfyui_service,
            "_fetch_comfyui_history",
            lambda *a: self.DONE if "queue" in log else {},
        )
        images, _ids = comfyui_service._wait_for_comfyui_outputs(
            "http://comfy", "prompt-1", ["7"]
        )
        assert [image["filename"] for image in images] == ["out_00001_.mp4"]

    @pytest.mark.parametrize(
        "queue, answer",
        [
            (
                {"queue_running": [[3, "prompt-1", {}]], "queue_pending": []},
                (frozenset({"prompt-1"}), frozenset()),
            ),
            (
                {
                    "queue_running": [],
                    "queue_pending": [[4, "prompt-1", {}], [5, "prompt-2", {}]],
                },
                (frozenset(), frozenset({"prompt-1", "prompt-2"})),
            ),
            (
                {"queue_running": [[3, "other", {}]]},
                (frozenset({"other"}), frozenset()),
            ),
            # Not an answer about the queue at all: unread, not empty.
            ([], None),
        ],
    )
    def test_the_queue_is_read_for_its_prompts(self, monkeypatch, queue, answer):
        asked = []

        def get(method, url, proxies=None, **kwargs):
            asked.append((method, url, kwargs.get("redirects", 0), proxies))
            return _answered(200, json.dumps(queue).encode())

        monkeypatch.setattr(comfyui_service, "bounded_request", get)
        assert comfyui_service._read_comfyui_queue("http://comfy") == answer
        # Straight to the address asked, never through an environment proxy,
        # and never on to wherever a redirect points.
        assert asked == [("GET", "http://comfy/queue", 0, comfyui_service.NO_PROXY)]

    @pytest.mark.parametrize(
        "get",
        [
            _raising(comfyui_service.requests.ConnectionError("refused")),
            lambda *a, **k: _answered(503, b"{}"),
            # Another host's address, whatever the body says.
            lambda *a, **k: _answered(
                302, b'{"queue_running": [], "queue_pending": []}'
            ),
            lambda *a, **k: _answered(200, b"<html>not json</html>"),
        ],
        ids=["refused", "error-status", "redirect", "not-json"],
    )
    def test_a_queue_that_cannot_be_read_is_no_answer(self, monkeypatch, get):
        monkeypatch.setattr(comfyui_service, "bounded_request", get)
        assert comfyui_service._read_comfyui_queue("http://comfy") is None

    def test_a_queue_that_cannot_be_read_does_not_abandon_the_run(self, monkeypatch):
        """Each pass then reads the prompt's history instead, as a lone run
        would, and the wait goes on while the history has no entry for it."""
        log = _comfyui_answers(monkeypatch, [{}, {}, self.DONE], [None, None, GONE])
        images, _ids = comfyui_service._wait_for_comfyui_outputs(
            "http://comfy", "prompt-1", ["7"]
        )
        assert [image["filename"] for image in images] == ["out_00001_.mp4"]
        assert log == ["sleep", "queue", "history"] * 3


COMFY = "http://comfy"
GUARD_S = 10.0
"""How long a test waits on one of its threads before calling it stuck."""


class _TestOver(BaseException):
    """Ends a waiter its test left waiting."""


class _Unexpected(Exception):
    """Not a queue that could not be read, which ``_read_comfyui_queue``
    answers as no answer, but anything else raised while reading it."""


class _Rounds:
    """Runs waited on at one ComfyUI, each in its own thread as the app waits
    on them. A waiter's sleep lasts until the test lets it go, so a round is
    each waiter going by the queue once, and no real time passes. Keeps how
    each wait ended, and who waits on a read of the queue another sent."""

    def __init__(self):
        self._turn = threading.Condition()
        self._asleep: set[str] = set()
        self._joining: set[str] = set()
        self._over = False
        self._threads: list[threading.Thread] = []
        self.outcomes: dict[str, object] = {}

    def sleep(self, _seconds):
        name = threading.current_thread().name
        with self._turn:
            self._asleep.add(name)
            self._turn.notify_all()
            woken = self._turn.wait_for(
                lambda: name not in self._asleep or self._over, GUARD_S
            )
        if self._over:
            raise _TestOver
        assert woken, f"{name} was never woken"

    def start(self, *prompt_ids):
        """A waiter on each of *prompt_ids*; back once all of them sleep."""
        for prompt_id in prompt_ids:
            thread = threading.Thread(
                target=self._wait,
                args=(prompt_id,),
                name=f"waiter-{prompt_id}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        self.settle()

    def _wait(self, prompt_id):
        try:
            outcome = comfyui_service._wait_for_comfyui_outputs(COMFY, prompt_id, None)
        except BaseException as exc:  # How the wait ended is what is tested.
            outcome = exc
        with self._turn:
            self.outcomes[prompt_id] = outcome
            self._turn.notify_all()

    def settle(self):
        """Back once every waiter sleeps or has ended."""
        with self._turn:
            settled = self._turn.wait_for(
                lambda: all(
                    thread.name in self._asleep
                    or thread.name.removeprefix("waiter-") in self.outcomes
                    for thread in self._threads
                ),
                GUARD_S,
            )
        assert settled, "a waiter neither slept nor ended"

    def round(self, *prompt_ids):
        """Wake the waiters on *prompt_ids*, or every one, and come back once
        each sleeps again or has ended."""
        with self._turn:
            if prompt_ids:
                self._asleep -= {f"waiter-{prompt_id}" for prompt_id in prompt_ids}
            else:
                self._asleep.clear()
            self._turn.notify_all()
        self.settle()

    def asleep(self) -> set[str]:
        """The prompt ids of the waiters asleep."""
        with self._turn:
            return {name.removeprefix("waiter-") for name in self._asleep}

    def joined(self, count: int) -> None:
        """Back once *count* waiters wait on a read of the queue another sent."""
        with self._turn:
            joined = self._turn.wait_for(lambda: len(self._joining) == count, GUARD_S)
        assert joined, f"{count} waiters never shared the read in flight"

    def read_type(self):
        """``_QueueRead``, whose ``done`` notes who waits on it and lets them
        go when the test ends. A waiter left on a read that never ended fails
        rather than hanging the run of tests."""
        rounds = self

        class _Done(threading.Event):
            def set(self):
                super().set()
                with rounds._turn:
                    rounds._turn.notify_all()

            def wait(self, timeout=None):
                name = threading.current_thread().name
                with rounds._turn:
                    rounds._joining.add(name)
                    rounds._turn.notify_all()
                    ended = rounds._turn.wait_for(
                        lambda: self.is_set() or rounds._over, GUARD_S
                    )
                    rounds._joining.discard(name)
                if rounds._over:
                    raise _TestOver
                assert ended, f"{name} was left waiting on a read that never ended"
                return True

        class _WatchedRead(comfyui_service._QueueRead):
            def __init__(self, number, cleared):
                super().__init__(number, cleared)
                self.done = _Done()

        return _WatchedRead

    def stop(self):
        """End every waiter still waiting, and join them all."""
        with self._turn:
            self._over = True
            self._turn.notify_all()
        for thread in self._threads:
            thread.join(GUARD_S)
        assert not [t.name for t in self._threads if t.is_alive()], (
            "a waiter outlived its test"
        )


class _FakeComfyUI:
    """A ComfyUI's queue and history, answered in process as ``bounded_request``.
    Notes each request made of it with whether a redirect would be followed
    and the proxies it was sent with."""

    def __init__(self, rounds: _Rounds):
        self._rounds = rounds
        self._lock = threading.Lock()
        self._asked: list[tuple[str, object, object]] = []
        self.running: list[str] = []
        self.pending: list[str] = []
        self.histories: dict[str, dict] = {}
        # (exception, how many other waiters share the read before it raises)
        self.queue_failure: tuple[BaseException, int] | None = None

    def get(self, method, url, proxies=None, **kwargs):
        """``bounded_request`` at this ComfyUI: the queue and the histories."""
        assert method == "GET"
        failure = None
        with self._lock:
            # (where, whether a redirect would be followed, the proxies)
            self._asked.append((url, bool(kwargs.get("redirects", 0)), proxies))
            if url == f"{COMFY}/queue":
                failure, self.queue_failure = self.queue_failure, None
                body = {
                    "queue_running": [[1, pid, {}] for pid in self.running],
                    "queue_pending": [
                        [2 + n, pid, {}] for n, pid in enumerate(self.pending)
                    ],
                }
            else:
                prompt_id = url.removeprefix(f"{COMFY}/history/")
                entry = self.histories.get(prompt_id)
                body = {} if entry is None else {prompt_id: entry}
        if failure is not None:
            exc, sharers = failure
            self._rounds.joined(sharers)
            raise exc
        return _answered(200, json.dumps(body).encode())

    def taken(self) -> list[tuple[str, object, object]]:
        """The requests made since this was last asked, in order."""
        with self._lock:
            asked, self._asked = self._asked, []
        return asked


@pytest.fixture
def comfyui(monkeypatch):
    """A fake ComfyUI and the runs waited on at it; each waiter is ended,
    and joined, with the test."""
    rounds = _Rounds()
    fake = _FakeComfyUI(rounds)
    monkeypatch.setattr(comfyui_service, "_queue_reads", {})
    monkeypatch.setattr(comfyui_service, "_queue_waits", {})
    monkeypatch.setattr(comfyui_service, "_QueueRead", rounds.read_type())
    monkeypatch.setattr(comfyui_service, "time", SimpleNamespace(sleep=rounds.sleep))
    monkeypatch.setattr(comfyui_service, "bounded_request", fake.get)
    yield fake, rounds
    rounds.stop()


QUEUE_READ = (f"{COMFY}/queue", False, comfyui_service.NO_PROXY)


def _history_read(prompt_id: str) -> tuple[str, object, object]:
    return (f"{COMFY}/history/{prompt_id}", False, comfyui_service.NO_PROXY)


def _saved(prompt_id: str) -> dict:
    """ComfyUI's history entry for a prompt that saved one picture."""
    return {
        "outputs": {"9": {"images": [{"filename": f"{prompt_id}.png"}]}},
        "status": {"status_str": "success"},
    }


def _collected(prompt_id: str) -> tuple[list[dict], None]:
    """What the wait for that prompt returns: its picture, to download."""
    return [{"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}], None


class TestOneQueueReadForEveryRun:
    """However many runs wait at one ComfyUI, each round asks it about its
    queue once; each run still ends its own way, and nothing is left behind
    once none waits."""

    @pytest.mark.parametrize("count", [1, 4])
    def test_a_round_reads_the_queue_once_however_many_runs_wait(self, comfyui, count):
        fake, rounds = comfyui
        before = set(threading.enumerate())
        prompt_ids = [f"test-{n}" for n in range(count)]
        fake.running, fake.pending = prompt_ids[:1], prompt_ids[1:]
        rounds.start(*prompt_ids)
        assert fake.taken() == []
        for _ in range(3):
            rounds.round()
            # Straight to ComfyUI, never through an environment proxy.
            assert fake.taken() == [QUEUE_READ]
        # ComfyUI finishes them all: the queue no longer holds them, and each
        # run reads its own history.
        fake.running, fake.pending = [], []
        fake.histories = {prompt_id: _saved(prompt_id) for prompt_id in prompt_ids}
        rounds.round()
        asked = fake.taken()
        assert asked[0] == QUEUE_READ
        assert sorted(asked[1:]) == sorted(map(_history_read, prompt_ids))
        assert rounds.outcomes == {
            prompt_id: _collected(prompt_id) for prompt_id in prompt_ids
        }
        # Nothing is kept for the address, and no thread is left behind.
        assert comfyui_service._queue_reads == {}
        assert comfyui_service._queue_waits == {}
        rounds.stop()
        assert set(threading.enumerate()) - before == set()

    def test_each_run_ends_its_own_way(self, comfyui):
        fake, rounds = comfyui
        fake.running = ["test-done"]
        fake.pending = ["test-failed", "test-gone", "test-later"]
        rounds.start("test-done", "test-failed", "test-gone", "test-later")
        rounds.round()
        assert fake.taken() == [QUEUE_READ]
        # One finishes, one fails, one is taken off the queue, one waits on.
        fake.running, fake.pending = ["test-later"], []
        fake.histories = {
            "test-done": _saved("test-done"),
            "test-failed": _ended("execution_error", ERROR_EVENT)["prompt-1"],
        }
        rounds.round()
        asked = fake.taken()
        assert asked[0] == QUEUE_READ
        assert sorted(asked[1:]) == sorted(
            map(_history_read, ["test-done", "test-failed", "test-gone"])
        )
        assert rounds.asleep() == {"test-later"}
        outcomes = rounds.outcomes
        assert outcomes["test-done"] == _collected("test-done")
        assert type(outcomes["test-failed"]) is comfyui_service.ComfyUIRunFailed
        assert "(UNETLoader) failed: KeyError" in str(outcomes["test-failed"])
        assert type(outcomes["test-gone"]) is comfyui_service.ComfyUIRunStopped
        assert "no longer has this run" in str(outcomes["test-gone"])

        fake.running = []
        fake.histories["test-later"] = _saved("test-later")
        rounds.round()
        assert fake.taken() == [QUEUE_READ, _history_read("test-later")]
        assert outcomes["test-later"] == _collected("test-later")
        assert comfyui_service._queue_reads == {}
        assert comfyui_service._queue_waits == {}

    def test_a_read_that_raises_ends_only_the_run_that_sent_it(self, comfyui):
        """The runs sharing it are not left waiting on it: the next sends a
        read of its own, and they wait on as before."""
        fake, rounds = comfyui
        prompt_ids = ["test-a", "test-b", "test-c"]
        fake.pending = list(prompt_ids)
        rounds.start(*prompt_ids)
        failure = _Unexpected("test-unexpected")
        fake.queue_failure = (failure, 2)
        rounds.round()
        assert fake.taken() == [QUEUE_READ, QUEUE_READ]
        (sender,) = rounds.outcomes
        assert rounds.outcomes[sender] is failure
        others = set(prompt_ids) - {sender}
        assert rounds.asleep() == others

        fake.pending = []
        fake.histories = {prompt_id: _saved(prompt_id) for prompt_id in others}
        rounds.round()
        assert {p: rounds.outcomes[p] for p in others} == {
            prompt_id: _collected(prompt_id) for prompt_id in others
        }
        assert comfyui_service._queue_reads == {}
        assert comfyui_service._queue_waits == {}

    def test_a_run_goes_by_no_read_sent_before_it_was_queued(self, comfyui):
        """Such a read predates the prompt, so it does not list it: gone by,
        it would read as ComfyUI no longer having the run."""
        fake, rounds = comfyui
        fake.running = ["test-first"]
        rounds.start("test-first")
        rounds.round()
        assert fake.taken() == [QUEUE_READ]
        # That read is kept for the run still waiting on it. A run queued
        # since wakes first.
        fake.pending = ["test-second"]
        rounds.start("test-second")
        rounds.round("test-second")
        assert fake.taken() == [QUEUE_READ]
        assert rounds.outcomes == {}
        assert rounds.asleep() == {"test-first", "test-second"}

    def test_a_run_that_ends_as_the_read_kept_for_it_ends_leaves_nothing_kept(
        self, monkeypatch
    ):
        """A read another sent is kept for the run still waiting, and that run
        ends just as the read does: once neither is left, nothing is kept for
        the address, whichever of the two went first."""
        go = threading.Event()
        asleep = threading.Event()
        outcomes = []

        def sleep(_seconds):
            # The run sleeps until the read is all but over, then ends, as a
            # run does once it has read its own history.
            asleep.set()
            assert go.wait(GUARD_S), "the read never came to an end"
            raise _TestOver

        class _EndsTheRunFirst(threading.Event):
            def set(self):
                go.set()
                # Long enough for the run to end, unless it must first take
                # the lock the read is set under.
                waiter.join(1.0)
                super().set()

        class _Read(comfyui_service._QueueRead):
            def __init__(self, number, cleared):
                super().__init__(number, cleared)
                self.done = _EndsTheRunFirst()

        def wait():
            try:
                comfyui_service._wait_for_comfyui_outputs(COMFY, "test-ending", None)
            except BaseException as exc:  # How the wait ended is what is tested.
                outcomes.append(exc)

        monkeypatch.setattr(comfyui_service, "_queue_reads", {})
        monkeypatch.setattr(comfyui_service, "_queue_waits", {})
        monkeypatch.setattr(comfyui_service, "_QueueRead", _Read)
        monkeypatch.setattr(comfyui_service, "time", SimpleNamespace(sleep=sleep))
        monkeypatch.setattr(comfyui_service, "_read_comfyui_queue", lambda _url: GONE)
        waiter = threading.Thread(target=wait, name="waiter-test-ending", daemon=True)
        waiter.start()
        assert asleep.wait(GUARD_S), "the run never waited"
        # Meanwhile the listener reads the queue, as it does on opening its
        # socket.
        read = comfyui_service._shared_queue_read(
            COMFY, comfyui_service._queue_reads_sent
        )
        waiter.join(GUARD_S)
        assert not waiter.is_alive(), "the run outlived its test"
        assert read.done.is_set()
        assert [type(exc) for exc in outcomes] == [_TestOver]
        assert comfyui_service._queue_reads == {}
        assert comfyui_service._queue_waits == {}

    def test_a_waiter_does_not_wait_for_ever_on_a_read_another_sent(self, monkeypatch):
        """The read is bounded, but the waiter sharing it must not rest its
        whole wait on that: past the read's bound it reads the prompt's
        history instead, as for a queue that could not be read."""
        stuck = comfyui_service._QueueRead(1, comfyui_service._queue_cleared)
        monkeypatch.setattr(comfyui_service, "_queue_reads", {COMFY: stuck})
        monkeypatch.setattr(comfyui_service, "QUEUE_READ_WAIT_S", 0.05)

        def never_sent(_url):
            raise AssertionError("a waiter sent a second read beside the first")

        monkeypatch.setattr(comfyui_service, "_read_comfyui_queue", never_sent)
        answers = []
        waiter = threading.Thread(
            target=lambda: answers.append(comfyui_service._shared_queue_read(COMFY, 0)),
            name="waiter-test-stuck-read",
            daemon=True,
        )
        waiter.start()
        waiter.join(GUARD_S)
        stuck.done.set()
        assert not waiter.is_alive(), "the waiter waited on the read for ever"
        [read] = answers
        assert read.finished and read.done.is_set()
        assert read.listed is None and read.running is None
        # The read still in flight is the sender's to end, not the waiter's.
        assert comfyui_service._queue_reads == {COMFY: stuck}


ERROR_EVENT = {
    "prompt_id": "prompt-1",
    "node_id": "12",
    "node_type": "UNETLoader",
    "exception_message": "'asym_w4a8_int8'",
    "exception_type": "KeyError",
    "traceback": ["..."],
}
TITLED_GRAPH = {
    "12": {
        "class_type": "UNETLoader",
        "inputs": {},
        "_meta": {"title": "Load Diffusion Model"},
    }
}


def _ended(event_name: str, event: dict, graph: dict | None = TITLED_GRAPH) -> dict:
    """A history answer for a prompt ComfyUI ended with *event_name*."""
    entry = {
        "outputs": {},
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_start", {"prompt_id": "prompt-1"}],
                [event_name, event],
            ],
        },
    }
    if graph is not None:
        entry["prompt"] = [3, "prompt-1", graph, {}, ["9"]]
    return {"prompt-1": entry}


class TestHowARunEnded:
    """A run that ends without a picture says which node ended it and why."""

    def _read(self, *args, **kwargs):
        return comfyui_service._extract_history_status_and_error(
            _ended(*args, **kwargs), "prompt-1"
        )

    def test_a_failed_node_is_named_with_its_title_class_and_exception(self):
        # A KeyError's message is the bare key, which says nothing alone.
        assert self._read("execution_error", ERROR_EVENT) == (
            "error",
            "Load Diffusion Model (UNETLoader) failed: KeyError 'asym_w4a8_int8'",
        )

    @pytest.mark.parametrize(
        "event, graph, expected",
        [
            # A title that only repeats the class is said once.
            (
                ERROR_EVENT,
                {"12": {"class_type": "UNETLoader", "_meta": {"title": "UNETLoader"}}},
                "UNETLoader failed: KeyError 'asym_w4a8_int8'",
            ),
            # No graph on the history entry: the class the event names.
            (ERROR_EVENT, None, "UNETLoader failed: KeyError 'asym_w4a8_int8'"),
            # The event names no class: the graph's.
            (
                {**ERROR_EVENT, "node_type": None},
                TITLED_GRAPH,
                "Load Diffusion Model (UNETLoader) failed: KeyError 'asym_w4a8_int8'",
            ),
            # No node at all: the exception still carries its type.
            (
                {"exception_type": "KeyError", "exception_message": "'x'"},
                None,
                "KeyError 'x'",
            ),
            # One line, however ComfyUI wrapped it.
            (
                {**ERROR_EVENT, "exception_message": "CUDA out of memory.\n  Tried\n"},
                None,
                "UNETLoader failed: KeyError CUDA out of memory. Tried",
            ),
            # A graph that is not shaped like one is not read, and not raised on.
            (
                ERROR_EVENT,
                ["not", "a", "graph"],
                "UNETLoader failed: KeyError 'asym_w4a8_int8'",
            ),
        ],
    )
    def test_the_sentence_is_built_from_what_comfyui_gave(self, event, graph, expected):
        assert self._read("execution_error", event, graph)[1] == expected

    def test_a_long_exception_is_cut_to_what_a_card_can_carry(self):
        # A state_dict size mismatch lists every tensor.
        event = {**ERROR_EVENT, "exception_message": "size mismatch " * 200}
        _status, text = self._read("execution_error", event)
        assert text.startswith(
            "Load Diffusion Model (UNETLoader) failed: KeyError size"
        )
        assert text.endswith("…")
        prefix = len("Load Diffusion Model (UNETLoader) failed: ")
        assert len(text) - prefix == comfyui_service.MAX_FAILURE_REASON + 1

    def test_an_interrupt_is_a_stopped_run_and_names_where(self):
        """ComfyUI files an interrupt under status ``error``; it is told apart
        here, and never shown as the event's JSON."""
        event = {"prompt_id": "prompt-1", "node_id": "12", "node_type": "UNETLoader"}
        assert self._read("execution_interrupted", event) == (
            "interrupted",
            "Interrupted at Load Diffusion Model (UNETLoader)",
        )
        assert self._read("execution_interrupted", {"prompt_id": "prompt-1"}, None) == (
            "interrupted",
            "Interrupted",
        )

    def test_the_wait_tells_a_stopped_run_from_a_failed_one(self, monkeypatch):
        _comfyui_answers(
            monkeypatch,
            [
                _ended("execution_error", ERROR_EVENT),
                _ended("execution_interrupted", {"node_type": "KSampler"}),
            ],
            [GONE, GONE],
        )
        with pytest.raises(
            comfyui_service.ComfyUIRunFailed, match="UNETLoader\\) failed"
        ) as failed:
            comfyui_service._wait_for_comfyui_outputs("http://comfy", "prompt-1", None)
        assert not isinstance(failed.value, comfyui_service.ComfyUIRunStopped)

        with pytest.raises(comfyui_service.ComfyUIRunStopped, match="at KSampler"):
            comfyui_service._wait_for_comfyui_outputs("http://comfy", "prompt-1", None)

    def test_an_error_without_a_failed_status_is_still_a_failed_run(self, monkeypatch):
        """A history carrying ComfyUI's error but no failed status (none at all
        here) ends as a ``ComfyUIRunFailed``, the one class whose text the run
        handler shows; any other ``RuntimeError`` it answers with a fixed
        sentence."""
        _comfyui_answers(
            monkeypatch,
            [
                {
                    "prompt-1": {
                        "outputs": {},
                        "status": {},
                        "error": "Prompt outputs failed validation",
                    }
                }
            ],
            [GONE],
        )
        with pytest.raises(
            comfyui_service.ComfyUIRunFailed,
            match="^Prompt outputs failed validation$",
        ):
            comfyui_service._wait_for_comfyui_outputs("http://comfy", "prompt-1", None)

    def test_comfyui_going_away_says_so(self, monkeypatch):
        def refuse(*a, **k):
            raise comfyui_service.requests.ConnectionError("refused")

        monkeypatch.setattr(comfyui_service, "bounded_request", refuse)
        with pytest.raises(comfyui_service.HTTPException) as gone:
            comfyui_service._fetch_comfyui_history("http://comfy", "prompt-1")
        assert gone.value.detail == "ComfyUI stopped answering before the run finished"


class TestTheEndingIsToldAndKept:
    """Each ending reaches the tab as an event and is kept for the route."""

    def _end(self, monkeypatch, prompt_id, wait, **kwargs):
        server = _FakeServer()
        monkeypatch.setattr(comfyui_service, "_wait_for_comfyui_outputs", wait)
        comfyui_service._process_comfyui_outputs(
            server, "http://comfy", prompt_id, None, None, None, **kwargs
        )
        (event,) = server.vault.events
        return event[1], comfyui_service.run_outcome(prompt_id)

    @staticmethod
    def _raises(exc):
        def wait(*a, **k):
            raise exc

        return wait

    def test_a_failure_names_its_workflow_and_is_not_a_stopped_run(self, monkeypatch):
        payload, outcome = self._end(
            monkeypatch,
            "ending-failed",
            self._raises(
                comfyui_service.ComfyUIRunFailed("UNETLoader failed: KeyError 'x'")
            ),
            workflow_id="manual:" + "a" * 32,
        )
        assert (
            payload["status"],
            payload["message"],
            payload["workflow_id"],
            payload["stopped"],
        ) == ("failed", "UNETLoader failed: KeyError 'x'", "manual:" + "a" * 32, False)
        assert outcome == {
            "prompt_id": "ending-failed",
            "status": "failed",
            "message": "UNETLoader failed: KeyError 'x'",
            "stopped": False,
            "picture_ids": [],
            "library_uuid": None,
            "run_id": None,
            "code": None,
        }

    def test_a_library_fault_during_the_import_is_logged_not_shown(
        self, monkeypatch, caplog
    ):
        """Only ComfyUI's account of a failed run is the ending's text. A
        ``RuntimeError`` met while the outputs are imported is a fault of the
        import: ``DatabaseUnavailableError`` is one, and its text names the
        writer's failure. The ending used to show any ``RuntimeError``.

        Wrong if ``test-leak`` is in the event or the kept outcome, or if the
        log does not hold the cause with its traceback.
        """
        cause = DatabaseUnavailableError(
            "VaultDatabase writer has failed (test-leak disk I/O error); task refused."
        )

        def write_fails(*_args):
            raise cause

        monkeypatch.setattr(
            comfyui_service, "_set_source_picture_id_on_pictures", write_fails
        )
        with caplog.at_level(logging.ERROR, logger=comfyui_service.__name__):
            payload, outcome = self._end(
                monkeypatch, "ending-writer-failed", lambda *a, **k: ([], [41])
            )
        assert (payload["status"], payload["message"], payload["stopped"]) == (
            "failed",
            comfyui_service.OUTPUT_IMPORT_FAILED,
            False,
        )
        assert outcome["message"] == comfyui_service.OUTPUT_IMPORT_FAILED
        assert "test-leak" not in repr(payload)
        assert "test-leak" not in repr(outcome)
        assert any(
            record.exc_info is not None and record.exc_info[1] is cause
            for record in caplog.records
        )

    def test_a_stopped_run_is_marked_stopped(self, monkeypatch):
        payload, outcome = self._end(
            monkeypatch,
            "ending-stopped",
            self._raises(comfyui_service.ComfyUIRunStopped("Interrupted at KSampler")),
        )
        assert (payload["status"], payload["stopped"]) == ("failed", True)
        assert (outcome["status"], outcome["stopped"], outcome["message"]) == (
            "failed",
            True,
            "Interrupted at KSampler",
        )

    def test_comfyui_going_away_is_told_in_words_not_a_status_code(self, monkeypatch):
        # str() of an HTTPException is "502: ..."; the detail is the sentence.
        payload, outcome = self._end(
            monkeypatch,
            "ending-gone",
            self._raises(
                comfyui_service.HTTPException(
                    status_code=502,
                    detail="ComfyUI stopped answering before the run finished",
                )
            ),
        )
        assert payload["message"] == "ComfyUI stopped answering before the run finished"
        assert outcome["message"] == payload["message"] and not outcome["stopped"]

    def test_a_finished_run_is_kept_with_the_pictures_it_added(self, monkeypatch):
        monkeypatch.setattr(
            comfyui_service, "_set_source_picture_id_on_pictures", lambda *a: None
        )
        monkeypatch.setattr(
            comfyui_service, "_copy_set_and_project_assignments", lambda *a: None
        )
        server = _FakeServer()
        monkeypatch.setattr(
            comfyui_service, "_wait_for_comfyui_outputs", lambda *a, **k: ([], [41, 42])
        )
        comfyui_service._process_comfyui_outputs(
            server,
            "http://comfy",
            "ending-done",
            None,
            None,
            None,
            workflow_id="auto:x",
        )
        assert server.vault.events[-1][1]["workflow_id"] == "auto:x"
        assert comfyui_service.run_outcome("ending-done") == {
            "prompt_id": "ending-done",
            "status": "completed",
            "message": None,
            "stopped": False,
            "picture_ids": [41, 42],
            "library_uuid": None,
            "run_id": None,
            "code": None,
        }

    def test_a_run_that_made_only_what_the_library_has_says_so(self, monkeypatch):
        """Exact duplicates are not imported twice, so a run repeated on its
        own seed adds nothing; the ending is where the owner hears why."""
        monkeypatch.setattr(
            comfyui_service,
            "_wait_for_comfyui_outputs",
            lambda *a, **k: ([{"filename": "out.png"}], None),
        )

        def fetched(_base_url, _entry, dest_dir):
            path = os.path.join(dest_dir, "test-out.png")
            with open(path, "wb") as handle:
                handle.write(b"png")
            return path, ".png"

        monkeypatch.setattr(comfyui_service, "_download_comfyui_image", fetched)
        monkeypatch.setattr(
            comfyui_service, "_import_comfyui_outputs", lambda *a, **k: ([], [7, 8])
        )
        # 8 is this run's own file, imported first by a watched output folder.
        pictures = {
            7: SimpleNamespace(imported_at=datetime(2020, 1, 1, tzinfo=timezone.utc)),
            8: SimpleNamespace(imported_at=datetime(2999, 1, 1, tzinfo=timezone.utc)),
        }
        server = _FakeServer()
        server.vault.db = SimpleNamespace(
            run_immediate_read_task=lambda fn: fn(
                SimpleNamespace(get=lambda model, pid: pictures.get(pid))
            )
        )
        comfyui_service._process_comfyui_outputs(
            server, "http://comfy", "ending-duplicate", None, None, None
        )
        (event,) = server.vault.events
        outcome = comfyui_service.run_outcome("ending-duplicate")
        assert outcome["status"] == "completed" and outcome["picture_ids"] == []
        assert "made a picture this library already has" in outcome["message"]
        assert event[1]["status"] == "completed"
        assert event[1]["notice"] == outcome["message"]

        # Nothing but the run's own file, already imported: nothing to say.
        del pictures[7]
        comfyui_service._process_comfyui_outputs(
            server, "http://comfy", "ending-watched", None, None, None
        )
        assert comfyui_service.run_outcome("ending-watched")["message"] is None
        assert server.vault.events[-1][1]["notice"] is None

    def test_only_the_newest_outcomes_are_kept(self, monkeypatch):
        monkeypatch.setattr(comfyui_service, "MAX_RUN_OUTCOMES", 2)
        monkeypatch.setattr(comfyui_service, "_run_outcomes", {})
        for prompt_id in ("kept-1", "kept-2", "kept-3"):
            comfyui_service.record_run_outcome(prompt_id, "running")
        # Nothing has ended, so the oldest goes.
        assert comfyui_service.run_outcome("kept-1") is None
        # An update replaces a run's row rather than adding one, and a run
        # that has ended goes before an older one that is still going.
        comfyui_service.record_run_outcome("kept-3", "completed")
        comfyui_service.record_run_outcome("kept-4", "running")
        assert comfyui_service.run_outcome("kept-3") is None
        assert comfyui_service.run_outcome("kept-2")["status"] == "running"
        assert comfyui_service.run_outcome("kept-4")["status"] == "running"
        assert comfyui_service.run_outcome("never-run") is None

    def test_a_runs_library_is_kept_through_its_ending(self):
        comfyui_service.record_run_outcome("lib-run", "running", library_uuid="lib-a")
        comfyui_service.record_run_outcome("lib-run", "completed", picture_ids=[7])
        assert comfyui_service.run_outcome("lib-run")["library_uuid"] == "lib-a"

    def test_outputs_of_a_library_that_was_switched_away_are_kept_as_a_failure(
        self, monkeypatch
    ):
        """No event (the tab is on another library by then), but a reader of
        the run is not left with "running" for good."""

        class _Gone:
            def acquire_read(self):
                return None

        server = _FakeServer()
        server.library_coordinator = _Gone()
        monkeypatch.setattr(
            comfyui_service, "_wait_for_comfyui_outputs", lambda *a, **k: ([], [41])
        )
        comfyui_service._process_comfyui_outputs(
            server, "http://comfy", "ending-switched", None, None, None
        )
        assert server.vault.events == []
        outcome = comfyui_service.run_outcome("ending-switched")
        assert outcome["status"] == "failed" and "library changed" in outcome["message"]


def demo() -> None:
    """Smoke the extraction split without pytest."""
    payload = _history(
        {
            "8": {"images": [{"filename": "out_00001.png"}]},
            "9": {
                "images": [{"filename": "v_00001.png", "type": "temp"}],
                "picture_ids": ["41,42"],
            },
        }
    )
    assert comfyui_service._extract_pixlstash_picture_ids(
        payload, "prompt-1", None
    ) == [
        41,
        42,
    ]
    assert [
        img["filename"]
        for img in comfyui_service._extract_comfyui_output_images(
            payload, "prompt-1", None
        )
    ] == ["out_00001.png"]
    print("ok")


if __name__ == "__main__":
    demo()
