"""The export of a search view packages the pictures the search shows.

The grid's search view is ``GET /pictures/search`` with the view's filters, and
its "Export" sends the same query string to ``GET /pictures/export``. The
export must answer with the search's pictures, in the search's order: the same
filters, the same Scrapheap meaning, the same ranking. It used to run a search
of its own (a prefixed query, no image-text model, no filters), so a filtered
search exported pictures the view did not show, in another order.

The grid's "In text" switch narrows a search to the pictures whose read text
matches. The search answers that from ``text_match_only``, which the grid and
the export both send, so the export packages the narrowed view too.
"""

import gc
import io
import json
import os
import tempfile
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from sqlmodel import select

from pixlstash.db_models.picture import Picture
from pixlstash.route_inventory import iter_api_query_params
from pixlstash.server import Server
from pixlstash.utils.service.export_utils import EXPORT_OPTION_PARAMS, _ViewRequest
from tests.utils import upload_pictures_and_wait

PICTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "pictures")
EXPORT = "/pictures/export"
SEARCH = {"query": "zebra", "threshold": 0.01}

# Pictures whose read text says "zebra": the ones the "In text" switch keeps.
# The rest are stored as read with no text, so the OCR sweep leaves all six.
TEXT_MATCHES = {"Bad1.png", "Changed1.png", "Changed2.png"}
ZEBRA_TEXT = "Zebra crossing ahead"
# File name, score, and whether the picture is in the Scrapheap.
LIBRARY = [
    ("Bad1.png", 5, False),
    ("Bad2.png", 4, False),
    ("Changed1.png", 1, False),
    ("Changed2.png", 5, True),
    ("Reference1.png", 4, True),
    ("Changed3.png", 1, True),
]


def _server(temp_dir: str) -> tuple[Server, TestClient]:
    server_config_path = os.path.join(temp_dir, "server-config.json")
    with open(server_config_path, "w") as f:
        f.write(json.dumps({"port": 8000}))
    server = Server(server_config_path)
    client = TestClient(server.api)
    resp = client.post(
        "/login", json={"username": "testuser", "password": "testpassword"}
    )
    assert resp.status_code == 200
    return server, client


def _upload_picture(client, filename) -> int:
    with open(os.path.join(PICTURES_DIR, filename), "rb") as f:
        result = upload_pictures_and_wait(
            client, [("file", (filename, f, "image/png"))]
        )
    assert result["status"] == "completed"
    return result["results"][0]["picture_id"]


@pytest.fixture(scope="module")
def library():
    """Six pictures tagged ``zebra``: three live, three in the Scrapheap.

    In each half the scores are 5, 4 and 1, so ``min_score=3`` keeps two of
    the three: a search that ignored the filter would show a third. Only the
    ``TEXT_MATCHES`` carry text that says "zebra".
    """
    temp_dir = tempfile.TemporaryDirectory()
    server, client = _server(temp_dir.name)
    try:
        ids = {name: _upload_picture(client, name) for name, _, _ in LIBRARY}
        for pid in ids.values():
            resp = client.post(f"/pictures/{pid}/tags", json={"tag": "zebra"})
            assert resp.status_code == 200, resp.text

        def seed(session):
            for name, score, deleted in LIBRARY:
                pic = session.get(Picture, ids[name])
                pic.score = score
                pic.deleted = deleted
                pic.ocr_text = ZEBRA_TEXT if name in TEXT_MATCHES else ""
                session.add(pic)
            session.commit()

        server.vault.db.run_task(seed)
        yield server, client
    finally:
        server.close()
        temp_dir.cleanup()
        gc.collect()


def _searched(client, params) -> list[int]:
    resp = client.get("/pictures/search", params={**SEARCH, **params})
    assert resp.status_code == 200, resp.text
    return [row["id"] for row in resp.json()]


def _finish(client, task_id, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"{EXPORT}/status", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"export {task_id} did not finish within {timeout_s}s")


def _exported_names(client, params) -> list[str]:
    """Export with the search's query string; the ZIP's pictures in order."""
    r = client.get(
        EXPORT,
        params={
            **SEARCH,
            **params,
            # Export options beside the view's filters: none may narrow, widen
            # or refuse the search.
            "caption_mode": "none",
            "use_original_file_names": "true",
            "resolution": "half",
            "tag_format": "underscores",
            "keep_generation_data": "false",
        },
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]
    status = _finish(client, task_id)
    assert status["status"] == "completed", status
    r = client.get(f"{EXPORT}/download/{task_id}")
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        return [name for name in archive.namelist() if name.endswith(".png")]


def _file_names(server, picture_ids) -> list[str]:
    def read(session):
        rows = session.exec(
            select(Picture.id, Picture.original_file_name).where(
                Picture.id.in_(picture_ids)
            )
        ).all()
        return {pid: name for pid, name in rows}

    by_id = server.vault.db.run_task(read)
    return [by_id[pid] for pid in picture_ids]


@pytest.mark.parametrize(
    "params, expected",
    [
        pytest.param({}, {"Bad1.png", "Bad2.png", "Changed1.png"}, id="no-filter"),
        pytest.param({"min_score": 3}, {"Bad1.png", "Bad2.png"}, id="min_score"),
        pytest.param(
            {"only_deleted": "true"},
            {"Changed2.png", "Reference1.png", "Changed3.png"},
            id="scrapheap",
        ),
        pytest.param(
            {"only_deleted": "true", "min_score": 3},
            {"Changed2.png", "Reference1.png"},
            id="scrapheap-min_score",
        ),
    ],
)
def test_a_search_export_packages_the_search_views_pictures_in_its_order(
    library, params, expected
):
    server, client = library
    shown = _searched(client, params)
    # The view itself is the one the filter and the Scrapheap flag make, so
    # the comparison below is not between two empty answers.
    assert set(_file_names(server, shown)) == expected

    assert _exported_names(client, params) == _file_names(server, shown)


IN_TEXT = {"text_match_only": "true"}


@pytest.mark.parametrize(
    "params, expected",
    [
        pytest.param({}, {"Bad1.png", "Changed1.png"}, id="no-filter"),
        pytest.param({"min_score": 3}, {"Bad1.png"}, id="min_score"),
        pytest.param({"only_deleted": "true"}, {"Changed2.png"}, id="scrapheap"),
    ],
)
def test_in_text_narrows_the_search_and_its_export_alike(library, params, expected):
    """With the switch on, the search answers only its text matches, and the
    export of that view packages exactly them; with it off, the same search
    answers more, so the narrowing is the switch's."""
    server, client = library
    shown = _searched(client, {**params, **IN_TEXT})
    assert set(_file_names(server, shown)) == expected
    assert set(_file_names(server, _searched(client, params))) > expected

    assert _exported_names(client, {**params, **IN_TEXT}) == _file_names(server, shown)


def test_in_text_narrows_before_the_page_is_cut(library):
    """Offset and limit page the narrowed list, not the whole search.

    By score the live search is Bad1 (5, text), Bad2 (4, no text), Changed1
    (1, text): a page cut before the narrowing would make page two Bad2 and
    then drop it, an empty page where the view shows Changed1.
    """
    server, client = library
    by_score = {**IN_TEXT, "sort": "SCORE", "descending": "true"}
    narrowed = _searched(client, by_score)
    assert _file_names(server, narrowed) == ["Bad1.png", "Changed1.png"]
    assert _file_names(server, _searched(client, {"sort": "SCORE"}))[1] == "Bad2.png"

    pages = [
        _searched(client, {**by_score, "offset": offset, "limit": 1})
        for offset in range(len(narrowed))
    ]
    assert pages == [[picture_id] for picture_id in narrowed]


def test_the_search_reads_the_views_params_and_none_of_the_exports():
    """What the export hands the search: the view's query string, options cut.

    Every export option is cut, every filter and the Scrapheap flag kept, in
    their order, repeats included; the request's state (the caller's token)
    is the real request's.
    """
    view = [
        ("min_score", "3"),
        ("tag", "zebra"),
        ("tag", "stripes"),
        ("only_deleted", "true"),
        ("sort", "DATE"),
    ]
    options = [(key, "x") for key in sorted(EXPORT_OPTION_PARAMS)]
    query_string = "&".join(f"{key}={value}" for key, value in options + view)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/pictures/export",
            "query_string": query_string.encode(),
            "headers": [],
            "state": {"marker": "the caller's"},
        }
    )

    seen = _ViewRequest(request)

    assert seen.query_params.multi_items() == view
    assert seen.state.marker == "the caller's"


def test_every_option_the_export_routes_declare_is_kept_from_the_search(library):
    """No export option the routes take can reach the search as a filter.

    Read off the mounted routes (``route_inventory``, the walk the guardrails
    use), so an option added to either export route without adding it to
    ``EXPORT_OPTION_PARAMS`` fails here. ``id`` and ``set_id`` are the view's
    own: the selection, and the set the view is in.
    """
    server, _client = library
    declared: dict = {}
    for method, path, name in iter_api_query_params(server.api):
        if (method, path) in (
            ("GET", "/api/v1/pictures/export"),
            ("POST", "/api/v1/pictures/export/folder"),
        ):
            declared.setdefault((method, path), set()).add(name)
    assert len(declared) == 2, declared
    for names in declared.values():
        options = names - {"id", "set_id"}
        assert {"caption_mode", "resolution", "keep_generation_data"} <= options
        assert options <= EXPORT_OPTION_PARAMS, options - EXPORT_OPTION_PARAMS
