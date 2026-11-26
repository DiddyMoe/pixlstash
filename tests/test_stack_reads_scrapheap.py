"""The two stack reads disagree about the Scrapheap on purpose.

``GET /stacks/{id}`` is the stack's membership: every member, Scrapheap
included. Dissolving a stack, removing a collapsed stack from a set and moving a
collapsed stack to the Scrapheap all expand the stack through it, and a member
left out there would stay grouped (or stay in the set) and come back so when it
is restored. ``GET /stacks/{id}/pictures`` is a picture listing for display: it
leaves out Scrapheap members as the grid does, unless ``include_deleted=true``.

So when every member is in the Scrapheap the membership still names them and
the listing is ``[]``. A resource-scoped token must get the same two answers
for the members it is granted. The listing used to decide the token's
visibility from its *live* in-scope members only, so it said 404 ("no such
stack") for a stack the membership read showed the same token.
"""

import contextlib
import io
import itertools
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixlstash.server import Server
from tests.authz_guard import no_spa_fallback  # noqa: F401
from tests.utils import upload_pictures_and_wait

API = "/api/v1"

pytestmark = pytest.mark.usefixtures("no_spa_fallback")

_colours = itertools.count(1)


def _png(name: str):
    n = next(_colours)
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(n * 40 % 256, n * 90 % 256, 7)).save(
        buf, format="PNG"
    )
    return ("file", (name, buf.getvalue(), "image/png"))


@contextlib.contextmanager
def _enforcing(server):
    prev = server.authz._enforcing
    server.authz._enforcing = True
    try:
        yield
    finally:
        server.authz._enforcing = prev


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def env():
    """A stack of A and B, both moved to the Scrapheap, plus a live stack of C
    and D; a picture token for A and one for a picture in no stack."""
    with tempfile.TemporaryDirectory() as tmp:
        with Server(os.path.join(tmp, "server-config.json")) as server:
            owner = TestClient(server.api, raise_server_exceptions=True)
            r = owner.post(
                f"{API}/login",
                json={"username": "owner", "password": "example-owner-password"},
            )
            assert r.status_code == 200, r.text
            names = ["sr_a.png", "sr_b.png", "sr_c.png", "sr_d.png", "sr_e.png"]
            status = upload_pictures_and_wait(owner, [_png(n) for n in names])
            assert status["status"] == "completed", status
            # One result per uploaded file, in upload order.
            assert [row["status"] for row in status["results"]] == ["success"] * 5
            a, b, c, d, e = (row["picture_id"] for row in status["results"])

            def stack(members):
                r = owner.post(f"{API}/stacks", json={"picture_ids": members})
                assert r.status_code == 200, r.text
                return r.json().get("id") or r.json()["stack_id"]

            scrapheaped_stack = stack([a, b])
            live_stack = stack([c, d])
            for pid in (a, b):
                r = owner.delete(f"{API}/pictures/{pid}")
                assert r.status_code == 200, r.text

            def mint(picture_id):
                r = owner.post(
                    f"{API}/users/me/token",
                    json={
                        "description": f"picture:{picture_id}",
                        "scope": "READ",
                        "resource_type": "picture",
                        "resource_id": picture_id,
                    },
                )
                assert r.status_code == 200, r.text
                return r.json()["token"]

            yield {
                "server": server,
                "owner": owner,
                "anon": TestClient(server.api, raise_server_exceptions=True),
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "scrapheaped_stack": scrapheaped_stack,
                "live_stack": live_stack,
                "token_a": mint(a),
                "token_c": mint(c),
                "token_outsider": mint(e),
            }


def _member_ids(client, stack_id, headers=None):
    r = client.get(f"{API}/stacks/{stack_id}", headers=headers)
    return r.status_code, (
        r.json().get("picture_ids") if r.status_code == 200 else r.text
    )


def _listed_ids(client, stack_id, headers=None, include_deleted=False):
    params = {"include_deleted": "true"} if include_deleted else None
    r = client.get(f"{API}/stacks/{stack_id}/pictures", headers=headers, params=params)
    if r.status_code != 200:
        return r.status_code, r.text
    return r.status_code, sorted(row["id"] for row in r.json())


def test_membership_names_scrapheaped_members_and_the_listing_leaves_them_out(env):
    owner, stack_id = env["owner"], env["scrapheaped_stack"]

    status, members = _member_ids(owner, stack_id)
    assert status == 200, members
    assert sorted(members) == sorted([env["a"], env["b"]])

    assert _listed_ids(owner, stack_id) == (200, [])
    assert _listed_ids(owner, stack_id, include_deleted=True) == (
        200,
        sorted([env["a"], env["b"]]),
    )


def test_a_scoped_token_gets_the_owners_answers_for_its_granted_members(env):
    anon, stack_id = env["anon"], env["scrapheaped_stack"]
    headers = _bearer(env["token_a"])

    with _enforcing(env["server"]):
        assert _member_ids(anon, stack_id, headers) == (200, [env["a"]])
        # The stack the membership read shows this token is not "not found"
        # here: it lists no live member, as it does for the owner.
        assert _listed_ids(anon, stack_id, headers) == (200, [])
        assert _listed_ids(anon, stack_id, headers, include_deleted=True) == (
            200,
            [env["a"]],
        )


def test_a_token_granted_no_member_sees_neither_stack(env):
    anon = env["anon"]
    headers = _bearer(env["token_outsider"])

    with _enforcing(env["server"]):
        for stack_id in (env["scrapheaped_stack"], env["live_stack"]):
            assert _member_ids(anon, stack_id, headers)[0] == 404
            assert _listed_ids(anon, stack_id, headers)[0] == 404
            assert _listed_ids(anon, stack_id, headers, include_deleted=True)[0] == 404


def test_a_live_stack_still_lists_only_the_granted_member(env):
    anon, stack_id = env["anon"], env["live_stack"]
    headers = _bearer(env["token_c"])

    with _enforcing(env["server"]):
        assert _member_ids(anon, stack_id, headers) == (200, [env["c"]])
        assert _listed_ids(anon, stack_id, headers) == (200, [env["c"]])
    assert _listed_ids(env["owner"], stack_id) == (200, sorted([env["c"], env["d"]]))
