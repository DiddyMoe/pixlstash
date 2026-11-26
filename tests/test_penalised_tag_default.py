"""A request with no user gets its own copy of the penalised-tag seed.

``DEFAULT_SMART_SCORE_PENALIZED_TAGS`` is one module-level dict, the seed of
every new owner's table and the fallback of every scorer that has none. A
request handed that dict itself, and editing it, would change the weights the
whole process scores with until it restarts.
"""

from types import SimpleNamespace

from pixlstash.db_models.tag import DEFAULT_SMART_SCORE_PENALIZED_TAGS
from pixlstash.scoring.smart_score import get_smart_score_penalised_tags_from_request


def test_a_request_with_no_user_cannot_edit_the_seed():
    seed = dict(DEFAULT_SMART_SCORE_PENALIZED_TAGS)
    server = SimpleNamespace(auth=SimpleNamespace(get_user_id=lambda request: None))

    tags = get_smart_score_penalised_tags_from_request(server, object())
    assert tags == seed
    tags["example-tag"] = 5
    tags.pop("bad anatomy")

    assert DEFAULT_SMART_SCORE_PENALIZED_TAGS == seed
