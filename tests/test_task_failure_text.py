"""A failed task's served ``error`` is a fixed sentence; the exception is logged.

``BaseTask.error`` reaches clients: an import's status and a model folder's
``scan_error`` read it. An exception's text can carry file paths and library
content, so it goes to the server log, with its traceback and the task, and
the field says only that the task failed. The one exception is a
``PublicTaskFailure``, whose message the server composed from values it
counted: that message is served, and its cause is still only logged.
"""

import logging

import pytest

from pixlstash.tasks.base_task import (
    TASK_FAILURE_MESSAGE,
    BaseTask,
    PublicTaskFailure,
    TaskStatus,
)
from pixlstash.tasks.model_folder_scan_task import claim_scan, forget_scan, scan_state

_SECRET = "example-secret at /home/me/library/private/a.png"


class _Failing(BaseTask):
    def __init__(self, exc: BaseException):
        super().__init__("FailingTask")
        self._exc = exc

    def _run_task(self):
        raise self._exc


@pytest.mark.parametrize("exc", [RuntimeError(_SECRET), SystemExit(_SECRET)])
def test_a_failed_task_serves_a_fixed_sentence_and_logs_the_exception(exc, caplog):
    task = _Failing(exc)

    with caplog.at_level(logging.WARNING, logger="pixlstash.tasks.base_task"):
        with pytest.raises(type(exc)):
            task.run()

    assert task.status is TaskStatus.FAILED
    assert task.error == TASK_FAILURE_MESSAGE
    assert _SECRET not in task.error
    logged = [r for r in caplog.records if task.id in r.getMessage()]
    assert logged, "the failure is not logged against its task"
    assert logged[0].exc_info and logged[0].exc_info[1] is exc
    assert "FailingTask" in logged[0].getMessage()


def _raise_public(cause: BaseException, message: str):
    try:
        raise cause
    except type(cause) as exc:
        raise PublicTaskFailure(message) from exc


def test_a_public_failure_serves_its_own_message_and_logs_the_cause(caplog):
    message = "The import stopped after 3 new picture(s) were added."
    task = _Failing(RuntimeError(_SECRET))
    task._run_task = lambda: _raise_public(RuntimeError(_SECRET), message)

    with caplog.at_level(logging.WARNING, logger="pixlstash.tasks.base_task"):
        with pytest.raises(PublicTaskFailure):
            task.run()

    assert task.status is TaskStatus.FAILED
    assert task.error == message
    logged = [r for r in caplog.records if task.id in r.getMessage()]
    assert logged, "the failure is not logged against its task"
    raised = logged[0].exc_info[1]
    assert isinstance(raised, PublicTaskFailure)
    assert str(raised.__cause__) == _SECRET, "the cause is not in the log record"


def test_only_a_public_failure_is_served_not_a_lookalike():
    """An exception that merely carries a ``public_message`` is not trusted."""

    class _Lookalike(RuntimeError):
        public_message = _SECRET

    task = _Failing(_Lookalike(_SECRET))
    with pytest.raises(_Lookalike):
        task.run()
    assert task.error == TASK_FAILURE_MESSAGE


def test_a_model_folder_scan_that_crashed_serves_no_exception_text(hub, tmp_path):
    task = claim_scan(hub, 4242, str(tmp_path), "lora")
    assert task is not None

    def die():
        raise RuntimeError(_SECRET)

    task._run_task = die
    try:
        with pytest.raises(RuntimeError):
            task.run()
        assert scan_state(4242) == ("failed", TASK_FAILURE_MESSAGE)
    finally:
        forget_scan(4242)
