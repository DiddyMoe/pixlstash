"""Finder that picks up requested auto-stack runs.

An auto-stack run, or the revert of one, is *requested* by writing a
:class:`~pixlstash.db_models.dedup.DedupAutoStackRun` row with status
``pending`` (``POST /dedup/auto-stack/runs`` does that and answers 202 at once).
This finder turns the oldest pending row into a
:class:`~pixlstash.tasks.dedup_auto_stack_task.DedupAutoStackTask`.

Like :class:`~pixlstash.tasks.dedup_scan_finder.DedupScanFinder` it claims no
pictures: the unit of work is a run row.
"""

from typing import TYPE_CHECKING, Optional

from pixlstash.pixl_logging import get_logger
from pixlstash.services.dedup_auto_stack_service import find_pending_run_id
from pixlstash.tasks.base_task_finder import BaseTaskFinder
from pixlstash.tasks.dedup_auto_stack_task import DedupAutoStackTask

if TYPE_CHECKING:
    from pixlstash.vault import Vault

logger = get_logger(__name__)


class DedupAutoStackFinder(BaseTaskFinder):
    """Create one :class:`DedupAutoStackTask` per pending run request.

    Attributes:
        _vault: The owning Vault, which the task needs for its writer slices,
            its change events and the undo registry.
    """

    def __init__(self, vault: "Vault") -> None:
        super().__init__()
        self._vault = vault

    def finder_name(self) -> str:
        return "DedupAutoStackFinder"

    def max_inflight_tasks(self) -> int:
        # One run at a time, which the service also enforces on the rows: two
        # runs would interleave their History segments and neither could be
        # reverted as a unit.
        return 1

    def find_task(self) -> Optional[DedupAutoStackTask]:
        run_id = self._vault.db.run_immediate_read_task(find_pending_run_id)
        if run_id is None:
            return None
        logger.info("[auto-stack] queueing run %s", run_id)
        return DedupAutoStackTask(vault=self._vault, run_id=run_id)
