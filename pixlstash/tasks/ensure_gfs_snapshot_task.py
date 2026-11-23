"""One-shot task that takes a GFS snapshot of the given tier."""

from typing import TYPE_CHECKING

from pixlstash.pixl_logging import get_logger
from pixlstash.services.snapshot_service import (
    SnapshotCancellation,
    SnapshotCancelledError,
)
from pixlstash.tasks.base_task import BaseTask, TaskPriority

if TYPE_CHECKING:
    from pixlstash.vault import Vault

logger = get_logger(__name__)


class EnsureGfsSnapshotTask(BaseTask):
    """Create a DAILY / WEEKLY / MONTHLY snapshot for the current period.

    Created by ``EnsureGfsSnapshotFinder``, which decides the tier. Runs at
    LOW priority so it does not starve other background work. The snapshot
    checks the task's cancel event, so ``on_cancel`` stops it mid-copy or
    mid-compression and nothing of it is kept.

    Attributes:
        _vault: The owning Vault, used to access SnapshotService.
        _kind: The snapshot tier to create (``DAILY`` / ``WEEKLY`` /
            ``MONTHLY``).
    """

    def __init__(self, vault: "Vault", kind: str = "DAILY") -> None:
        """Initialise the task.

        Args:
            vault: The owning Vault, used to access SnapshotService.
            kind: The snapshot tier to create.
        """
        super().__init__(task_type="EnsureGfsSnapshotTask")
        self._vault = vault
        self._kind = kind

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.LOW

    def _run_task(self):
        try:
            cp = self._vault.snapshot_service.create_snapshot(
                self._kind, cancellation=SnapshotCancellation(self._cancel_event)
            )
            logger.info(
                "EnsureGfsSnapshotTask: %s snapshot id=%d created", self._kind, cp.id
            )
            return {"snapshot_id": cp.id, "kind": self._kind}
        except SnapshotCancelledError:
            logger.info(
                "EnsureGfsSnapshotTask: %s snapshot cancelled; nothing was kept",
                self._kind,
            )
            raise
        except Exception as exc:
            logger.error(
                "EnsureGfsSnapshotTask: failed to create %s snapshot: %s",
                self._kind,
                exc,
                exc_info=True,
            )
            raise
