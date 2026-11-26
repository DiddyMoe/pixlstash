"""Service layer for image plugin operations.

Extracted from pixlstash/routes/pictures.py to keep route handlers thin.
Provides the async orchestration for running image plugins with progress tracking.
"""

import asyncio
import uuid
from typing import TYPE_CHECKING

from pixlstash.event_types import EventType
from pixlstash.image_plugins.registry import get_image_plugin_manager
from pixlstash.image_plugins.service import (
    PictureNotFoundError,
    apply_plugin_to_pictures,
)
from pixlstash.pixl_logging import get_logger

if TYPE_CHECKING:
    from pixlstash.image_plugins.base import ImagePlugin
    from pixlstash.vault import Vault

logger = get_logger(__name__)

#: What a client is told when a plugin run fails. Whatever raised (the plugin's
#: own code, PIL, a picture whose file went missing) is logged with the
#: plugin's name; its text reaches neither the event nor the response.
PLUGIN_FAILED_MESSAGE = "The plugin failed; the log says why."
#: The sentence an error report carries when its plugin gave none.
PLUGIN_ERROR_MESSAGE = "Plugin error"


class PluginNotFoundError(LookupError):
    """No installed image plugin has the requested name.

    Raised before a run starts, so nothing was emitted; a run that starts and
    then fails raises ``RuntimeError`` instead, whatever the plugin raised.
    """


def list_plugins(vault: "Vault") -> dict:
    """Return the available image plugins.

    Neither the scanned folders nor the load errors are returned, and both
    omissions are the same rule: ``GET /pictures/plugins`` was ANY_TOKEN when
    they were dropped (it is OWNER_ONLY now, for the third-party plugin text it
    still serves), the folders are host paths under the owner's home directory,
    and an error row carries the absolute ``file`` it failed on plus exception
    text from third-party code. Nothing in the UI ever read either - the failures are in
    the server log, where they were being read from anyway. The tagger
    equivalents are served instead by ``GET /taggers/plugin-diagnostics``,
    which is LOCAL_OWNER_ONLY, because the Auto-tagging screen does show them.

    Args:
        vault: Application vault (unused currently; reserved for future
            vault-scoped plugin discovery).

    Returns:
        Dict with the single key ``plugins``.
    """
    manager = get_image_plugin_manager()
    manager.reload()
    return {"plugins": manager.list_plugins()}


async def run_plugin_on_pictures(
    server,
    name: str,
    picture_ids: list[int],
    parameters: dict,
    captions: list[str] | None = None,
    origin_client_id: str | None = None,
    stack: bool = True,
) -> dict:
    """Run a named image plugin on a list of pictures with progress tracking.

    Emits PLUGIN_PROGRESS WebSocket events throughout the run.  Returns the
    result dict from the plugin, enriched with ``status: "success"``.

    No exception text leaves through either. A failed run's event says
    ``PLUGIN_FAILED_MESSAGE``, and each per-image error report, in its event
    and in the returned ``errors``, is cut to what ``_client_error`` keeps.
    The exception and the full report, ``details`` included, are logged.

    Args:
        server: Application server (provides vault for event notification).
        name: Plugin name to run.
        picture_ids: List of picture IDs to process.
        parameters: Plugin-specific parameter dict.
        captions: Optional per-picture captions, must match ``picture_ids`` length.
        origin_client_id: Opaque ``X-Client-Id`` of the originating tab, captured
            at request entry. Plugin output is a UI-initiated import, so it is
            echoed on the PICTURE_IMPORTED event so the originating tab can do a
            targeted grid insert instead of a full reload. Echo-matching only.
        stack: When ``True`` (default), plugin outputs are placed in the source
            picture's stack. When ``False``, the physical stacking is skipped but
            all source associations (set/project/face) are still copied.

    Raises:
        PluginNotFoundError: No plugin has that name; nothing ran.
        PictureNotFoundError: A requested picture has no row.
        RuntimeError: The run failed, whatever raised (the plugin, PIL, a
            picture whose file is gone). Its text is for the log only.
    """
    manager = get_image_plugin_manager()
    # A reload imports every plugin file from disk, third-party code included:
    # on a worker, so the event loop every other request shares never waits on
    # it.
    await asyncio.to_thread(manager.reload)
    plugin: "ImagePlugin | None" = manager.get_plugin(name)
    if plugin is None:
        raise PluginNotFoundError(f"Plugin not found: {name!r}")

    vault: "Vault" = server.vault
    plugin_run_id = str(uuid.uuid4())

    def _emit_progress(progress_payload: dict) -> None:
        if not isinstance(progress_payload, dict):
            return
        vault.notify(
            EventType.PLUGIN_PROGRESS,
            {
                "run_id": plugin_run_id,
                "plugin": str(progress_payload.get("plugin") or name),
                "status": "running",
                **progress_payload,
            },
        )

    def _emit_error(error_payload: dict) -> None:
        # The report's details hold exception text and source paths: the log
        # gets all of it, the event only what _client_error keeps.
        logger.warning("Plugin %r reported an error: %r", name, error_payload)
        if not isinstance(error_payload, dict):
            return
        error = _client_error(name, error_payload)
        vault.notify(
            EventType.PLUGIN_PROGRESS,
            {
                "run_id": plugin_run_id,
                "plugin": error["plugin"],
                "status": "error",
                "message": error["message"],
                "error": error,
            },
        )

    vault.notify(
        EventType.PLUGIN_PROGRESS,
        {
            "run_id": plugin_run_id,
            "plugin": name,
            "status": "started",
            "current": 0,
            "total": len(picture_ids),
            "progress": 0.0,
            "message": f"Starting plugin: {name}",
        },
    )

    try:
        result = await asyncio.to_thread(
            apply_plugin_to_pictures,
            server,
            plugin,
            picture_ids,
            parameters,
            captions,
            progress_reporter=_emit_progress,
            error_reporter=_emit_error,
            stack=stack,
        )
    except Exception as exc:
        logger.warning(
            "Plugin %r failed on %d picture(s): %r", name, len(picture_ids), exc
        )
        vault.notify(
            EventType.PLUGIN_PROGRESS,
            {
                "run_id": plugin_run_id,
                "plugin": name,
                "status": "failed",
                "message": PLUGIN_FAILED_MESSAGE,
            },
        )
        if isinstance(exc, PictureNotFoundError):
            raise
        # A ValueError here is a failed run too (a file gone from disk, a
        # plugin returning too few images), never a lookup miss.
        raise RuntimeError(str(exc)) from exc

    vault.notify(
        EventType.PLUGIN_PROGRESS,
        {
            "run_id": plugin_run_id,
            "plugin": name,
            "status": "completed",
            "current": len(picture_ids),
            "total": len(picture_ids),
            "progress": 100.0,
            "message": f"Completed plugin: {name}",
        },
    )

    created_ids = result.get("created_picture_ids") or []
    output_ids = result.get("output_picture_ids") or []
    if created_ids:
        vault.notify(
            EventType.PICTURE_IMPORTED,
            {
                "ids": created_ids,
                "source": "ui",
                "origin_client_id": origin_client_id,
                "change_kind": "added",
            },
        )
    if output_ids:
        vault.notify(
            EventType.CHANGED_PICTURES,
            {
                "picture_ids": output_ids,
                "source": "ui",
                "origin_client_id": origin_client_id,
                "change_kind": "updated",
            },
        )

    return {
        "status": "success",
        **result,
        "errors": [_client_error(name, error) for error in result.get("errors") or []],
    }


def _client_error(name: str, payload: object) -> dict:
    """Cut one plugin error report down to what a client may be shown.

    ``ImagePlugin.report_error`` puts exception text and source paths in a
    report's ``details``, and a plugin may hand its error callback anything at
    all, so only the plugin name, the input's index and the plugin's own
    sentence are kept. ``run_plugin_on_pictures`` logs the whole report.

    Args:
        name: The plugin's registered name.
        payload: The report as the plugin passed it to its error callback.

    Returns:
        ``{"plugin", "index", "message"}``; a report that is not a dict keeps
        only the plugin's name and ``PLUGIN_ERROR_MESSAGE``.
    """
    if not isinstance(payload, dict):
        return {"plugin": name, "index": None, "message": PLUGIN_ERROR_MESSAGE}
    index = payload.get("index")
    return {
        "plugin": str(payload.get("plugin") or name),
        "index": index if isinstance(index, int) else None,
        "message": str(payload.get("message") or PLUGIN_ERROR_MESSAGE),
    }
