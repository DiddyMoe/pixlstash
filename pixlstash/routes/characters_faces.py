"""Character face-assignment endpoints.

The two handlers that assign / unassign faces to a character
(``POST`` and ``DELETE`` on ``/characters/{character_id}/faces``) live here,
split out of :mod:`pixlstash.routes.characters` to keep that module focused on
character CRUD and search. Behaviour, paths, and methods are unchanged; the
router is mounted adjacently to the characters router in ``server.py``.

Scope enforcement for these mutations is handled by
:func:`_enforce_face_mutation_scope`, which resolves both the ``face_ids`` and
``picture_ids`` branches to the affected picture set and denies the whole request
if any targeted picture falls outside a scoped token's grant (BOLA guard).
"""

from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select

from pixlstash.utils.sqlite_ints import SqliteInt, parse_sqlite_int
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
)
from pixlstash.event_types import EventType
from pixlstash.scoring import (
    compute_character_likeness_for_faces,
    select_reference_faces_for_character,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.services import operation_log_service
from pixlstash.services.project_membership_service import (
    character_project_ids,
    reconcile_entity_projects_change,
)
from pixlstash.services.stack_membership import expand_picture_ids_to_stacks
from pixlstash.utils.service.filter_helpers import fetch_scope_allowed_picture_ids
from pixlstash.utils.sql_chunking import chunked

logger = get_logger(__name__)


class CharacterFaceAssignmentResponse(BaseModel):
    """Result of assigning or unassigning faces for a character."""

    model_config = ConfigDict(extra="allow")

    status: str
    face_ids: Optional[list[int]] = None
    character_id: int
    already_assigned_ids: Optional[list[int]] = None


def _enforce_face_mutation_scope(
    server,
    request,
    *,
    face_ids: list[int] | None,
    picture_ids: list[int] | None,
    expand_stacks: bool = False,
) -> None:
    """Raise 403 if a scoped token targets faces/pictures outside its scope.

    *face_ids* and *picture_ids* are already parsed by :func:`_id_list`, so
    this check and the write that follows read the very same ids.

    The character face-assign / face-unassign handlers accept *either* a list
    of ``face_ids`` *or* a list of ``picture_ids``. This resolves both paths to
    the full set of affected picture ids and checks every one against the
    token's scope. Owner / unscoped tokens (``fetch_scope_allowed_picture_ids``
    returns ``None``) pass straight through. This is all-or-nothing: if *any*
    targeted picture is out of scope the whole request is denied, so neither the
    ``face_ids`` branch nor the ``picture_ids`` branch can mutate an
    out-of-scope picture.
    """
    scope_allowed = fetch_scope_allowed_picture_ids(server, request)
    if scope_allowed is None:
        return

    affected: set[int] = set(picture_ids or [])
    normalized_face_ids: list[int] = list(face_ids or [])
    if normalized_face_ids:

        def _resolve(session: Session, ids: list[int]) -> set[int]:
            # The body's id list has no cap: chunked under SQLite's
            # bound-parameter limit, and the sets merge.
            return {
                int(r)
                for id_chunk in chunked(ids)
                for r in session.exec(
                    select(Face.picture_id).where(Face.id.in_(id_chunk))
                ).all()
                if r is not None
            }

        affected |= server.vault.db.run_immediate_read_task(
            _resolve, normalized_face_ids
        )

    if expand_stacks and affected:

        def _expand(session: Session, ids: set[int]) -> set[int]:
            return set(expand_picture_ids_to_stacks(session, ids))

        affected = server.vault.db.run_immediate_read_task(_expand, affected)

    if any(pid not in scope_allowed for pid in affected):
        raise HTTPException(
            status_code=403,
            detail="Token is not authorised to access these pictures",
        )


def _id_list(values, name: str) -> list[int] | None:
    """The JSON list *values* from a request body as ids, or None when absent.

    Every entry must be an integer, or a string ``int()`` reads as one, inside
    SQLite's range; anything else is refused with 400 naming the list. Nothing
    is skipped: the scope check and the write must see the same ids, and SQLite
    compares a text value such as ``"5e0"`` or ``"5.0"`` to an INTEGER column
    as the number 5, which ``int()`` refuses.
    """
    if values is None:
        return None
    if not isinstance(values, list):
        raise HTTPException(status_code=400, detail=f"{name} must be a list")
    ids: list[int] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise HTTPException(status_code=400, detail=f"{name} must be a list of ids")
        try:
            ids.append(parse_sqlite_int(raw))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"{name} must be a list of ids"
            ) from exc
    return ids


def _picture_ids_for_faces(face_ids):
    """Resolver for the operation log: the pictures behind a face-id request.

    Character assignment can be addressed either by picture id or by face id.
    The face-id form does not name its pictures, so the operation log would
    snapshot nothing and record a half-change. This returns a
    ``(session) -> picture ids`` callable the recorder runs on the mutation's own
    session just before the write, or ``None`` when the request already named its
    pictures.
    """
    if not face_ids:
        return None

    def _resolve(session: Session):
        # Chunked like the scope check: the request's face ids have no cap.
        return [
            picture_id
            for id_chunk in chunked(list(face_ids))
            for picture_id in session.exec(
                select(Face.picture_id).where(Face.id.in_(id_chunk))
            ).all()
            if picture_id is not None
        ]

    return _resolve


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/characters/{character_id}/faces",
        summary="Assign faces to character",
        description="Assigns provided face ids or largest faces from picture ids to a character.",
        response_model=CharacterFaceAssignmentResponse,
    )
    def assign_face_to_character(
        request: Request, character_id: SqliteInt, payload: dict = Body(...)
    ):
        origin_client_id = getattr(request.state, "origin_client_id", None)
        face_ids = _id_list(payload.get("face_ids"), "face_ids")
        picture_ids = _id_list(payload.get("picture_ids"), "picture_ids")
        face_assignments_raw = payload.get("face_assignments")
        if face_assignments_raw is not None and not isinstance(
            face_assignments_raw, list
        ):
            raise HTTPException(
                status_code=400, detail="face_assignments must be a list"
            )
        if face_assignments_raw and (face_ids or picture_ids):
            raise HTTPException(
                status_code=400,
                detail=(
                    "face_assignments is authoritative and cannot be combined "
                    "with face_ids or picture_ids"
                ),
            )

        face_assignments: list[tuple[int, int]] = []
        seen_pictures: set[int] = set()
        seen_faces: set[int] = set()
        for index, raw in enumerate(face_assignments_raw or []):
            if not isinstance(raw, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"face_assignments[{index}] must be an object",
                )
            try:
                picture_id = parse_sqlite_int(raw["picture_id"])
                face_id = parse_sqlite_int(raw["face_id"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"face_assignments[{index}] requires integer picture_id "
                        "and face_id"
                    ),
                )
            if picture_id <= 0 or face_id <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"face_assignments[{index}] picture_id and face_id must "
                        "be positive"
                    ),
                )
            if picture_id in seen_pictures or face_id in seen_faces:
                raise HTTPException(
                    status_code=400,
                    detail="face_assignments must contain unique pictures and faces",
                )
            seen_pictures.add(picture_id)
            seen_faces.add(face_id)
            face_assignments.append((picture_id, face_id))

        assignment_picture_ids = [picture_id for picture_id, _ in face_assignments]
        assignment_face_ids = [face_id for _, face_id in face_assignments]
        # Scope guard (BOLA): a write-capable resource-scoped token may only
        # assign faces on pictures within its granted resource. Covers both the
        # face_ids and picture_ids branches.
        _enforce_face_mutation_scope(
            server,
            request,
            face_ids=[*(face_ids or []), *assignment_face_ids],
            picture_ids=[*(picture_ids or []), *assignment_picture_ids],
            expand_stacks=bool(picture_ids or face_assignments),
        )

        def assign_faces(
            session: Session,
            face_assignments: list[tuple[int, int]],
            face_ids: list[int],
            picture_ids: list[int],
            character_id: int,
        ):
            faces_to_assign = []
            existing_faces = []
            for expected_picture_id, face_id in face_assignments:
                face = session.get(Face, face_id)
                if face is None:
                    raise HTTPException(
                        status_code=404, detail=f"Face {face_id} not found"
                    )
                if int(face.picture_id) != expected_picture_id:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Face {face_id} belongs to picture {face.picture_id}, "
                            f"not submitted picture {expected_picture_id}"
                        ),
                    )
                if face.character_id == character_id:
                    existing_faces.append(face)
                else:
                    # Suggest More already compared every face and named this
                    # winner. Assign it verbatim; a second likeness pass here
                    # could choose a different person under another reducer.
                    faces_to_assign.append(face)
            selection_picture_ids = []
            if picture_ids:
                selection_picture_ids = expand_picture_ids_to_stacks(
                    session, picture_ids
                )
            elif face_assignments:
                # The reviewed picture/face pairs are authoritative, but the
                # rest of each live stack still moves as one character unit.
                # Exclude named pictures from this selection pass so their
                # submitted winners cannot be silently re-ranked.
                authoritative_picture_ids = {
                    picture_id for picture_id, _ in face_assignments
                }
                selection_picture_ids = [
                    picture_id
                    for picture_id in expand_picture_ids_to_stacks(
                        session, authoritative_picture_ids
                    )
                    if picture_id not in authoritative_picture_ids
                ]
            if selection_picture_ids:
                # Stacks move as a unit: assigning any stacked picture to a
                # character assigns every member of its stack, so a collapsed
                # stack dragged onto a character moves all of its pictures
                # (reassigning each member's face also moves it off the old
                # character, keeping character counts consistent).
                reference_faces = select_reference_faces_for_character(
                    session, character_id
                )

                def face_area(face):
                    try:
                        return (face.width or 0) * (face.height or 0)
                    except Exception:
                        # Sort-key guard: a face missing usable dimensions sorts
                        # as zero area; 0 IS the answer, not an error.
                        return 0

                def pick_best_face(faces, comparison_faces):
                    """Select the face to assign from one picture's real faces.

                    With a comparison set, rank by likeness to it (area breaks
                    ties); without one, or when no face has features, fall back
                    to the largest face. This is the long-standing selection
                    rule, shared by the reference-faces path and the bootstrap
                    path below.
                    """
                    if comparison_faces:
                        faces_with_features = [f for f in faces if f.features]
                        if faces_with_features:
                            likeness_map = compute_character_likeness_for_faces(
                                comparison_faces, faces_with_features
                            )
                            return max(
                                faces_with_features,
                                key=lambda f: (
                                    likeness_map.get(f.id, 0.0),
                                    face_area(f),
                                ),
                            )
                    return max(faces, key=face_area)

                def record_face(best_face):
                    if best_face.character_id == character_id:
                        existing_faces.append(best_face)
                    else:
                        faces_to_assign.append(best_face)

                # Bootstrap heuristic for a character with no reference faces
                # yet (a freshly created person, issue #645): assign the
                # unambiguous single-face pictures first, then use those faces
                # as the comparison set for the multi-face pictures, so group
                # shots pick the same identity instead of whoever happens to
                # have the largest face. With reference faces present both
                # lists stay empty and the loop behaves exactly as before.
                bootstrap_refs = []
                deferred_multi_face = []

                for pic_id in selection_picture_ids:
                    faces = Face.find(session, picture_id=pic_id)
                    if not faces:
                        # Face.find excludes sentinel records (face_index == -1),
                        # so an empty result means either extraction hasn't run yet
                        # or ran and found nothing.  Check for any record at all.
                        any_face_id = session.exec(
                            select(Face.id).where(Face.picture_id == pic_id).limit(1)
                        ).first()
                        if any_face_id is None:
                            # Extraction not yet run; defer assignment until it does.
                            pic = session.get(Picture, pic_id)
                            if pic is not None:
                                pic.pending_character_id = character_id
                                session.add(pic)
                        continue

                    if not reference_faces and len(faces) > 1:
                        deferred_multi_face.append((pic_id, faces))
                        continue

                    best_face = pick_best_face(faces, reference_faces)
                    record_face(best_face)
                    if not reference_faces and best_face.features:
                        bootstrap_refs.append(best_face)

                if deferred_multi_face:
                    deferred_pic_ids = [pic_id for pic_id, _ in deferred_multi_face]
                    if bootstrap_refs:
                        logger.info(
                            "Bootstrapping face selection for character %s: "
                            "%d single-face picture(s) form the comparison set "
                            "for %d multi-face picture(s) %s",
                            character_id,
                            len(bootstrap_refs),
                            len(deferred_multi_face),
                            deferred_pic_ids,
                        )
                    else:
                        logger.info(
                            "No reference faces and no single-face pictures to "
                            "bootstrap from for character %s; using largest-face "
                            "fallback for %d multi-face picture(s) %s",
                            character_id,
                            len(deferred_multi_face),
                            deferred_pic_ids,
                        )
                    for pic_id, faces in deferred_multi_face:
                        record_face(pick_best_face(faces, bootstrap_refs))
            if face_ids:
                for face_id in face_ids:
                    face = session.get(Face, face_id)
                    if not face:
                        raise HTTPException(
                            status_code=404, detail=f"Face {face_id} not found"
                        )
                    if face.character_id == character_id:
                        existing_faces.append(face)
                    else:
                        faces_to_assign.append(face)
            unique_faces = {face.id: face for face in faces_to_assign}.values()
            for face in unique_faces:
                face.character_id = character_id
                session.add(face)
            session.flush()
            for face in unique_faces:
                session.refresh(face)
            character = session.get(Character, character_id)
            if character is not None:
                # Issue #125: a character may belong to several projects, so the
                # newly assigned pictures join *all* of them. Reading the primary
                # FK here would leave them out of every secondary project.
                assigned_picture_ids = [
                    int(face.picture_id) for face in unique_faces if face.picture_id
                ]
                project_ids = character_project_ids(session, character_id)
                if assigned_picture_ids and project_ids:
                    reconcile_entity_projects_change(
                        session,
                        picture_ids=assigned_picture_ids,
                        ensure_project_ids=project_ids,
                        remove_project_ids=[],
                    )
                    session.flush()
            faces_payload = [
                {
                    "id": face.id,
                    "picture_id": face.picture_id,
                    "character_id": face.character_id,
                }
                for face in unique_faces
            ]
            existing_face_ids = [face.id for face in existing_faces]
            return faces_payload, existing_face_ids

        # Assignment is stack-atomic (the work function expands the request to
        # whole stacks), so the snapshot expands too. A request addressed by face
        # id does not name its pictures, so resolve them on the mutation's own
        # session before the write - otherwise the operation would record a
        # half-change that undo could not reverse.
        (faces, existing_face_ids), _operation = (
            operation_log_service.run_recorded_metadata_task(
                server.vault,
                assign_faces,
                face_assignments,
                face_ids,
                picture_ids,
                character_id,
                op_type="characters.assign",
                picture_ids=[*(picture_ids or []), *assignment_picture_ids],
                resolve_picture_ids=_picture_ids_for_faces(
                    [*(face_ids or []), *assignment_face_ids]
                ),
                expand_stacks=True,
                summary="Assigned pictures to a character",
                **operation_log_service.request_context(request),
            )
        )
        if not faces and len(existing_face_ids) > 0:
            # All requested faces are already assigned to this character - the
            # desired state is already achieved.  Return success so callers
            # (e.g. the ComfyUI node re-importing a duplicate picture) do not
            # treat this as an error.
            return {
                "status": "success",
                "face_ids": [],
                "character_id": character_id,
                "already_assigned_ids": existing_face_ids,
            }
        server.vault.db.run_task(
            Picture.clear_field,
            [face["picture_id"] for face in faces],
            "text_embedding",
        )
        for face in faces:
            if face["character_id"] != character_id:
                logger.error(
                    "Assigning faces to character %s left face %s (picture %s) on "
                    "character %s.",
                    character_id,
                    face["id"],
                    face["picture_id"],
                    face["character_id"],
                )
                raise HTTPException(
                    status_code=500,
                    detail="Not every face could be assigned to that character.",
                )
        server.vault.notify(
            EventType.CHANGED_CHARACTERS, {"origin_client_id": origin_client_id}
        )
        # CHANGED_FACES serializes to the ``characters_changed`` wire type, whose
        # payload carries no picture_ids/change_kind (the frontend reacts with a
        # sidebar refresh). Only origin_client_id is read into the envelope.
        server.vault.notify(
            EventType.CHANGED_FACES,
            {"origin_client_id": origin_client_id},
        )
        return {
            "status": "success",
            "face_ids": [face["id"] for face in faces],
            "character_id": character_id,
        }

    @router.delete(
        "/characters/{character_id}/faces",
        summary="Unassign faces from character",
        description="Removes character assignment from provided face ids or from faces in provided picture ids.",
        response_model=CharacterFaceAssignmentResponse,
    )
    def remove_character_from_faces(
        request: Request, character_id: SqliteInt, payload: dict = Body(...)
    ):
        origin_client_id = getattr(request.state, "origin_client_id", None)
        face_ids = payload.get("face_ids", None)
        picture_ids = payload.get("picture_ids", None)
        if not isinstance(face_ids, list) and not isinstance(picture_ids, list):
            raise HTTPException(
                status_code=400,
                detail="Must send a list of picture_ids or face_ids",
            )
        face_ids = _id_list(face_ids, "face_ids")
        picture_ids = _id_list(picture_ids, "picture_ids")
        # Scope guard (BOLA): a write-capable resource-scoped token may only
        # unassign faces on pictures within its granted resource. Covers both
        # the face_ids and picture_ids branches.
        _enforce_face_mutation_scope(
            server, request, face_ids=face_ids, picture_ids=picture_ids
        )

        def remove_faces_from_character(
            session: Session,
            character_id: int,
            face_ids: list[int] = None,
            picture_ids: list[int] = None,
        ) -> list[int]:
            """Unassign this character's faces named by the request.

            A face on another character, or a face id that does not exist, is
            left alone, so a request that matches nothing changes nothing.

            Returns:
                list[int]: The ids of the pictures whose faces were actually
                unassigned, sorted and without repeats.
            """
            faces = []
            if picture_ids:
                for pic_id in picture_ids:
                    pic_faces = Face.find(session, picture_id=pic_id)
                    for face in pic_faces:
                        if face.character_id == character_id:
                            face.character_id = None
                            session.add(face)
                            faces.append(face)
            elif face_ids:
                for face_id in face_ids:
                    face = session.get(Face, face_id)
                    if face and face.character_id == character_id:
                        face.character_id = None
                        session.add(face)
                        faces.append(face)
            session.flush()
            # Face.picture_id is a nullable column; the resolver above skips a
            # face without a picture the same way.
            return sorted(
                {face.picture_id for face in faces if face.picture_id is not None}
            )

        # Unlike the assign above, this handler does not expand to stacks, so the
        # snapshot covers exactly the requested pictures - plus the pictures of a
        # face-id-addressed request, resolved on the mutation's own session.
        unassigned_picture_ids, _operation = (
            operation_log_service.run_recorded_metadata_task(
                server.vault,
                remove_faces_from_character,
                character_id,
                face_ids,
                picture_ids,
                op_type="characters.unassign",
                picture_ids=picture_ids,
                resolve_picture_ids=_picture_ids_for_faces(face_ids),
                summary="Unassigned pictures from a character",
                **operation_log_service.request_context(request),
            )
        )

        # Only a picture that lost a face of this character has a stale text
        # embedding. clear_field([]) matches nothing anyway (id IN ()), so this
        # check only spares the writer an empty task.
        if unassigned_picture_ids:
            server.vault.db.run_task(
                Picture.clear_field, unassigned_picture_ids, "text_embedding"
            )
        server.vault.notify(
            EventType.CHANGED_CHARACTERS, {"origin_client_id": origin_client_id}
        )
        # CHANGED_FACES serializes to the ``characters_changed`` wire type, whose
        # payload carries no picture_ids/change_kind (the frontend reacts with a
        # sidebar refresh). Only origin_client_id is read into the envelope.
        server.vault.notify(
            EventType.CHANGED_FACES,
            {"origin_client_id": origin_client_id},
        )
        return {
            "status": "success",
            "face_ids": face_ids,
            "character_id": character_id,
        }

    return router
