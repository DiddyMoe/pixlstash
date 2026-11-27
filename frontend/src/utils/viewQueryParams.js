// The query params that name the view on screen: its scope (sets, people, a
// project, the Scrapheap, the unassigned view) and every filter that narrows
// it. The grid's list (`useGridFetch.buildPictureIdsQueryParams`, which adds
// its sort, text query and display keys) builds them here.

import { PIL_IMAGE_EXTENSIONS, VIDEO_EXTENSIONS } from "./media.js";
import { workflowFilterParams } from "../stores/useFilterStore";
import {
  ALL_PICTURES_ID,
  SCRAPHEAP_PICTURES_ID,
  UNASSIGNED_PICTURES_ID,
} from "../stores/useViewStore";

/**
 * The selected sets as positive ids, each once; a lone `selectedSet` as a
 * list of one.
 */
export function normalizedSetIds(selection) {
  const ids = (
    Array.isArray(selection.selectedSetIds) ? selection.selectedSetIds : []
  )
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id > 0);
  if (ids.length > 0) return Array.from(new Set(ids));
  const single = Number(selection.selectedSet);
  return Number.isFinite(single) && single > 0 ? [single] : [];
}

/** The selected people as positive ids, ascending. */
export function normalizedCharacterIds(selection) {
  return (
    Array.isArray(selection.selectedCharacterIds)
      ? selection.selectedCharacterIds
      : []
  )
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id > 0)
    .sort((a, b) => a - b);
}

/** Whether *No character* or *In no set* is on. */
export function unassignedFilterOn(filters) {
  return !!(filters.noCharacterFilter || filters.noSetFilter);
}

/**
 * "No character" and "In no set" are halves of one backend view,
 * `character_id=UNASSIGNED`: both on is the whole of it (neither a person
 * nor a set), one on narrows it with `unassigned_by`.
 */
export function appendUnassignedParams(params, filters) {
  params.set("character_id", String(UNASSIGNED_PICTURES_ID));
  if (!(filters.noCharacterFilter && filters.noSetFilter)) {
    params.set(
      "unassigned_by",
      filters.noCharacterFilter ? "character" : "set",
    );
  }
}

/**
 * The Likeness filter's cluster: only in one person's view, which is the only
 * view whose clusters it numbers.
 */
export function appendFaceClusterParams(params, pick) {
  if (!pick) return;
  params.set("face_cluster", String(pick.cluster));
  params.set("face_cluster_threshold", String(pick.threshold));
}

/** The Images / Videos switch, as the formats each one lists. */
export function appendMediaTypeParams(params, mediaTypeFilter) {
  const extensions =
    mediaTypeFilter === "images"
      ? PIL_IMAGE_EXTENSIONS
      : mediaTypeFilter === "videos"
        ? VIDEO_EXTENSIONS
        : [];
  for (const ext of extensions) params.append("format", ext.toUpperCase());
}

/**
 * The view's scope: the sets, the people, one person or the Scrapheap, or the
 * unassigned view, with the project when the sidebar is in project mode.
 *
 * @param {URLSearchParams} params - Appended to.
 * @param {Object} view
 * @param {number[]} view.setIds - {@link normalizedSetIds}.
 * @param {number[]} view.characterIds - {@link normalizedCharacterIds}.
 * @param {Object} view.selection - The selection store.
 * @param {Object} view.project - The project store.
 * @param {Object} view.filters - The filter store.
 * @param {Object} [options]
 * @param {boolean} [options.unassignedOnly=true] - Send *No character* / *In
 *   no set*. The filter menu's base count leaves them out, as filters.
 * @param {boolean} [options.faceCluster=true] - Send a person's likeness
 *   cluster.
 */
export function appendViewSelectionParams(
  params,
  view,
  { unassignedOnly = true, faceCluster = true } = {},
) {
  const { setIds, characterIds, selection, project, filters } = view;
  const inProjectView = project.projectViewMode === "project";
  const selectedProject =
    project.selectedProjectId != null
      ? project.selectedProjectId
      : "UNASSIGNED";
  // Several sets or people: their one shared project, and no project filter
  // when they span more than one.
  const appendSharedProject = (ids, projectIdOf) => {
    if (!inProjectView) return;
    const pids = new Set(ids.map((id) => projectIdOf?.[id] ?? null));
    if (pids.size !== 1) return;
    const [pid] = pids;
    params.append("project_id", pid != null ? pid : "UNASSIGNED");
  };
  const selectedCharacter = selection.selectedCharacter;

  if (setIds.length > 1) {
    for (const id of setIds) params.append("set_ids", String(id));
    params.append("set_mode", selection.setMultiMode ?? "intersection");
    if (
      selection.setMultiMode === "difference" &&
      selection.setDifferenceBaseId != null
    ) {
      params.append("base_set_id", String(selection.setDifferenceBaseId));
    }
    appendSharedProject(setIds, project.setProjectIds);
  } else if (setIds.length === 1) {
    params.append("set_id", String(setIds[0]));
    if (inProjectView) params.append("project_id", selectedProject);
  } else if (characterIds.length > 1) {
    for (const id of characterIds) params.append("character_ids", String(id));
    params.append("character_mode", selection.characterMultiMode ?? "union");
    appendSharedProject(characterIds, project.characterProjectIds);
  } else if (
    selectedCharacter != null &&
    selectedCharacter !== "" &&
    selectedCharacter !== ALL_PICTURES_ID
  ) {
    if (selectedCharacter === String(SCRAPHEAP_PICTURES_ID)) {
      params.append("only_deleted", "true");
    } else {
      params.append("character_id", selectedCharacter);
      if (faceCluster) {
        appendFaceClusterParams(params, filters.faceClusterFilter);
      }
      if (inProjectView) params.append("project_id", selectedProject);
    }
  } else if (selectedCharacter === ALL_PICTURES_ID) {
    if (unassignedOnly && unassignedFilterOn(filters)) {
      appendUnassignedParams(params, filters);
    }
    if (inProjectView) params.append("project_id", selectedProject);
  }
}

/**
 * The filters that narrow the view.
 *
 * @param {URLSearchParams} params - Appended to.
 * @param {Object} view
 * @param {Object} view.filters - The filter store.
 * @param {boolean} view.applyTagFilter - The user's hidden-tags preference.
 * @param {number|null} view.referenceFolderId - The sidebar's folder
 *   selection, as its three facets.
 * @param {string|null} view.filePathPrefix
 * @param {string|null} view.importSourceFolder - The Filter menu's "Imported
 *   from".
 * @param {string|null} view.inboxPeriod - The Inbox's period, or null outside
 *   it.
 */
export function appendViewFilterParams(params, view) {
  const { filters } = view;
  appendMediaTypeParams(params, filters.mediaTypeFilter);
  (filters.comfyuiModelFilter || []).forEach((m) =>
    params.append("comfyui_model", m),
  );
  (filters.comfyuiLoraFilter || []).forEach((l) =>
    params.append("comfyui_lora", l),
  );
  workflowFilterParams(filters.workflowFilter).forEach(([name, value]) =>
    params.append(name, value),
  );
  if (filters.minScoreFilter != null) {
    params.append("min_score", filters.minScoreFilter);
  }
  if (filters.maxScoreFilter != null) {
    params.append("max_score", filters.maxScoreFilter);
  }
  if (filters.unscoredOnlyFilter) params.append("unscored", "1");
  if (filters.favouriteOnlyFilter) params.append("favourite", "1");
  if (filters.smartScoreBucketFilter != null) {
    params.append("smart_score_bucket", filters.smartScoreBucketFilter);
  }
  if (filters.resolutionBucketFilter != null) {
    params.append("resolution_bucket", filters.resolutionBucketFilter);
  }
  (filters.tagFilter || []).forEach((t) => params.append("tag", t));
  (filters.tagRejectedFilter || []).forEach((t) =>
    params.append("rejected_tag", t),
  );
  (filters.tagConfidenceAboveFilter || []).forEach((e) =>
    params.append("tag_confidence_above", e),
  );
  (filters.tagConfidenceBelowFilter || []).forEach((e) =>
    params.append("tag_confidence_below", e),
  );
  if (view.applyTagFilter) params.append("apply_tag_filter", "true");
  if (view.referenceFolderId != null) {
    params.append("reference_folder_id", String(view.referenceFolderId));
  }
  if (view.filePathPrefix != null) {
    params.append("file_path_prefix", view.filePathPrefix);
  }
  if (view.importSourceFolder != null) {
    params.append("import_source_folder", view.importSourceFolder);
  }
  if (view.inboxPeriod) params.append("inbox", view.inboxPeriod);
  if (filters.faceBboxFilter != null) {
    params.append("face_filter", filters.faceBboxFilter);
  }
  (filters.impossibleSources || []).forEach((s) =>
    params.append("impossible_tag_source", s),
  );
  // "all" is the absence of the filter, so it is expressed by omission rather
  // than by a sentinel the backend would have to know a second spelling for.
  if (filters.stackStateFilter && filters.stackStateFilter !== "all") {
    params.append("stack_state", filters.stackStateFilter);
  }
  if (filters.tagStateFilter)
    params.append("tag_state", filters.tagStateFilter);
  if (filters.sharedOnlyFilter) params.append("shared_only", "true");
}
