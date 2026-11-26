import { ref, computed, markRaw, nextTick, onScopeDispose } from "vue";
import { isReadOnly, onSessionReset } from "../utils/apiClient";
import {
  getPictureCount,
  streamPictures,
  getLikenessGroups,
  faceSearch,
  characterFaceSearch,
  setLikenessSearch,
  likenessSearch,
  searchPictures,
  listPicturesByIds,
} from "../api/pictures";
import { getCharacterSummary } from "../api/characters";
import { getProjectSummary } from "../api/projects";
import { useEntityListsStore } from "../stores/useEntityListsStore";
import { getStackColor, getStackThreshold } from "../utils/utils.js";
import { cutFaceSuggestions } from "../utils/faceSuggestionCut.js";
import {
  cutSetSuggestions,
  setCohesion,
  SET_SUGGEST_FALLBACK_THRESHOLD,
} from "../utils/setSuggestionCut.js";
import { getPictureId } from "../utils/media.js";
import { debounce } from "../utils/utils";
import {
  appendFaceClusterParams,
  appendMediaTypeParams,
  appendUnassignedParams,
  appendViewSelectionParams,
  unassignedFilterOn,
  viewQueryParams,
} from "../utils/viewQueryParams.js";
import {
  useFilterStore,
  workflowFilterParams,
} from "../stores/useFilterStore";
import { useGridStore } from "../stores/useGridStore";
import { useSelectionStore } from "../stores/useSelectionStore";
import { useUserPrefsStore } from "../stores/useUserPrefsStore";
import { useSortStore } from "../stores/useSortStore";
import { useProjectStore } from "../stores/useProjectStore";
import { useSearchStore } from "../stores/useSearchStore";
import { useInboxStore } from "../stores/useInboxStore";
import {
  ALL_PICTURES_ID,
  SCRAPHEAP_PICTURES_ID,
  UNASSIGNED_PICTURES_ID,
} from "../stores/useViewStore";

const LIKENESS_GROUPS_SORT_KEY = "LIKENESS_GROUPS";

/**
 * Manages grid image fetching: fetch state, URL query building, and the
 * debounced fetch trigger.
 *
 * The filter facets of the query and the grid geometry come from the stores
 * directly (Phase 3);
 * `props` still supplies the selection/sort/project fields that have not been
 * migrated yet.
 *
 * @param {Object} deps - Reactive state from other composables / ImageGrid
 * @param {Object} props - Component props
 * @param {Object} callbacks - Functions provided by ImageGrid or other composables
 */
export function useGridFetch(
  {
    allGridImages,
    lastFetchedGridImages,
    scrollWrapper,
    preserveScrollOnNextFetch,
    pendingScrollTop,
    lightboxOpen,
    pendingGridImages,
    pendingLightboxGridRefresh,
    visibleStart,
    visibleEnd,
    divisibleViewWindow,
    initialRender,
    rowHeight,
    sharedPictureIds,
    guestConsentState,
    guestSessionId,
    highlightNextFetch,
    hasLoadedOnce,
    previousImageIds,
    normalizedSelectedCharacterIds,
    normalizedSelectedSetIds,
    hasSetSelection,
    isSetOverlapView,
    primarySelectedSetId,
    smartScoreProgress,
    exportProgress,
    reverseImageSearchPictureIds,
    faceLikenessSearchFaceId,
    faceSearchCharacter,
    faceSearchThreshold,
    faceSearchMinRefs,
    faceSearchRanked,
    setSuggestSet,
    setSuggestThreshold,
    setSuggestMinTags,
    setSuggestRanked,
    textSearchResults,
  },
  props,
  {
    collapseStackImages,
    mapGridImages,
    syncExpandAllStacksFromFetchedImages,
    refreshExpandedStacksAfterFetch,
    resetThumbnailState,
    triggerNewImageHighlight,
    updateVisibleThumbnails,
    forgetThumbnailRangesFrom,
    holdView,
    fetchThumbnailsBatch,
    startSmartScoreProgress,
    completeSmartScoreProgress,
    onGridFetchStart,
    onGridVisibleMetadataReady,
    onGridFetchDone,
  },
) {
  const sortStore = useSortStore();
  const projectStore = useProjectStore();
  const searchStore = useSearchStore();
  const filterStore = useFilterStore();
  const gridStore = useGridStore();
  const selectionStore = useSelectionStore();
  const userPrefsStore = useUserPrefsStore();
  const inboxStore = useInboxStore();

  // The three folder facets are one selection in the sidebar; the query wants
  // them as separate params.
  const referenceFolderIdFilter = computed(
    () => selectionStore.selectedFolderFilter?.referenceFolderId ?? null,
  );
  const filePathPrefixFilter = computed(
    () => selectionStore.selectedFolderFilter?.pathPrefix ?? null,
  );
  // The Filter menu's "Imported from" narrows any view, the Inbox included.
  const importSourceFolderFilter = computed(
    () => filterStore.importedFromFilter ?? null,
  );
  // The Inbox: the period it is narrowed to (`inbox=`), or null outside it.
  // The backend reads it as the unassigned view (no person, no set) added
  // within that period.
  const inboxPeriod = computed(() =>
    selectionStore.selectedFolderFilter?.inbox ? inboxStore.period : null,
  );

  // ============================================================
  // GRID FETCH STATE
  // ============================================================
  const imagesLoading = ref(false);
  const imagesError = ref(null);
  const totalAllPicturesCount = ref(0);
  // Whether the count above is an ANSWER or still the initial zero. The catch
  // below swallows a failure and leaves the ref at 0, which reads as "the
  // library is empty" - true for a fresh install and false for a share session
  // the summary route refuses, a backend restart, or any timeout. Anything that
  // acts on emptiness has to know which zero it is looking at.
  const totalAllPicturesCountLoaded = ref(false);
  const totalCurrentCategoryCount = ref(0);
  const gridReady = ref(false);
  const gridLoadEpoch = ref(0);
  const lastFetchKey = ref("");
  const lastFetchError = ref({ key: "", at: 0 });
  const lastFetchSuccess = ref({ key: "", at: 0 });
  // Whether the rows in lastFetchedGridImages show each stack as one card.
  // The server lists every row of a view whose scope query asks for scrapped
  // pictures (only_deleted, the Scrapheap), and stack leaders only elsewhere.
  // Each load decides this once from its view and commits it together with
  // its rows; the load's fold and every later rebuild read this one value
  // (collapseStackImages in useStackOrdering), so no caller guesses it again.
  const loadFoldsStacks = ref(true);
  const smartScoreLoadingVisible = computed(
    () =>
      !!getActiveSortKey() &&
      smartScoreProgress.visible &&
      !exportProgress.visible,
  );

  // ============================================================
  // GRID FETCH FUNCTIONS
  // ============================================================
  function getNowMs() {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  function getActiveSortKey() {
    if (typeof sortStore.selectedSort !== "string") return "";
    return sortStore.selectedSort.trim().toUpperCase();
  }

  function buildGridFetchKey() {
    const selectedSetIds = Array.isArray(selectionStore.selectedSetIds)
      ? selectionStore.selectedSetIds
          .map((id) => Number(id))
          .filter((id) => Number.isFinite(id) && id > 0)
          .sort((a, b) => a - b)
      : [];
    const selectedCharacterIds = normalizedSelectedCharacterIds.value;
    return JSON.stringify({
      selectedCharacter: selectionStore.selectedCharacter ?? null,
      selectedCharacterIds,
      isMultiCharacterView: selectedCharacterIds.length > 1,
      characterMultiMode:
        selectedCharacterIds.length > 1
          ? (selectionStore.characterMultiMode ?? "union")
          : null,
      selectedSet: selectionStore.selectedSet ?? null,
      selectedSetIds,
      isSetOverlapView: selectedSetIds.length > 1,
      setMultiMode:
        selectedSetIds.length > 1
          ? (selectionStore.setMultiMode ?? "intersection")
          : null,
      setDifferenceBaseId:
        selectedSetIds.length > 1 &&
        selectionStore.setMultiMode === "difference"
          ? (selectionStore.setDifferenceBaseId ?? null)
          : null,
      projectViewMode: projectStore.projectViewMode ?? "global",
      selectedProjectId: projectStore.selectedProjectId ?? null,
      searchQuery: searchStore.searchQuery ?? "",
      selectedSort: sortStore.selectedSort ?? "",
      selectedDescending: sortStore.selectedDescending ?? null,
      stackThreshold: sortStore.stackThreshold ?? null,
      mediaTypeFilter: filterStore.mediaTypeFilter ?? "all",
      impossibleSources: filterStore.impossibleSources ?? [],
      // Changes which pictures the grid shows, so an unforced fetch must not
      // early-return as a no-op against the previous state's key.
      stackStateFilter: filterStore.stackStateFilter ?? "all",
      favouriteOnlyFilter: Boolean(filterStore.favouriteOnlyFilter),
      similarityCharacter: sortStore.selectedSimilarityCharacter ?? null,
      comfyuiModelFilter: filterStore.comfyuiModelFilter ?? [],
      comfyuiLoraFilter: filterStore.comfyuiLoraFilter ?? [],
      // Changes which pictures the grid shows, so an unforced fetch must not
      // early-return as a no-op against the previous state's key.
      workflowFilter: new URLSearchParams(
        workflowFilterParams(filterStore.workflowFilter),
      ).toString(),
      faceClusterFilter: filterStore.faceClusterFilter,
      referenceFolderIdFilter: referenceFolderIdFilter.value ?? null,
      filePathPrefixFilter: filePathPrefixFilter.value ?? null,
      importSourceFolderFilter: importSourceFolderFilter.value ?? null,
      inboxPeriod: inboxPeriod.value,
      noCharacterFilter: filterStore.noCharacterFilter ?? false,
      noSetFilter: filterStore.noSetFilter ?? false,
      applyTagFilter: userPrefsStore.applyTagFilter ?? false,
      reverseImageSearchPictureIds: reverseImageSearchPictureIds?.value ?? [],
      faceLikenessSearchFaceId: faceLikenessSearchFaceId?.value ?? null,
      faceSearchCharacterId: faceSearchCharacter?.value?.id ?? null,
      // Both suggestion knobs belong in the key: moving either changes which
      // pictures the grid shows, so a fetch that early-returns as a no-op would
      // leave the grid disagreeing with the count in the bar. The rebuild costs
      // no network call - the ranked list and its rows are both cached.
      faceSearchThreshold: faceSearchCharacter?.value
        ? (faceSearchThreshold?.value ?? null)
        : null,
      faceSearchMinRefs: faceSearchCharacter?.value
        ? (faceSearchMinRefs?.value ?? null)
        : null,
      // Same reasoning for the set suggestion's two knobs.
      setSuggestSetId: setSuggestSet?.value?.id ?? null,
      setSuggestThreshold: setSuggestSet?.value
        ? (setSuggestThreshold?.value ?? null)
        : null,
      setSuggestMinTags: setSuggestSet?.value
        ? (setSuggestMinTags?.value ?? null)
        : null,
      // Narrowing to text matches changes which pictures the grid shows.
      textMatchesOnly: searchStore.textMatchesOnly === true,
      // The queries the reads send. A filter or derived parameter that changes
      // a request changes the key, including one added to a builder later, so
      // two loads with the same key ask the server for the same rows.
      stream: buildStreamSelectionSuffix(),
      ids: buildPictureIdsQueryParams(),
      likeness: buildLikenessGroupQueryParams(),
    });
  }

  // The view as `utils/viewQueryParams` reads it: the selection ImageGrid
  // derives, the stores, and the folder facets above. The helpers below hand
  // it to that module, so the grid's builders and the Details panel share one
  // definition of the view's scope and filters.
  function _gridView() {
    return {
      setIds: normalizedSelectedSetIds.value,
      characterIds: normalizedSelectedCharacterIds.value,
      selection: selectionStore,
      project: projectStore,
      filters: filterStore,
      applyTagFilter: userPrefsStore.applyTagFilter,
      referenceFolderId: referenceFolderIdFilter.value,
      filePathPrefix: filePathPrefixFilter.value,
      importSourceFolder: importSourceFolderFilter.value,
      inboxPeriod: inboxPeriod.value,
    };
  }

  function _unassignedFilterOn() {
    return unassignedFilterOn(filterStore);
  }

  function _setUnassignedParams(params) {
    appendUnassignedParams(params, filterStore);
  }

  function _appendFaceClusterParams(params) {
    appendFaceClusterParams(params, filterStore.faceClusterFilter);
  }

  function _appendSelectionParams(params, options) {
    appendViewSelectionParams(params, _gridView(), options);
  }

  function _appendMediaTypeParams(params) {
    appendMediaTypeParams(params, filterStore.mediaTypeFilter);
  }

  /**
   * The view's query, with the result pill's "In text" switch when the view is
   * a text search and the switch is on. The grid's search sends this and an
   * export of the view sends it too, so the server narrows both by one rule.
   */
  function searchViewQueryParams() {
    const params = new URLSearchParams(buildPictureIdsQueryParams());
    if (params.has("query") && searchStore.textMatchesOnly === true) {
      params.set("text_match_only", "true");
    }
    return params.toString();
  }

  // The view's scope and filters (`utils/viewQueryParams`, which the Details
  // panel's stats read uses too), then what only the list needs: its order,
  // the text query in its place, and the fields a tile shows.
  function buildPictureIdsQueryParams() {
    const params = viewQueryParams(_gridView());
    if (
      sortStore.selectedSort === "CHARACTER_LIKENESS" &&
      sortStore.selectedSimilarityCharacter
    ) {
      params.append(
        "reference_character_id",
        sortStore.selectedSimilarityCharacter,
      );
    }
    if (searchStore.searchQuery && searchStore.searchQuery.trim()) {
      params.append("query", searchStore.searchQuery.trim());
    } else {
      if (sortStore.selectedSort && sortStore.selectedSort.trim()) {
        params.append("sort", sortStore.selectedSort.trim());
      }
      if (typeof sortStore.selectedDescending === "boolean") {
        params.append(
          "descending",
          sortStore.selectedDescending ? "true" : "false",
        );
      } else {
        console.warn(
          "[ImageGrid.vue] selectedDescending is not boolean, skipping param. Type:",
          typeof sortStore.selectedDescending,
        );
      }
    }
    params.append("fields", "grid");
    // For rejected-consent guests: pass the in-memory session ID so the backend
    // can overlay their scores for the current page session (no cookie available).
    if (
      isReadOnly.value &&
      guestConsentState.value === "rejected" &&
      guestSessionId.value
    ) {
      params.append("guest_session_id", guestSessionId.value);
    }
    return params.toString();
  }

  /**
   * The grid's view with none of the filter menu's filters: what the filter
   * strip's "of N" counts, and what each menu row adds one filter to for its
   * own count. Stack leaders only, like the grid stream, so the numbers agree
   * with the tiles. "No character" is a filter here, so it is left out.
   * `null` during a search: `/pictures/count` cannot count a search's hits, and
   * no number is better than the view's number beside 40 search results.
   */
  function buildFilterCountBaseQuery() {
    if (searchStore.searchQuery && searchStore.searchQuery.trim()) return null;
    const params = new URLSearchParams();
    _appendSelectionParams(params, { unassignedOnly: false, faceCluster: false });
    params.append("stack_leaders_only", "true");
    // Character likeness changes the row set, so the count needs the sort.
    if (
      sortStore.selectedSort === "CHARACTER_LIKENESS" &&
      sortStore.selectedSimilarityCharacter
    ) {
      params.append("sort", "CHARACTER_LIKENESS");
      params.append(
        "reference_character_id",
        sortStore.selectedSimilarityCharacter,
      );
    }
    if (referenceFolderIdFilter.value != null) {
      params.append(
        "reference_folder_id",
        String(referenceFolderIdFilter.value),
      );
    }
    if (filePathPrefixFilter.value != null) {
      params.append("file_path_prefix", filePathPrefixFilter.value);
    }
    if (inboxPeriod.value) params.append("inbox", inboxPeriod.value);
    if (userPrefsStore.applyTagFilter) {
      params.append("apply_tag_filter", "true");
    }
    return params.toString();
  }

  function buildLikenessGroupQueryParams() {
    const params = new URLSearchParams();
    _appendSelectionParams(params);
    _appendMediaTypeParams(params);
    (filterStore.comfyuiModelFilter || []).forEach((m) =>
      params.append("comfyui_model", m),
    );
    (filterStore.comfyuiLoraFilter || []).forEach((l) =>
      params.append("comfyui_lora", l),
    );
    workflowFilterParams(filterStore.workflowFilter).forEach(([name, value]) =>
      params.append(name, value),
    );
    if (filterStore.minScoreFilter != null) {
      params.append("min_score", filterStore.minScoreFilter);
    }
    if (filterStore.maxScoreFilter != null) {
      params.append("max_score", filterStore.maxScoreFilter);
    }
    if (filterStore.unscoredOnlyFilter) {
      params.append("unscored", "1");
    }
    if (filterStore.favouriteOnlyFilter) {
      params.append("favourite", "1");
    }
    if (filterStore.smartScoreBucketFilter != null) {
      params.append("smart_score_bucket", filterStore.smartScoreBucketFilter);
    }
    if (filterStore.resolutionBucketFilter != null) {
      params.append("resolution_bucket", filterStore.resolutionBucketFilter);
    }
    (filterStore.tagFilter || []).forEach((t) => params.append("tag", t));
    (filterStore.tagRejectedFilter || []).forEach((t) =>
      params.append("rejected_tag", t),
    );
    (filterStore.tagConfidenceAboveFilter || []).forEach((e) =>
      params.append("tag_confidence_above", e),
    );
    (filterStore.tagConfidenceBelowFilter || []).forEach((e) =>
      params.append("tag_confidence_below", e),
    );
    if (filterStore.faceBboxFilter != null) {
      params.append("face_filter", filterStore.faceBboxFilter);
    }
    // Why the untagged are untagged (the stats rail's rows). Every view's
    // list and count must carry it, or the rail's row would narrow nothing.
    if (filterStore.tagStateFilter != null) {
      params.append("tag_state", filterStore.tagStateFilter);
    }
    (filterStore.impossibleSources || []).forEach((s) =>
      params.append("impossible_tag_source", s),
    );
    // "all" is the absence of the filter, so it is expressed by omission rather
    // than by a sentinel the backend would have to know a second spelling for.
    if (
      filterStore.stackStateFilter &&
      filterStore.stackStateFilter !== "all"
    ) {
      params.append("stack_state", filterStore.stackStateFilter);
    }
    if (userPrefsStore.applyTagFilter) {
      params.append("apply_tag_filter", "true");
    }
    if (filterStore.sharedOnlyFilter) {
      params.append("shared_only", "true");
    }
    return params.toString();
  }

  // The selection, sort, media-type and filter part of the stream and count
  // queries. Both reads use it, so the count covers exactly the rows the
  // stream returns.
  function buildStreamSelectionSuffix() {
    const _charIds = normalizedSelectedCharacterIds.value;
    const _selChar = selectionStore.selectedCharacter;
    // Pass sort/descending to both the stream and the count URLs.  The sort
    // is not always count-neutral: CHARACTER_LIKENESS joins through Face and
    // changes the row set, so the count must run over the same query as the
    // stream or the placeholder grid ends up larger than the stream can fill.
    const _sort = sortStore.selectedSort?.trim();
    const _desc =
      typeof sortStore.selectedDescending === "boolean"
        ? sortStore.selectedDescending
        : true;
    // For CHARACTER_LIKENESS the backend also needs reference_character_id in
    // both the stream and count URLs - the reference determines the row set.
    const _refCharSuffix =
      _sort === "CHARACTER_LIKENESS" &&
      sortStore.selectedSimilarityCharacter
        ? `&reference_character_id=${encodeURIComponent(sortStore.selectedSimilarityCharacter)}`
        : "";
    const _sortSuffix = _sort
      ? `&sort=${encodeURIComponent(_sort)}&descending=${_desc}${_refCharSuffix}`
      : "";
    // Build character/set + project filter params for count and stream URLs.
    const _charP = new URLSearchParams();
    if (hasSetSelection.value) {
      // Set view - mirrors _appendSelectionParams set branch.
      if (isSetOverlapView.value) {
        for (const setId of normalizedSelectedSetIds.value) {
          _charP.append("set_ids", String(setId));
        }
        _charP.set(
          "set_mode",
          selectionStore.setMultiMode ?? "intersection",
        );
        if (
          selectionStore.setMultiMode === "difference" &&
          selectionStore.setDifferenceBaseId != null
        ) {
          _charP.set(
            "base_set_id",
            String(selectionStore.setDifferenceBaseId),
          );
        }
        if (projectStore.projectViewMode === "project") {
          const _pidSet = new Set(
            normalizedSelectedSetIds.value.map(
              (id) => projectStore.setProjectIds?.[id] ?? null,
            ),
          );
          if (_pidSet.size === 1) {
            const _pid = [..._pidSet][0];
            _charP.set(
              "project_id",
              _pid != null ? String(_pid) : "UNASSIGNED",
            );
          }
        }
      } else if (primarySelectedSetId.value != null) {
        _charP.set("set_id", String(primarySelectedSetId.value));
        if (projectStore.projectViewMode === "project") {
          _charP.set(
            "project_id",
            projectStore.selectedProjectId != null
              ? String(projectStore.selectedProjectId)
              : "UNASSIGNED",
          );
        }
      }
    } else if (_charIds.length > 1) {
      for (const id of _charIds) _charP.append("character_ids", String(id));
      _charP.set(
        "character_mode",
        selectionStore.characterMultiMode ?? "union",
      );
      // Only apply project filter when all selected characters share the same project.
      if (projectStore.projectViewMode === "project") {
        const _pidSet = new Set(
          _charIds.map(
            (id) => projectStore.characterProjectIds?.[id] ?? null,
          ),
        );
        if (_pidSet.size === 1) {
          const _pid = [..._pidSet][0];
          _charP.set(
            "project_id",
            _pid != null ? String(_pid) : "UNASSIGNED",
          );
        }
      }
    } else if (_selChar === String(SCRAPHEAP_PICTURES_ID)) {
      _charP.set("only_deleted", "true");
    } else if (
      _selChar != null &&
      _selChar !== "" &&
      _selChar !== ALL_PICTURES_ID
    ) {
      _charP.set("character_id", String(_selChar));
      _appendFaceClusterParams(_charP);
      if (projectStore.projectViewMode === "project") {
        _charP.set(
          "project_id",
          projectStore.selectedProjectId != null
            ? String(projectStore.selectedProjectId)
            : "UNASSIGNED",
        );
      }
    } else if (
      _selChar === ALL_PICTURES_ID &&
      _unassignedFilterOn()
    ) {
      _setUnassignedParams(_charP);
      if (projectStore.projectViewMode === "project") {
        _charP.set(
          "project_id",
          projectStore.selectedProjectId != null
            ? String(projectStore.selectedProjectId)
            : "UNASSIGNED",
        );
      }
    } else if (projectStore.projectViewMode === "project") {
      _charP.set(
        "project_id",
        projectStore.selectedProjectId != null
          ? String(projectStore.selectedProjectId)
          : "UNASSIGNED",
      );
    }
    if (referenceFolderIdFilter.value != null) {
      _charP.set(
        "reference_folder_id",
        String(referenceFolderIdFilter.value),
      );
    }
    if (importSourceFolderFilter.value != null) {
      _charP.set(
        "import_source_folder",
        String(importSourceFolderFilter.value),
      );
    }
    if (inboxPeriod.value) _charP.set("inbox", inboxPeriod.value);
    // Filter params
    if (filePathPrefixFilter.value != null) {
      _charP.set("file_path_prefix", String(filePathPrefixFilter.value));
    }
    const _charSuffix = _charP.size ? `&${_charP.toString()}` : "";
    // Build media type format filter params for count and stream URLs.
    const _formatP = new URLSearchParams();
    _appendMediaTypeParams(_formatP);
    const _formatSuffix = _formatP.size ? `&${_formatP.toString()}` : "";
    // Build filter-menu params for count and stream URLs.
    // Each labelled block corresponds to a separate commit - remove any single
    // block to bisect a regression.
    const _filterP = new URLSearchParams();
    // Filter params: score range
    if (filterStore.minScoreFilter != null)
      _filterP.set("min_score", String(filterStore.minScoreFilter));
    if (filterStore.maxScoreFilter != null)
      _filterP.set("max_score", String(filterStore.maxScoreFilter));
    if (filterStore.unscoredOnlyFilter) _filterP.set("unscored", "1");
    if (filterStore.favouriteOnlyFilter) _filterP.set("favourite", "1");
    // Filter params: smart score bucket
    if (filterStore.smartScoreBucketFilter != null)
      _filterP.set(
        "smart_score_bucket",
        String(filterStore.smartScoreBucketFilter),
      );
    // Filter params: resolution bucket
    if (filterStore.resolutionBucketFilter != null)
      _filterP.set(
        "resolution_bucket",
        String(filterStore.resolutionBucketFilter),
      );
    // Filter params: ComfyUI model / LoRA
    (filterStore.comfyuiModelFilter || []).forEach((m) =>
      _filterP.append("comfyui_model", m),
    );
    (filterStore.comfyuiLoraFilter || []).forEach((l) =>
      _filterP.append("comfyui_lora", l),
    );
    // Filter params: the ticked workflows' pictures (F7, #1797)
    workflowFilterParams(filterStore.workflowFilter).forEach(
      ([name, value]) => _filterP.append(name, value),
    );
    // Filter params: tag filters
    (filterStore.tagFilter || []).forEach((t) => _filterP.append("tag", t));
    (filterStore.tagRejectedFilter || []).forEach((t) =>
      _filterP.append("rejected_tag", t),
    );
    (filterStore.tagConfidenceAboveFilter || []).forEach((e) =>
      _filterP.append("tag_confidence_above", e),
    );
    (filterStore.tagConfidenceBelowFilter || []).forEach((e) =>
      _filterP.append("tag_confidence_below", e),
    );
    // Filter params: face bbox filter
    if (filterStore.faceBboxFilter != null)
      _filterP.set("face_filter", String(filterStore.faceBboxFilter));
    // Filter params: why the untagged are untagged (the stats rail's rows)
    if (filterStore.tagStateFilter != null)
      _filterP.set("tag_state", String(filterStore.tagStateFilter));
    // Filter params: impossible-tag sources (repeatable, OR'd)
    (filterStore.impossibleSources || []).forEach((s) =>
      _filterP.append("impossible_tag_source", s),
    );
    // Filter params: stack state. "all" is the absence of the filter, so it
    // is expressed by omission rather than by a sentinel the backend would
    // have to know a second spelling for. This is the default grid path;
    // the search and likeness-group builders carry the same param, and
    // leaving it out here was why moving the Stacks segments did nothing.
    if (
      filterStore.stackStateFilter &&
      filterStore.stackStateFilter !== "all"
    )
      _filterP.set("stack_state", String(filterStore.stackStateFilter));
    // Filter params: shared only
    if (filterStore.sharedOnlyFilter) _filterP.set("shared_only", "true");
    // Filter params: hidden-tag filter
    if (userPrefsStore.applyTagFilter)
      _filterP.set("apply_tag_filter", "true");
    const _filterSuffix = _filterP.size ? `&${_filterP.toString()}` : "";
    return `${_charSuffix}${_sortSuffix}${_formatSuffix}${_filterSuffix}`;
  }

  // ============================================================
  // GRID FETCH
  // ============================================================
  // The load that owns the grid right now: its fetch key, the controller that
  // cancels its reads, and its promise. Null when no load is running.
  let inFlight = null;

  // The pictures taken out of the grid while the running load still streams
  // it in, by id, or null when no stream is running. removeImagesById reports
  // them through noteRemovedDuringStream. A batch the server answered before
  // the delete committed still lists them, and must not write them back.
  let streamRemovedIds = null;

  /**
   * Tell the running stream that these pictures have left the grid, so no
   * batch still to land writes them in again. Does nothing when no stream
   * is running.
   *
   * @param {Array<number|string>} ids
   */
  function noteRemovedDuringStream(ids) {
    if (!streamRemovedIds) return;
    for (const id of ids) {
      if (id != null) streamRemovedIds.add(String(id));
    }
  }

  // Cancel the running load and move the epoch on, so nothing it still has in
  // flight can write into the grid or finalize a newer load's state. Neither
  // it nor a re-read queued behind it (runTrailingLoad) will reach its end,
  // so no load is left to put back a place pending for its view.
  function cancelGridLoad() {
    const current = inFlight;
    inFlight = null;
    gridLoadEpoch.value += 1;
    if (current && pendingScrollTop.value?.key === current.key) {
      pendingScrollTop.value = null;
    }
    current?.controller.abort();
  }

  // A scroll place: where a load that keeps the scroll puts the view back
  // once its answer is in, for the view `key` it was recorded in.
  // `pendingScrollTop` holds at most one. The loads of its view share it and
  // the first to reach its end puts the view back there; a load of another
  // view, a cancel, and a user scroll (ImageGrid) drop it. A load knows its
  // place by identity, so the place is kept raw: a deep ref would hand back
  // a reactive proxy that never equals the load's own object.
  function newScrollPlace(key) {
    return markRaw({ top: scrollWrapper.value.scrollTop, key });
  }

  /**
   * Record where the view is now for the next load of this view to put back:
   * the gridVersion refresh calls this, then clears the flag that says to
   * keep the scroll before its debounced re-read runs.
   */
  function recordScrollPlace() {
    pendingScrollTop.value = scrollWrapper.value
      ? newScrollPlace(buildGridFetchKey())
      : null;
  }

  /**
   * The place the load this call starts for view `key` puts the view back
   * to, or null.
   *
   * A place pending for the same view is shared, whatever the call's own
   * flag says: the one the gridVersion refresh recorded (recordScrollPlace),
   * or the one the running load of the view, or the re-read already queued
   * behind it, holds. Whichever load reaches its end first puts it back, and
   * only once. A place for another view is dropped, since the view it was
   * recorded in is gone. Otherwise a call that keeps the scroll records where
   * the view is now, and a call that does not gets none, so it cannot put the
   * view back where a load it cancels began.
   */
  function claimScrollPlace(key, keepsScroll) {
    const pending = pendingScrollTop.value;
    if (pending?.key === key) return pending;
    pendingScrollTop.value = null;
    if (!keepsScroll || !scrollWrapper.value) return null;
    const place = newScrollPlace(key);
    pendingScrollTop.value = place;
    return place;
  }

  // A credential change: the previous principal's load must never land, and
  // its completion records must not debounce the first read of the next one.
  function resetGridLoadForSession() {
    cancelGridLoad();
    lastFetchKey.value = "";
    lastFetchSuccess.value = { key: "", at: 0 };
    lastFetchError.value = { key: "", at: 0 };
    imagesLoading.value = false;
    gridReady.value = true;
  }

  const unsubscribeSessionReset = onSessionReset(resetGridLoadForSession);
  onScopeDispose(() => {
    unsubscribeSessionReset();
    cancelGridLoad();
  });

  /**
   * Load the grid for the current view.
   *
   * At most one load runs at a time. A call for the view that is already
   * loading waits for it and then re-reads once, however many such calls
   * arrived meanwhile, because the data may have changed under the running
   * read. A call that passes `join: true` (it only needs the view loaded, not
   * re-read) shares the running load instead, and is skipped if the same view
   * finished loading moments ago. A call for a different view cancels the
   * running load.
   */
  async function fetchAllGridImages(options = {}) {
    const force = options?.force === true;
    const join = options?.join === true;
    // Capture scroll-preservation intent *synchronously* before any await so
    // that it is not affected by the gridVersion watcher clearing it later.
    const fetchStartedWithPreserveScroll = preserveScrollOnNextFetch.value;
    const fetchKey = buildGridFetchKey();
    const current = inFlight;
    if (current && current.key === fetchKey) {
      if (!force && join) return current.promise;
      const place = claimScrollPlace(fetchKey, fetchStartedWithPreserveScroll);
      const runTrailing = () =>
        runTrailingLoad(current, options, fetchStartedWithPreserveScroll, place);
      return current.promise.then(runTrailing, runTrailing);
    }
    const now = Date.now();
    if (
      !force &&
      join &&
      lastFetchSuccess.value.key === fetchKey &&
      now - lastFetchSuccess.value.at < 1200
    ) {
      return undefined;
    }
    if (
      !force &&
      join &&
      lastFetchError.value.key === fetchKey &&
      now - lastFetchError.value.at < 2500
    ) {
      return undefined;
    }
    return beginGridLoad(
      fetchKey,
      options,
      fetchStartedWithPreserveScroll,
      claimScrollPlace(fetchKey, fetchStartedWithPreserveScroll),
    );
  }

  // The forced re-read behind `previous`, run once it settles. Every call for
  // its view that arrived while it ran and did not join lands here: the first
  // starts the re-read and the rest join it. Nothing runs once the view has
  // moved on or a session reset cancelled `previous`.
  function runTrailingLoad(
    previous,
    options,
    fetchStartedWithPreserveScroll,
    place,
  ) {
    if (previous.controller.signal.aborted) return undefined;
    if (buildGridFetchKey() !== previous.key) return undefined;
    if (inFlight?.key === previous.key) {
      // The re-read stands for every call that joins it, so a place one of
      // them recorded is the re-read's to put back.
      if (place && !inFlight.place) inFlight.place = place;
      return inFlight.promise;
    }
    return beginGridLoad(
      previous.key,
      { ...options, force: true },
      fetchStartedWithPreserveScroll,
      place,
    );
  }

  function beginGridLoad(
    fetchKey,
    options,
    fetchStartedWithPreserveScroll,
    place,
  ) {
    inFlight?.controller.abort();
    const record = {
      key: fetchKey,
      controller: new AbortController(),
      promise: null,
      // The scroll place this load puts back at its end (claimScrollPlace).
      place,
    };
    inFlight = record;
    record.promise = runGridLoad(
      record,
      options,
      fetchStartedWithPreserveScroll,
    ).finally(() => {
      if (inFlight === record) inFlight = null;
    });
    return record.promise;
  }

  // Which read fills the grid for the current settings: the first of these
  // that applies, else the stream. A text search outranks the suggestion and
  // likeness searches, and the likeness-groups sort outranks a text search.
  // runGridLoad branches on it, and the reconnect read re-reads a view only
  // while it is still in the mode it loaded in.
  function currentFetchMode() {
    if (sortStore.selectedSort === LIKENESS_GROUPS_SORT_KEY) {
      return "likeness-groups";
    }
    if (searchStore.searchQuery?.trim()) return "text-search";
    if (reverseImageSearchPictureIds?.value?.length) {
      return "reverse-image-search";
    }
    if (faceSearchCharacter?.value?.id) return "character-face-search";
    if (setSuggestSet?.value?.id) return "set-suggest-search";
    if (faceLikenessSearchFaceId?.value) return "face-likeness-search";
    return "stream";
  }

  // The suggestion lists as their sliders cut them.
  function cutFaceView(ranked) {
    return cutFaceSuggestions(
      ranked,
      faceSearchThreshold?.value ?? 0,
      faceSearchMinRefs?.value ?? 1,
    );
  }

  function cutSetView(ranked) {
    return cutSetSuggestions(
      ranked,
      setSuggestThreshold?.value ?? 0,
      setSuggestMinTags?.value ?? 0,
    );
  }

  // A ranked search's grid rows, in rank order, less any picture the server
  // no longer lists.
  function inRankOrder(idOrder, rows) {
    const picturesById = {};
    for (const pic of rows) picturesById[pic.id] = pic;
    return idOrder.map((id) => picturesById[id]).filter(Boolean);
  }

  async function rowsInRankOrder(idOrder, signal) {
    if (!idOrder.length) return [];
    const rows = await listPicturesByIds(idOrder, { fields: "grid", signal });
    return inRankOrder(idOrder, Array.isArray(rows) ? rows : []);
  }

  // A reconnect read's answer must be a list: an empty one reads as "every
  // card left", so a malformed one must not be taken for it.
  function reconcileList(body, read) {
    if (!Array.isArray(body)) {
      throw new Error(`readIdListForReconcile: ${read} returned no list`);
    }
    return body;
  }

  /**
   * Re-read the rows of an id-list view (the likeness groups, a text search,
   * a suggestion or likeness search) for the reconnect reconcile: the read
   * runGridLoad makes for the current mode, asked of the server rather than a
   * cached ranked list, and writing nothing. Each of these reads returns the
   * view's whole list. Rejects on an answer that is not a list.
   *
   * @param {{signal?: AbortSignal}} [options]
   * @returns {Promise<Array<Object>>} the view's rows, in load order.
   */
  async function readIdListForReconcile({ signal } = {}) {
    const mode = currentFetchMode();
    if (mode === "likeness-groups") {
      const data = await getLikenessGroups(
        getStackThreshold(sortStore.stackThreshold),
        buildLikenessGroupQueryParams(),
        { signal },
      );
      return reconcileList(data, "the likeness-groups read");
    }
    if (mode === "text-search") {
      const body = await searchPictures(searchStore.searchQuery.trim(), {
        query: searchViewQueryParams(),
        signal,
      });
      return reconcileList(body, "the text search");
    }
    let ranked;
    if (mode === "reverse-image-search") {
      ranked = await likenessSearch(reverseImageSearchPictureIds.value, {
        signal,
      });
    } else if (mode === "character-face-search") {
      ranked = await characterFaceSearch(faceSearchCharacter.value.id, {
        signal,
      });
    } else if (mode === "set-suggest-search") {
      ranked = await setLikenessSearch(setSuggestSet.value.id, { signal });
    } else if (mode === "face-likeness-search") {
      ranked = await faceSearch(faceLikenessSearchFaceId.value, { signal });
    } else {
      throw new Error(
        `readIdListForReconcile: the ${mode} view is not an id-list view`,
      );
    }
    ranked = reconcileList(ranked, `the ${mode}`);
    if (mode === "character-face-search") ranked = cutFaceView(ranked);
    if (mode === "set-suggest-search") ranked = cutSetView(ranked);
    const idOrder = ranked.map((r) => r.picture_id);
    if (!idOrder.length) return [];
    const rows = await listPicturesByIds(idOrder, { fields: "grid", signal });
    return inRankOrder(idOrder, reconcileList(rows, "the by-ids read"));
  }

  async function runGridLoad(record, options, fetchStartedWithPreserveScroll) {
    const { key: fetchKey } = record;
    const { signal } = record.controller;
    const force = options?.force === true;
    const activeSortKey = getActiveSortKey();
    const isSortedFetch = !!activeSortKey;
    const fetchStartedAt = getNowMs();
    let fetchMode = "default";
    let fetchSucceeded = false;
    let fetchError = null;
    let sortedFetchStartedAt = 0;
    const fetchPhaseTimings = {
      countMs: null,
      placeholderMs: null,
      firstBatchMs: null,
      tailBatchMs: null,
      backgroundTotalMs: 0,
      backgroundNetworkTotalMs: 0,
      backgroundUiTotalMs: 0,
      backgroundSlowestBatchMs: 0,
      backgroundSlowestNetworkBatchMs: 0,
      backgroundSlowestUiBatchMs: 0,
      backgroundBatchCount: 0,
      postProcessMs: null,
    };
    // The view the cards on screen were read for, so a stream load can tell a
    // re-read of the same view (keep the cards until the new rows land) from
    // a load of another view.
    const previousFetchKey = lastFetchKey.value;
    lastFetchKey.value = fetchKey;
    const loadId = (gridLoadEpoch.value += 1);
    // False once a newer load, a session reset or teardown has taken the grid:
    // from then on this load must not write rows or state.
    const isCurrent = () => loadId === gridLoadEpoch.value;
    // A stream load's set of pictures removed while it runs (see
    // streamRemovedIds). Held here so the finally below can let go of it
    // however the load ends: done, aborted, failed or superseded.
    let loadRemovedIds = null;
    // This load's place, taken out of pendingScrollTop by the finally below
    // while it is still pending, for the end-of-load restore.
    let placeToRestore = null;
    if (typeof onGridFetchStart === "function") {
      onGridFetchStart({
        loadId,
        fetchKey,
        force,
        selectedSort: sortStore.selectedSort ?? null,
        selectedCharacter: selectionStore.selectedCharacter ?? null,
        selectedSet: selectionStore.selectedSet ?? null,
        visibleStart: visibleStart.value,
        visibleEnd: visibleEnd.value,
      });
    }
    gridReady.value = false;
    imagesLoading.value = true;
    imagesError.value = null;
    if (isSortedFetch && options?.showProgress === true) {
      sortedFetchStartedAt = getNowMs();
      startSmartScoreProgress(loadId, activeSortKey);
    }
    // A suggestion search's ranked list and its picture rows, cached in
    // `cache` against the subject `id` (under `key`) so neither slider costs a
    // round trip; only a new subject or a force refetches. Null when a newer
    // fetch overtook this one.
    async function rankedWithRows(cache, key, id, search, phase) {
      const cached = cache?.value;
      if (!force && cached?.[key] === id && cached.matches)
        return cached.matches;
      let raw;
      try {
        raw = await search(id, { signal });
      } catch (error) {
        error.gridFetchPhase = phase;
        throw error;
      }
      if (!isCurrent()) return null;
      const ranked = Array.isArray(raw) ? raw : [];
      const rowsById = {};
      if (ranked.length) {
        const rows = await listPicturesByIds(
          ranked.map((r) => r.picture_id),
          { fields: "grid", signal },
        );
        if (!isCurrent()) return null;
        for (const pic of Array.isArray(rows) ? rows : []) {
          rowsById[pic.id] = pic;
        }
      }
      if (cache) cache.value = { [key]: id, matches: ranked, rowsById };
      return ranked;
    }
    try {
      let images = [];

      fetchMode = currentFetchMode();
      const foldsStacks = !viewQueryParams(_gridView()).has("only_deleted");
      if (fetchMode === "likeness-groups") {
        const threshold = getStackThreshold(sortStore.stackThreshold);
        const likenessGroupParams = buildLikenessGroupQueryParams();
        const data = await getLikenessGroups(threshold, likenessGroupParams, {
          signal,
        });
        if (!isCurrent()) {
          if (isSortedFetch && options?.showProgress === true)
            completeSmartScoreProgress(loadId, 0, false);
          return;
        }
        const likenessGroupImages = Array.isArray(data) ? data : [];
        images = likenessGroupImages.map((img) => {
          const stackIndex =
            typeof img.stack_index === "number"
              ? img.stack_index
              : typeof img.stackIndex === "number"
                ? img.stackIndex
                : null;
          return {
            ...img,
            stackIndex,
            stackColor:
              typeof stackIndex === "number" ? getStackColor(stackIndex) : null,
          };
        });
      } else if (fetchMode === "character-face-search") {
        // "Suggest more pictures of <person>" (#636): query with the character's
        // reference faces, and let BOTH suggestion sliders re-cut the SAME
        // ranked list. The ranked refs and their picture rows are both cached
        // against the character id, so dragging either costs no round trip,
        // which is the difference between a slider that feels live and one that
        // stutters. That is also why the reference-agreement knob is served by
        // `reference_likeness` on each match rather than by re-querying with a
        // different combine mode: a server-side k-of-n would put a round trip
        // under a drag. Only a change of character (or an explicit force)
        // refetches.
        const ranked = await rankedWithRows(
          faceSearchRanked,
          "characterId",
          faceSearchCharacter.value.id,
          characterFaceSearch,
          "character-face-search-request",
        );
        if (!ranked) {
          if (isSortedFetch && options?.showProgress === true)
            completeSmartScoreProgress(loadId, 0, false);
          return;
        }
        const rowsById = faceSearchRanked?.value?.rowsById ?? {};
        images = cutFaceView(ranked)
          .map((r) => rowsById[r.picture_id])
          .filter(Boolean);
      } else if (fetchMode === "set-suggest-search") {
        // "Suggest more pictures for <set>" (#1489): the same cached-ranked-list
        // shape as the person search above, so neither slider costs a round
        // trip. Only a change of set (or an explicit force) refetches.
        const ranked = await rankedWithRows(
          setSuggestRanked,
          "setId",
          setSuggestSet.value.id,
          setLikenessSearch,
          "set-suggest-search-request",
        );
        if (!ranked) {
          if (isSortedFetch && options?.showProgress === true)
            completeSmartScoreProgress(loadId, 0, false);
          return;
        }
        // Seat the strength slider at the set's own cohesion the first time
        // the list arrives: no fixed default suits both a tight photoshoot
        // and a loose theme. Floored, so the seat never lands above the
        // set's own median. A one-picture set has no cohesion and takes the
        // fixed fallback. A refetch keeps whatever the user dragged to.
        if (
          setSuggestThreshold &&
          setSuggestThreshold.value == null &&
          ranked.length
        ) {
          const cohesion = setCohesion(ranked);
          setSuggestThreshold.value =
            cohesion != null
              ? Math.floor(cohesion * 100) / 100
              : SET_SUGGEST_FALLBACK_THRESHOLD;
          // The seat is part of this view's key: re-key the running load so
          // a call that queued on it still gets its re-read.
          record.key = buildGridFetchKey();
        }
        const rowsById = setSuggestRanked?.value?.rowsById ?? {};
        images = cutSetView(ranked)
          .map((r) => rowsById[r.picture_id])
          .filter(Boolean);
      } else if (fetchMode === "face-likeness-search") {
        // Face likeness search: POST to face-search with source_face_id.
        const queryFaceId = faceLikenessSearchFaceId.value;
        const faceResultsRaw = await faceSearch(queryFaceId, { signal });
        const faceResults = Array.isArray(faceResultsRaw) ? faceResultsRaw : [];
        images = await rowsInRankOrder(
          faceResults.map((r) => r.picture_id),
          signal,
        );
      } else if (fetchMode === "reverse-image-search") {
        // Reverse image search: POST to likeness-search with stored CLIP embeddings.
        // Multiple IDs are combined with min similarity (must match all sources).
        const queryPicIds = reverseImageSearchPictureIds.value;
        const likenessRaw = await likenessSearch(queryPicIds, { signal });
        const likenessResults = Array.isArray(likenessRaw) ? likenessRaw : [];
        images = await rowsInRankOrder(
          likenessResults.map((r) => r.picture_id),
          signal,
        );
      } else if (fetchMode === "text-search") {
        // Use /pictures/search endpoint for text search
        // The "In text" switch is the search's to apply, so the grid and an
        // export of this view (viewExportQuery) narrow by the same rule.
        const params = searchViewQueryParams();
        const searchText = searchStore.searchQuery.trim();
        const body = await searchPictures(searchText, {
          query: params,
          signal,
        });
        const rows = Array.isArray(body) ? body : [];
        if (textSearchResults && isCurrent()) {
          textSearchResults.value = {
            key: `${searchText}\u0000${params}`,
            query: searchText,
            rows,
          };
        }
        images = rows;
      } else {
        // Lightbox open: the streaming path rebuilds a placeholder grid and
        // re-fills allGridImages from the (now possibly narrower) filter query,
        // which would drop the picture being viewed out of the grid mid-session.
        // Unlike the id-list/search modes below, streaming writes allGridImages
        // through its own return paths and never reaches the shared lightboxOpen
        // guard. Defer the whole reconcile to lightbox close instead.
        if (lightboxOpen.value) {
          pendingLightboxGridRefresh.value = true;
          // We started the sort progress bar above but are deferring this fetch
          // to lightbox-close, so it will never reach the completion below.
          // Dismiss the bar now, otherwise "Sorting by …" is stranded forever.
          if (isSortedFetch && options?.showProgress === true)
            completeSmartScoreProgress(loadId, 0, false);
          return;
        }
        // The stream leaves the scroll where it is (each removal it makes
        // above the view is held as it lands): none of its returns reach the
        // end-of-load restore, and the finally below lets its place go.
        // Streaming: COUNT(*) → placeholder grid → parallel first/last batches → background fill.
        //   1. Fast SELECT COUNT(*) → total
        //   2. Pre-build placeholder grid so END key works before streaming starts
        //   3. First batch (visible area) + last batch (END cells) in parallel
        //   4. Background stream fills the middle at BG_BATCH rows per request
        const streamSelection = buildStreamSelectionSuffix();
        if (isSortedFetch && options?.showProgress === true) {
          completeSmartScoreProgress(loadId, 0, true);
        }
        // Compute FIRST_BATCH to cover all items visible in the viewport so that
        // fetchThumbnailsBatch (called after splicing the first batch) never
        // encounters placeholder items (no id) for visible cells.  Placeholders
        // are skipped by fetchThumbnailsBatch but the range is still marked as
        // loaded, causing those cells to remain as permanent spinners.
        // Cell height is derived from clientWidth/cols (square thumbnails) rather
        // than rowHeight?.value, which may still hold the initial thumbnailSize
        // estimate when this runs (DOM measurement via updateRowHeightFromGrid
        // happens asynchronously and may not have fired yet).
        const _fbCols = gridStore.columns || 1;
        const _fbViewH = scrollWrapper.value?.clientHeight || 0;
        const _fbViewW = scrollWrapper.value?.clientWidth || 0;
        const _fbCellH =
          _fbViewW > 0
            ? Math.round(_fbViewW / _fbCols) + (gridStore.compactMode ? 0 : 24)
            : rowHeight?.value > 0
              ? rowHeight.value
              : 200;
        const _fbVisibleItems =
          _fbViewH > 0 && _fbCellH > 0
            ? Math.ceil(_fbViewH / _fbCellH) * _fbCols
            : 0;
        const FIRST_BATCH = Math.max(200, _fbVisibleItems + _fbCols * 2);
        const LAST_BATCH = Math.max(200, _fbVisibleItems + _fbCols * 2);
        const streamQuery = `fields=grid&grid_lite=true&stack_leaders_only=true${streamSelection}`;

        // Wraps a resource-module call (which resolves to the response BODY,
        // not the Axios envelope) so the grid can report per-batch timings.
        async function timeRequest(requestPromise) {
          const startedAt = getNowMs();
          const body = await requestPromise;
          return {
            body,
            elapsedMs: Math.max(0, getNowMs() - startedAt),
          };
        }

        // The cards on screen when a re-read of the same view swapped in its
        // placeholder grid, by picture id, or null. Their thumbnails and faces
        // go to the same picture's new cell, wherever it now sits, so a card
        // that stays keeps its bitmap instead of blanking and loading again.
        let carryById = null;

        // Write raw picture metadata into the placeholder grid at `offset`,
        // preserving thumbnail/face data for cells already loaded. Each row
        // replaces the cell at its own index, in place, so a batch costs its
        // own rows rather than a copy of the whole grid; nothing shifts. The
        // write reaches what reads the grid's rows, not a watcher of the grid
        // array itself, which fires only when the array is replaced. Returns
        // the rows written.
        const writeRow = (grid, idx, pic) => {
          const existing =
            (pic?.id != null ? carryById?.get(String(pic.id)) : null) ??
            grid[idx];
          const row = {
            ...pic,
            idx,
            thumbnail: existing?.thumbnail ?? null,
            faces: existing?.faces ?? [],
            penalised_tags: existing?.penalised_tags ?? [],
            thumbnail_width: pic.thumbnail_width ?? existing?.thumbnail_width,
            thumbnail_height: pic.thumbnail_height ?? existing?.thumbnail_height,
            square_crop_x: pic.square_crop_x ?? existing?.square_crop_x,
            square_crop_y: pic.square_crop_y ?? existing?.square_crop_y,
            square_crop_side: pic.square_crop_side ?? existing?.square_crop_side,
          };
          grid[idx] = row;
          return row;
        };
        const splicePictures = (pictures, offset) => {
          const grid = allGridImages.value;
          const written = [];
          for (let i = 0; i < pictures.length; i++) {
            const idx = offset + i;
            if (idx >= grid.length) break;
            written.push(writeRow(grid, idx, pictures[i]));
          }
          return written;
        };

        // The first run of placeholders after the loaded rows, as
        // [start, end), looking at most `span` cells into it. `hint` is where
        // the last batch stopped: still the run's start unless something
        // (a delete, a stack folding up) moved rows since, and then the grid
        // is read from the top. Null when no placeholder is left.
        const findGap = (grid, hint, span) => {
          let start = hint;
          const hintHolds =
            start > 0 &&
            start < grid.length &&
            grid[start - 1]?.id != null &&
            grid[start]?.id == null;
          if (!hintHolds) {
            start = 0;
            while (start < grid.length && grid[start]?.id != null) start += 1;
          }
          if (start >= grid.length) return null;
          const stop = Math.min(grid.length, start + span);
          let end = start;
          while (end < stop && grid[end]?.id == null) end += 1;
          return { start, end };
        };

        // Take the placeholders [from, to) out of the grid: their rows are not
        // on the server. The cells after them move up, so the thumbnail
        // ranges from `from` on, which are by slot, are forgotten (all of
        // them, not only the visible window's: a loaded range further down
        // would otherwise count rows loaded that moved into it). The cells
        // can sit above the view (a viewport batch loaded further down), so
        // the drop goes through the grid's hold, which scrolls up by the
        // rows they took the way a removal above the view does.
        const dropPlaceholders = (from, to) => {
          holdView(
            (_img, i) => i >= from && i < to,
            () => {
              const grid = allGridImages.value;
              const next = grid.slice(0, from);
              for (let i = to; i < grid.length; i++) {
                const img = grid[i];
                const idx = next.length;
                next.push(img.idx === idx ? img : { ...img, idx });
              }
              allGridImages.value = next;
              if (visibleEnd.value > next.length) {
                visibleEnd.value = next.length;
              }
              if (visibleStart.value > visibleEnd.value) {
                visibleStart.value = visibleEnd.value;
              }
            },
          );
          forgetThumbnailRangesFrom(from);
        };

        // Removals are tracked from before the count: the old cards stay on
        // screen while it is read, and a picture deleted then can still be
        // listed by any answer the server gives before the delete commits.
        const removedIds = new Set();
        loadRemovedIds = removedIds;
        streamRemovedIds = removedIds;

        // 1. Fast total count - single indexed SQL query.
        const countStartedAt = getNowMs();
        const countBody = await getPictureCount(
          `stack_leaders_only=true${streamSelection}`,
          { signal },
        );
        if (!isCurrent()) return;
        fetchPhaseTimings.countMs = Math.max(0, getNowMs() - countStartedAt);
        const total =
          typeof countBody?.count === "number" ? countBody.count : 0;

        // 2. Pre-build placeholder grid - scroll area immediately reflects full size.
        const placeholderStartedAt = getNowMs();
        const cols = gridStore.columns || 1;
        // Compute window count from actual viewport capacity so visibleEnd covers
        // all initially visible items even when the viewport shows more than VIEW_WINDOW.
        // Derive cell height from clientWidth/cols (square thumbnails) rather than
        // rowHeight?.value, which may still hold the initial thumbnailSize estimate
        // when this runs (DOM measurement via updateRowHeightFromGrid is async).
        const _fastViewW = scrollWrapper.value?.clientWidth || 0;
        const _fastViewH = scrollWrapper.value?.clientHeight || 0;
        const _effectiveRowHeight0 =
          _fastViewW > 0
            ? Math.round(_fastViewW / cols) + (gridStore.compactMode ? 0 : 24)
            : rowHeight?.value > 0
              ? rowHeight.value
              : Math.round(
                  Math.min(384, Math.max(128, gridStore.thumbnailSize || 128)) +
                    (gridStore.compactMode ? 0 : 24),
                );
        const _viewportItemCount0 =
          _fastViewH > 0 && _effectiveRowHeight0 > 0
            ? Math.ceil(_fastViewH / _effectiveRowHeight0) * cols
            : 0;
        const windowCount = Math.max(
          cols,
          _viewportItemCount0 || divisibleViewWindow.value || cols,
        );
        // Stale-while-revalidate: a re-read of the view already on screen (the
        // refetch after a delete, a run's end, a realtime change) keeps its
        // cards until the first batch has answered, then swaps in one step.
        // Laying the placeholders down at the count blanked every card for the
        // first batch's round trip and reloaded every thumbnail after it. A
        // load of another view, or a view that has nothing yet, lays them down
        // at once so its size shows before any row. An answered empty view is
        // written at once either way: the server has said the cards are gone.
        const revalidating =
          previousFetchKey === fetchKey &&
          total > 0 &&
          allGridImages.value.some((img) => img?.id != null);
        let streamGrid = null;
        const writePlaceholderGrid = () => {
          carryById = revalidating
            ? new Map(
                allGridImages.value
                  .filter((img) => img?.id != null)
                  .map((img) => [String(img.id), img]),
              )
            : null;
          resetThumbnailState();
          allGridImages.value = Array.from({ length: total }, (_, i) => ({
            id: null,
            idx: i,
          }));
          streamGrid = allGridImages.value;
          if (!fetchStartedWithPreserveScroll) {
            visibleStart.value = 0;
            visibleEnd.value = Math.min(total, windowCount);
          }
        };
        if (!revalidating) writePlaceholderGrid();
        gridReady.value = true; // render placeholder grid immediately
        fetchPhaseTimings.placeholderMs = Math.max(
          0,
          getNowMs() - placeholderStartedAt,
        );

        if (total === 0) {
          hasLoadedOnce.value = true;
          initialRender.value = false;
          lastFetchSuccess.value = { key: fetchKey, at: Date.now() };
          return;
        }

        // 3. First (+ optional tail) batches + pre-launch background.
        //
        // API enforces batch_limit <= 5000.
        const BG_BATCH = 5000;
        // potentialLastBatchStart: where a dedicated tail batch would begin.
        const potentialLastBatchStart = Math.max(
          FIRST_BATCH,
          total - LAST_BATCH,
        );
        const backgroundGap = Math.max(
          0,
          potentialLastBatchStart - FIRST_BATCH,
        );
        // Fetch tail in parallel with the first batch whenever the gap is large
        // enough to matter for user responsiveness.  Use a fixed threshold of
        // 1000 items - independent of BG_BATCH - so that medium-sized
        // collections (gap 1000–5000) still get the tail batch early instead of
        // waiting for a single large background request to complete.
        const TAIL_THRESHOLD = 1000;
        const shouldFetchTailEarly =
          total > FIRST_BATCH && backgroundGap > TAIL_THRESHOLD;

        // Kick off the first batch and the optional tail batch concurrently, but
        // do NOT block visible-thumbnail loading on the tail (which fills
        // off-screen end-of-grid cells and is often the slower of the two).
        // Await the first batch alone, splice it and request its thumbnails, then
        // await the tail. Both requests are already in flight, so the tail is not
        // delayed by this ordering.
        const firstReqPromise = timeRequest(
          streamPictures(streamQuery, {
            offset: 0,
            batchLimit: FIRST_BATCH,
            signal,
          }),
        );
        // Tail batch: for collections where the gap exceeds TAIL_THRESHOLD.
        // Settled into an outcome so it is never an unhandled rejection when
        // this load stops before awaiting it below (the first batch failed, or
        // a newer load took the grid). A failure is rethrown there.
        const tailReqPromise = (
          shouldFetchTailEarly
            ? timeRequest(
                streamPictures(streamQuery, {
                  offset: potentialLastBatchStart,
                  batchLimit: LAST_BATCH,
                  signal,
                }),
              )
            : Promise.resolve(null)
        ).then(
          (result) => ({ result }),
          (error) => ({ error }),
        );

        // A re-read that keeps the scroll position where the first batch does
        // not reach (a delete's refetch at row 1,500) also asks for the rows
        // on screen, with a viewport's margin each side, alongside the first
        // batch, and swaps both in at once: the viewport keeps its cards
        // across the swap instead of turning to placeholders until the
        // background stream reaches them. The rows go in at their offsets; the
        // stream fills the runs on either side of them, each from the loaded
        // row before it.
        const windowSpan = divisibleViewWindow.value || windowCount;
        const viewStart = Math.max(
          FIRST_BATCH,
          Math.min(visibleStart.value, total) - windowSpan,
        );
        const viewEnd = Math.min(
          total,
          viewStart + BG_BATCH,
          visibleEnd.value + windowSpan,
        );
        const viewReqPromise =
          revalidating &&
          fetchStartedWithPreserveScroll &&
          visibleEnd.value > FIRST_BATCH &&
          viewEnd > viewStart
            ? timeRequest(
                streamPictures(streamQuery, {
                  offset: viewStart,
                  batchLimit: viewEnd - viewStart,
                  signal,
                }),
              ).then(
                (result) => ({ result }),
                (error) => ({ error }),
              )
            : null;

        const firstResTimed = await firstReqPromise;
        if (!isCurrent()) return;
        fetchPhaseTimings.firstBatchMs = firstResTimed?.elapsedMs ?? null;
        const viewOutcome = viewReqPromise ? await viewReqPromise : null;
        if (!isCurrent()) return;
        if (viewOutcome?.error) throw viewOutcome.error;

        // A picture taken out while these were out is not written: the server
        // may have read it before the change committed. The cell it leaves is
        // filled, or dropped, by the background stream.
        const notRemoved = (pic) => !removedIds.has(String(pic?.id));
        const firstPics = (firstResTimed?.body?.pictures ?? []).filter(
          notRemoved,
        );
        if (revalidating) writePlaceholderGrid();
        splicePictures(firstPics, 0);
        if (viewOutcome) {
          // A row the first batch holds already is one that moved up across
          // the join (a delete above it between the two reads): the rows after
          // it go up with it.
          const firstIds = new Set(firstPics.map((pic) => String(pic?.id)));
          splicePictures(
            (viewOutcome.result?.body?.pictures ?? []).filter(
              (pic) => notRemoved(pic) && !firstIds.has(String(pic?.id)),
            ),
            viewStart,
          );
        }
        hasLoadedOnce.value = true;
        initialRender.value = false;
        const prefetchEnd = Math.min(
          total,
          visibleEnd.value + (divisibleViewWindow.value || windowCount),
        );
        // Request thumbnails for the actually-visible cells first; defer the
        // off-screen margin by a frame so the visible thumbnails are not queued
        // behind margin ones over the browser's limited per-origin connections.
        fetchThumbnailsBatch(visibleStart.value, visibleEnd.value, {
          reason: "initial-visible-prefetch",
        });
        if (prefetchEnd > visibleEnd.value) {
          const marginStart = visibleEnd.value;
          requestAnimationFrame(() => {
            if (!isCurrent()) return;
            fetchThumbnailsBatch(marginStart, prefetchEnd, {
              reason: "initial-margin-prefetch",
            });
          });
        }
        if (typeof onGridVisibleMetadataReady === "function") {
          onGridVisibleMetadataReady({
            loadId,
            total,
            firstBatchCount: firstPics.length,
            visibleStart: visibleStart.value,
            visibleEnd: prefetchEnd,
          });
        }

        const tailOutcome = await tailReqPromise;
        if (!isCurrent()) return;
        // A failed tail leaves the end of the grid as placeholders the
        // background stream never reaches; it fails the load like any other
        // batch rather than passing as a shorter grid.
        if (tailOutcome.error) throw tailOutcome.error;
        const lastResTimed = tailOutcome.result;
        fetchPhaseTimings.tailBatchMs = lastResTimed?.elapsedMs ?? null;
        if (lastResTimed?.body) {
          // The tail is the end of the list in whichever state the server
          // read it: a delete above it while it was out shortens the grid and
          // the server's list alike, and an answer read before the delete
          // committed still ends at the list's end. So it fills the grid's
          // last cells, less any picture taken out since.
          // Nor a row the grid holds already (the viewport's rows, read
          // separately, can reach into the tail).
          const inGrid = new Set();
          for (const img of allGridImages.value) {
            if (img?.id != null) inGrid.add(String(img.id));
          }
          const lastPics = (lastResTimed?.body?.pictures ?? []).filter(
            (pic) => notRemoved(pic) && !inGrid.has(String(pic?.id)),
          );
          splicePictures(
            lastPics,
            Math.max(0, allGridImages.value.length - lastPics.length),
          );
        }

        lastFetchSuccess.value = { key: fetchKey, at: Date.now() };

        // Sync lastFetchedGridImages after the initial batches so that
        // removeImagesById/rebuildGridImagesFromLastFetch works correctly even
        // if a delete happens before background streaming finishes.
        lastFetchedGridImages.value = allGridImages.value.filter(
          (img) => img && img.id != null,
        );
        loadFoldsStacks.value = foldsStacks;
        const streamFetched = lastFetchedGridImages.value;

        // 4. Background stream: fill the placeholders left between the first
        // batch and the tail, one request at a time (in parallel they contend
        // in the database with the first batch, which is on screen).
        //
        // Each request is worked out from the grid as it is when it goes, and
        // each answer is placed by picture id, not by the offset it was asked
        // at. A delete while a request is out takes rows from the grid and
        // moves every cell after them up, and the server may answer from the
        // list before the delete or after it. So each request starts one row
        // early, at the last loaded row before the run (the anchor), and its
        // rows go into the grid from the one after the anchor, wherever the
        // anchor sits in the answer. An answer without it is asked again from
        // further back. The loaded row after the run ends it: an answer that
        // reaches that row before the run is full shows that the cells left
        // held pictures that are gone (one removed before its row arrived), and
        // they go, as do the cells an exhausted list cannot fill.
        const placedIds = new Set();
        for (const img of allGridImages.value) {
          if (img?.id != null) placedIds.add(String(img.id));
        }
        let fillFrom = FIRST_BATCH;
        let lookback = 1;
        while (true) {
          if (!isCurrent()) return;
          const askGrid = allGridImages.value;
          const ask = findGap(askGrid, fillFrom, BG_BATCH + 1);
          if (!ask) break;
          const back = Math.min(lookback, ask.start);
          const reachesNext =
            ask.end < askGrid.length && askGrid[ask.end]?.id != null ? 1 : 0;
          const limit = Math.min(
            BG_BATCH,
            back + (ask.end - ask.start) + reachesNext,
          );
          const bgResTimed = await timeRequest(
            streamPictures(streamQuery, {
              offset: ask.start - back,
              batchLimit: limit,
              signal,
            }),
          );
          if (!isCurrent()) return;
          const bgNetworkElapsedMs = bgResTimed.elapsedMs;
          const bgUiStartedAt = getNowMs();
          const bgPics = bgResTimed?.body?.pictures ?? [];
          const grid = allGridImages.value;
          const gap = findGap(grid, fillFrom, bgPics.length + 1);
          if (!gap) break;
          // Where the answer's new rows begin: after the anchor, or after the
          // nearest loaded row before the run that the answer holds.
          const answerAt = new Map();
          bgPics.forEach((pic, i) => answerAt.set(String(pic?.id), i));
          let from = gap.start === 0 && ask.start === back ? 0 : -1;
          const lowest = Math.max(0, gap.start - back);
          for (let k = gap.start - 1; from < 0 && k >= lowest; k -= 1) {
            const at = answerAt.get(String(grid[k].id));
            if (at !== undefined) from = at + 1;
          }
          if (from < 0) {
            if (gap.start > 0 && (back >= gap.start || back * 2 > BG_BATCH / 2)) {
              console.error(
                "[useGridFetch] A background batch never lined up with the grid; the rest of the view is not loaded.",
                { offset: ask.start - back, limit, gapStart: gap.start },
              );
              break;
            }
            lookback = back * 2 || 1;
            continue;
          }
          lookback = 1;
          const nextId =
            gap.end < grid.length && grid[gap.end]?.id != null
              ? String(grid[gap.end].id)
              : null;
          const bgOffset = gap.start;
          const bgRows = [];
          let reachedNext = false;
          let w = gap.start;
          for (let i = from; i < bgPics.length && w < gap.end; i += 1) {
            const id = String(bgPics[i]?.id);
            if (id === nextId) {
              reachedNext = true;
              break;
            }
            if (placedIds.has(id) || removedIds.has(id)) continue;
            bgRows.push(writeRow(grid, w, bgPics[i]));
            placedIds.add(id);
            w += 1;
          }
          fillFrom = w;
          const listEnded = bgPics.length < limit;
          if (w < gap.end && (reachedNext || listEnded || w === gap.start)) {
            // The run cannot fill any further. Its end, which `gap` may not
            // have looked as far as when the list has run out.
            let runEnd = gap.end;
            while (runEnd < grid.length && grid[runEnd]?.id == null) runEnd += 1;
            if (removedIds.size === 0) {
              console.warn(
                "[useGridFetch] The stream holds fewer rows than the count; dropping",
                runEnd - w,
                "placeholder cells.",
              );
            }
            dropPlaceholders(w, runEnd);
          }
          updateVisibleThumbnails();
          // Keep lastFetchedGridImages current so deletes during streaming work.
          // While both lists are still the ones this load wrote, the batch's
          // rows go in ahead of the early tail's, the only rows already past
          // them. Once anything else has replaced either list (a delete, a
          // re-score, a stack toggle), the whole grid is read again instead.
          if (
            allGridImages.value === streamGrid &&
            lastFetchedGridImages.value === streamFetched
          ) {
            let at = streamFetched.length;
            while (at > 0 && streamFetched[at - 1].idx >= bgOffset) at -= 1;
            streamFetched.splice(
              at,
              0,
              ...bgRows.filter((img) => img.id != null),
            );
          } else {
            lastFetchedGridImages.value = allGridImages.value.filter(
              (img) => img && img.id != null,
            );
          }
          await nextTick();
          const bgUiElapsedMs = Math.max(0, getNowMs() - bgUiStartedAt);
          const bgElapsedMs = bgNetworkElapsedMs + bgUiElapsedMs;
          fetchPhaseTimings.backgroundTotalMs += bgElapsedMs;
          fetchPhaseTimings.backgroundNetworkTotalMs += bgNetworkElapsedMs;
          fetchPhaseTimings.backgroundUiTotalMs += bgUiElapsedMs;
          fetchPhaseTimings.backgroundSlowestBatchMs = Math.max(
            fetchPhaseTimings.backgroundSlowestBatchMs,
            bgElapsedMs,
          );
          fetchPhaseTimings.backgroundSlowestNetworkBatchMs = Math.max(
            fetchPhaseTimings.backgroundSlowestNetworkBatchMs,
            bgNetworkElapsedMs,
          );
          fetchPhaseTimings.backgroundSlowestUiBatchMs = Math.max(
            fetchPhaseTimings.backgroundSlowestUiBatchMs,
            bgUiElapsedMs,
          );
          fetchPhaseTimings.backgroundBatchCount += 1;
        }

        // Sync lastFetchedGridImages so that removeImagesById/rebuildGridImagesFromLastFetch
        // works correctly after a delete during streaming.
        const postProcessStartedAt = getNowMs();
        lastFetchedGridImages.value = allGridImages.value.filter(
          (img) => img && img.id != null,
        );
        // Safety net: trim trailing placeholders the stream never filled.  If the
        // count and stream queries ever drift again (stream yields fewer rows than
        // COUNT(*)), the surplus tail cells would sit as permanent spinners.  Only
        // the contiguous trailing run of id-less cells is dropped - mid-grid holes
        // are left alone.
        const _finalGrid = allGridImages.value;
        let _filledEnd = _finalGrid.length;
        while (_filledEnd > 0 && _finalGrid[_filledEnd - 1]?.id == null) {
          _filledEnd -= 1;
        }
        if (_filledEnd < _finalGrid.length) {
          console.warn(
            "[ImageGrid.vue] Stream returned fewer rows than the count; trimming",
            _finalGrid.length - _filledEnd,
            "trailing placeholder cells.",
          );
          allGridImages.value = _finalGrid.slice(0, _filledEnd);
          if (visibleEnd.value > _filledEnd) {
            visibleEnd.value = _filledEnd;
          }
        }
        fetchPhaseTimings.postProcessMs = Math.max(
          0,
          getNowMs() - postProcessStartedAt,
        );
        fetchSucceeded = true;
        return;
      }

      if (!isCurrent()) {
        if (isSortedFetch && options?.showProgress === true)
          completeSmartScoreProgress(loadId, 0, false);
        return;
      }
      lastFetchedGridImages.value = Array.isArray(images) ? images.slice() : [];
      loadFoldsStacks.value = foldsStacks;
      syncExpandAllStacksFromFetchedImages();
      images = collapseStackImages(images);
      const shouldHighlight = highlightNextFetch.value && hasLoadedOnce.value;
      const nextIdSet = new Set(
        Array.isArray(images)
          ? images
              .map((img) => getPictureId(img?.id))
              .filter((id) => id !== null)
          : [],
      );
      if (shouldHighlight) {
        const newIds = [];
        nextIdSet.forEach((id) => {
          if (!previousImageIds.has(id)) {
            newIds.push(id);
          }
        });
        if (newIds.length) {
          triggerNewImageHighlight(newIds);
        }
      }
      previousImageIds.clear();
      nextIdSet.forEach((id) => previousImageIds.add(id));
      highlightNextFetch.value = false;
      hasLoadedOnce.value = true;
      const newImages = mapGridImages(images);
      resetThumbnailState();
      if (lightboxOpen.value) {
        // Don't replace allGridImages while the lightbox is open - the filmstrip
        // and prev/next navigation read from it directly. Store the fetched
        // result and apply it once the lightbox closes.
        pendingGridImages.value = newImages;
        pendingLightboxGridRefresh.value = true;
      } else {
        allGridImages.value = newImages;
      }
      // When the shared-only filter is active every returned image is shared by
      // definition. Pre-seed sharedPictureIds immediately so badges appear
      // without waiting for the async batch-check round trip.
      if (filterStore.sharedOnlyFilter && !isReadOnly.value) {
        const next = new Set(sharedPictureIds.value);
        for (const img of newImages) {
          if (img.id) next.add(img.id);
        }
        sharedPictureIds.value = next;
      }
      if (isSetOverlapView.value) {
        totalCurrentCategoryCount.value = newImages.length;
      }
      const cols = gridStore.columns || 1;
      // Compute window count from actual viewport capacity so visibleEnd covers
      // all initially visible items even when the viewport shows more than VIEW_WINDOW.
      // Derive cell height from clientWidth/cols (square thumbnails) rather than
      // rowHeight?.value, which may still hold the initial thumbnailSize estimate
      // when this runs (DOM measurement via updateRowHeightFromGrid is async).
      const _slowViewW = scrollWrapper.value?.clientWidth || 0;
      const _slowViewH = scrollWrapper.value?.clientHeight || 0;
      const _effectiveRowHeight1 =
        _slowViewW > 0
          ? Math.round(_slowViewW / cols) + (gridStore.compactMode ? 0 : 24)
          : rowHeight?.value > 0
            ? rowHeight.value
            : Math.round(
                Math.min(384, Math.max(128, gridStore.thumbnailSize || 128)) +
                  (gridStore.compactMode ? 0 : 24),
              );
      const _viewportItemCount1 =
        _slowViewH > 0 && _effectiveRowHeight1 > 0
          ? Math.ceil(_slowViewH / _effectiveRowHeight1) * cols
          : 0;
      const windowCount = Math.max(
        cols,
        _viewportItemCount1 || divisibleViewWindow.value || cols,
      );
      if (!fetchStartedWithPreserveScroll) {
        // Normal (non-preserve) fetch: jump to top so thumbnails load from index 0.
        visibleStart.value = 0;
        visibleEnd.value = Math.min(newImages.length, windowCount);
      } else {
        // Scroll-preserving fetch: keep visibleStart/End as-is so
        // updateVisibleThumbnails loads the range the user is actually viewing.
        visibleEnd.value = Math.min(visibleEnd.value, newImages.length);
        if (visibleStart.value > visibleEnd.value)
          visibleStart.value = Math.max(0, visibleEnd.value - 1);
      }
      if (initialRender.value) {
        const prefetchEnd = Math.min(
          newImages.length,
          visibleEnd.value + divisibleViewWindow.value,
        );
        fetchThumbnailsBatch(visibleStart.value, prefetchEnd);
      }
      await refreshExpandedStacksAfterFetch();
      // Superseded while the stacks refreshed: recording this view's success
      // now would debounce a quick return to it while the grid shows another.
      if (!isCurrent()) {
        if (isSortedFetch && options?.showProgress === true)
          completeSmartScoreProgress(loadId, 0, false);
        return;
      }
      requestAnimationFrame(() => {
        if (initialRender.value) {
          initialRender.value = false;
          updateVisibleThumbnails();
        }
      });
      lastFetchSuccess.value = { key: fetchKey, at: Date.now() };
      if (isSortedFetch) {
        const elapsedMs = Math.max(0, getNowMs() - sortedFetchStartedAt);
        completeSmartScoreProgress(loadId, elapsedMs, true);
      }
      fetchSucceeded = true;
    } catch (e) {
      // A superseded load fails silently: a cancelled read rejects here, and
      // its error belongs to a view that is no longer on screen.
      if (!isCurrent()) {
        if (isSortedFetch && options?.showProgress === true)
          completeSmartScoreProgress(loadId, 0, false);
        return;
      }
      fetchError = e;
      console.warn("[useGridFetch] Grid fetch failed", { fetchKey, fetchMode }, e);
      imagesError.value = e.message || String(e);
      // Don't wipe the grid on a transient error while the lightbox is open -
      // the user would see the grid flash empty behind the lightbox.
      if (!lightboxOpen.value) {
        allGridImages.value = [];
      }
      lastFetchError.value = { key: fetchKey, at: Date.now() };
      if (isSortedFetch) {
        const elapsedMs = Math.max(0, getNowMs() - sortedFetchStartedAt);
        completeSmartScoreProgress(loadId, elapsedMs, false);
      }
    } finally {
      // The stream's batches are all in, or will never land: stop tracking
      // removals for it, unless a newer load has put its own set there.
      if (loadRemovedIds && streamRemovedIds === loadRemovedIds) {
        streamRemovedIds = null;
      }
      // However the load ends it lets go of its place; only a load that runs
      // to the end below puts the view back there. A place that is no longer
      // pending was dropped (a user scroll, a newer load, a refresh that does
      // not keep the scroll) or already put back by the load it was shared
      // with, and is not this load's to restore.
      if (record.place && pendingScrollTop.value === record.place) {
        placeToRestore = record.place;
        pendingScrollTop.value = null;
      }
      if (typeof onGridFetchDone === "function") {
        onGridFetchDone({
          loadId,
          fetchMode,
          success: fetchSucceeded,
          elapsedMs: Math.max(0, getNowMs() - fetchStartedAt),
          resultCount: Array.isArray(allGridImages.value)
            ? allGridImages.value.length
            : 0,
          ...fetchPhaseTimings,
        });
      }
      if (loadId === gridLoadEpoch.value) {
        imagesLoading.value = false;
        gridReady.value = true;
      }
    }
    if (!initialRender.value) {
      updateVisibleThumbnails();
    }
    if (placeToRestore && scrollWrapper.value) {
      const targetTop = placeToRestore.top;
      nextTick(() => {
        if (!scrollWrapper.value) return;
        const maxScroll =
          scrollWrapper.value.scrollHeight - scrollWrapper.value.clientHeight;
        const clamped = Math.max(0, Math.min(targetTop, maxScroll));
        scrollWrapper.value.scrollTop = clamped;
        updateVisibleThumbnails();
      });
    }
    return { success: fetchSucceeded, fetchMode, error: fetchError };
  }

  async function fetchAllPicturesCount() {
    try {
      const data = await getCharacterSummary(
        ALL_PICTURES_ID,
        userPrefsStore.applyTagFilter ? { apply_tag_filter: true } : undefined,
      );
      totalAllPicturesCount.value = Number(data.image_count) || 0;
      totalAllPicturesCountLoaded.value = true;
    } catch (e) {
      console.warn("[ImageGrid.vue] Failed to fetch all pictures count:", e);
    }

    try {
      // The scoped count comes from one of two summary resources; the ladder
      // below picks which one and under what id, then a single call runs it.
      let summaryKind = "character";
      let summaryId = ALL_PICTURES_ID;
      let summaryProjectId = null;
      const selectedCharacter = String(selectionStore.selectedCharacter ?? "");
      if (isSetOverlapView.value) {
        totalCurrentCategoryCount.value =
          Number(allGridImages.value.length) || 0;
        return;
      }
      const selectedSetId = primarySelectedSetId.value;
      if (
        selectedSetId !== null &&
        selectedSetId !== undefined &&
        String(selectedSetId) !== ""
      ) {
        // One shared, in-flight-de-duplicated read of the set list: this is the
        // same request the sidebar makes on the same triggers, and the count
        // shown here must be the fresh one, so it forces a revalidation rather
        // than reading whatever happens to be cached.
        const setList = await useEntityListsStore().refresh("sets");
        const selectedSetNumericId = Number(selectedSetId);
        const selectedSet = Array.isArray(setList)
          ? setList.find((item) => {
              const itemId = Number(item?.id);
              if (Number.isFinite(selectedSetNumericId)) {
                return (
                  Number.isFinite(itemId) && itemId === selectedSetNumericId
                );
              }
              return String(item?.id) === String(selectedSetId);
            })
          : null;
        totalCurrentCategoryCount.value =
          Number(selectedSet?.picture_count) || 0;
        return;
      }
      const inProjectView = projectStore.projectViewMode === "project";
      const activeProjectId =
        projectStore.selectedProjectId != null
          ? projectStore.selectedProjectId
          : "UNASSIGNED";
      if (selectedCharacter === String(ALL_PICTURES_ID)) {
        if (inProjectView) {
          summaryKind = "project";
          summaryId = activeProjectId;
        }
      } else if (selectedCharacter === String(UNASSIGNED_PICTURES_ID)) {
        summaryId = UNASSIGNED_PICTURES_ID;
        if (inProjectView) summaryProjectId = activeProjectId;
      } else if (selectedCharacter === String(SCRAPHEAP_PICTURES_ID)) {
        summaryId = SCRAPHEAP_PICTURES_ID;
      } else if (selectedCharacter && !hasSetSelection.value) {
        summaryId = selectedCharacter;
        if (inProjectView) summaryProjectId = activeProjectId;
      }

      const summaryParams = {};
      if (summaryProjectId != null) summaryParams.project_id = summaryProjectId;
      if (userPrefsStore.applyTagFilter) summaryParams.apply_tag_filter = true;
      const hasParams = Object.keys(summaryParams).length > 0;

      const scopedData =
        summaryKind === "project"
          ? await getProjectSummary(
              summaryId,
              hasParams ? summaryParams : undefined,
            )
          : await getCharacterSummary(
              summaryId,
              hasParams ? summaryParams : undefined,
            );
      totalCurrentCategoryCount.value = Number(scopedData.image_count) || 0;
    } catch (e) {
      console.warn("[ImageGrid.vue] Failed to fetch scoped category count:", e);
      totalCurrentCategoryCount.value = 0;
    }
  }

  const debouncedFetchAllGridImages = debounce(fetchAllGridImages, 1000);

  return {
    imagesLoading,
    imagesError,
    totalAllPicturesCount,
    totalAllPicturesCountLoaded,
    totalCurrentCategoryCount,
    gridReady,
    gridLoadEpoch,
    lastFetchKey,
    lastFetchError,
    lastFetchSuccess,
    loadFoldsStacks,
    smartScoreLoadingVisible,
    buildGridFetchKey,
    buildPictureIdsQueryParams,
    buildStreamSelectionSuffix,
    searchViewQueryParams,
    buildFilterCountBaseQuery,
    buildLikenessGroupQueryParams,
    currentFetchMode,
    readIdListForReconcile,
    fetchAllGridImages,
    fetchAllPicturesCount,
    debouncedFetchAllGridImages,
    noteRemovedDuringStream,
    recordScrollPlace,
  };
}
