// media.js - Shared media/file helpers for PixlStash frontend

import { isOwnShareCopyDrop, outgoingDragIds } from "./shareDrag.js";
import { useNoticeStore } from "../stores/useNoticeStore";

export const PIL_IMAGE_EXTENSIONS = [
  "jpg",
  "jpeg",
  "png",
  "bmp",
  "gif",
  "tiff",
  "tif",
  "webp",
  "ppm",
  "pgm",
  "pbm",
  "pnm",
  "ico",
  "icns",
  "svg",
  "dds",
  "msp",
  "pcx",
  "xbm",
  "im",
  "fli",
  "flc",
  "eps",
  "psd",
  "pdf",
  "jp2",
  "j2k",
  "jpf",
  "jpx",
  "j2c",
  "jpc",
  "tga",
  "ras",
  "sgi",
  "rgb",
  "rgba",
  "bw",
  "exr",
  "hdr",
  "pic",
  "pict",
  "pct",
  "cur",
  "emf",
  "wmf",
  "heic",
  "heif",
  "avif",
];

export const VIDEO_EXTENSIONS = [
  "mp4",
  "avi",
  "mov",
  "webm",
  "mkv",
  "flv",
  "wmv",
  "m4v",
];

/**
 * What the picture importer will actually take.
 *
 * NOT `PIL_IMAGE_EXTENSIONS` + `VIDEO_EXTENSIONS`: those two say what the app
 * can *display*, and the import endpoint takes a much shorter list. Filtering a
 * drop against the display lists let a `.psd`, a `.pdf` or a `.wmv` upload in
 * full before the backend skipped it as unsupported and the commit came back
 * "No staged files to import" - a gigabyte spent to reach an error the name
 * already gave away.
 *
 * Mirrors `STAGING_ALLOWED_MEDIA_EXTS` in
 * `pixlstash/routes/pictures/_import.py`, which is the one the server enforces.
 * `tests/test_architecture_guardrails.py::test_frontend_import_extensions_match_the_staging_allowlist`
 * fails the build if the two drift apart.
 */
export const IMPORT_MEDIA_EXTENSIONS = [
  "jpg",
  "jpeg",
  "png",
  "webp",
  "gif",
  "bmp",
  "tiff",
  "tif",
  "heic",
  "heif",
  "avif",
  "mp4",
  "webm",
  "mov",
  "avi",
  "mkv",
];

export const ARCHIVE_EXTENSIONS = ["zip"];

export const CAPTION_EXTENSIONS = ["txt"];

/**
 * The `accept` for every `<input type="file">` that feeds the picture importer.
 *
 * One constant rather than a literal per input, because it has to agree with
 * `isSupportedImportFile` and there is now more than one place to forget: the
 * empty-library card shipped with `image/*,video/*` alone, which hid zip and
 * caption imports behind an "All Files" the picker only offers if you look for
 * it. `accept` remains advisory - the grid still filters what comes back - so
 * this decides what the picker *offers*, not what the app will take.
 *
 * Both spellings of each non-media type are listed because browsers disagree
 * about which one they match on: Windows reports a zip as
 * `application/x-zip-compressed`, and a file with no registered handler
 * reports no type at all, leaving only the extension.
 */
export const IMPORT_FILE_ACCEPT =
  "image/*,video/*,.zip,application/zip,application/x-zip-compressed,.txt,text/plain";

/** What the model shelf catalogues - `pixlstash/routes/model_files.py`. */
const MODEL_FILE_EXTENSION = ".safetensors";

export function isSupportedImageFile(file) {
  const filename = typeof file === "string" ? file : file?.name || "";
  const ext = filename.split(".").pop().toLowerCase();
  return PIL_IMAGE_EXTENSIONS.includes(ext);
}

export function isSupportedVideoFile(file) {
  const filename = typeof file === "string" ? file : file.name || "";

  const ext = filename.split(".").pop().toLowerCase();
  return VIDEO_EXTENSIONS.includes(ext);
}

function isSupportedArchiveFile(file) {
  const filename = typeof file === "string" ? file : file.name || "";
  const ext = filename.split(".").pop().toLowerCase();
  return ARCHIVE_EXTENSIONS.includes(ext);
}

function isImportableMediaFile(file) {
  const filename = typeof file === "string" ? file : file?.name || "";
  const ext = filename.split(".").pop().toLowerCase();
  return IMPORT_MEDIA_EXTENSIONS.includes(ext);
}

function isSupportedCaptionFile(file) {
  const filename = typeof file === "string" ? file : file?.name || "";
  const lastDot = filename.lastIndexOf(".");
  const ext =
    lastDot > 0 && lastDot < filename.length - 1
      ? filename.slice(lastDot + 1).toLowerCase()
      : "";
  return CAPTION_EXTENSIONS.includes(ext);
}

/** A file the model shelf catalogues. The route refuses anything else. */
export function isModelFile(file) {
  const filename = typeof file === "string" ? file : file?.name || "";
  return filename.toLowerCase().endsWith(MODEL_FILE_EXTENSION);
}

/** A file the Workflows view adds: a ComfyUI workflow is a `.json`. */
export function isWorkflowFile(file) {
  const filename = typeof file === "string" ? file : file?.name || "";
  return filename.toLowerCase().endsWith(".json");
}

export function isSupportedImportFile(file) {
  return (
    isImportableMediaFile(file) ||
    isSupportedArchiveFile(file) ||
    isSupportedCaptionFile(file)
  );
}

// The path each file had in the drop (`/holiday/beach/x.png`), for files a
// folder walk found. Kept beside the File rather than on it: a File's own
// properties are read-only, and only the walk knows where the file sat.
const _dropPaths = new WeakMap();

// How many unreadable items a report names; the count covers all of them.
const _UNREADABLE_EXAMPLES = 5;

/**
 * The path a dropped file had inside the drop, such as `/holiday/x.png`, or
 * `""` for a file that did not come from a folder walk (a file picker, or a
 * browser without the entries API). The import sends it with the file so a
 * caption only tags the picture in its own folder.
 *
 * @param {File} file
 * @returns {string}
 */
export function importRelativePath(file) {
  return (file && _dropPaths.get(file)) || "";
}

function _fileDedupKey(file) {
  // A walked file is told apart by its place in the drop, so `a/x.png` and
  // `b/x.png` are both kept. The leading `/` is dropped so a top-level file
  // the drop also lists as `dataTransfer.files` still counts once.
  const name = importRelativePath(file).replace(/^\/+/, "") || file?.name || "";
  const size = Number.isFinite(file?.size) ? file.size : 0;
  const lastModified = Number.isFinite(file?.lastModified)
    ? file.lastModified
    : 0;
  return `${name}::${size}::${lastModified}`;
}

function _addIfSupportedFile(file, uniqueMap, accept) {
  if (!file || !accept(file)) return;
  const key = _fileDedupKey(file);
  if (!uniqueMap.has(key)) {
    uniqueMap.set(key, file);
  }
}

/**
 * Note one item the walk could not read, and log why.
 *
 * @param {{count: number, examples: string[]}} unreadable - the drop's tally.
 * @param {object} entry - the file or folder entry that failed.
 * @param {*} error - what the browser reported.
 */
function _noteUnreadable(unreadable, entry, error) {
  const path = entry?.fullPath || entry?.name || "";
  unreadable.count += 1;
  if (unreadable.examples.length < _UNREADABLE_EXAMPLES) {
    unreadable.examples.push(path);
  }
  console.warn("Skipping a dropped item that could not be read:", path, error);
}

/**
 * Every entry a folder lists, batch by batch. Settles on an error too, with
 * the entries listed before it, so one unreadable folder cannot stall the drop.
 *
 * @returns {Promise<{entries: object[], error: *}>} `error` is null when the
 *   whole listing was read.
 */
function _readAllWebkitDirectoryEntries(reader) {
  return new Promise((resolve) => {
    const entries = [];
    const readBatch = () => {
      try {
        reader.readEntries(
          (batch) => {
            if (!batch || batch.length === 0) {
              resolve({ entries, error: null });
              return;
            }
            entries.push(...batch);
            readBatch();
          },
          (error) => resolve({ entries, error }),
        );
      } catch (error) {
        resolve({ entries, error });
      }
    };
    readBatch();
  });
}

async function _collectFromWebkitEntry(entry, uniqueMap, accept, unreadable) {
  if (!entry) return;
  if (entry.isFile) {
    await new Promise((resolve) => {
      entry.file(
        (file) => {
          if (entry.fullPath) _dropPaths.set(file, entry.fullPath);
          _addIfSupportedFile(file, uniqueMap, accept);
          resolve();
        },
        (error) => {
          _noteUnreadable(unreadable, entry, error);
          resolve();
        },
      );
    });
    return;
  }
  if (!entry.isDirectory) return;
  let listing;
  try {
    listing = await _readAllWebkitDirectoryEntries(entry.createReader());
  } catch (error) {
    _noteUnreadable(unreadable, entry, error);
    return;
  }
  // A folder whose listing failed part-way still gives up what it listed.
  if (listing.error) _noteUnreadable(unreadable, entry, listing.error);
  for (const child of listing.entries) {
    await _collectFromWebkitEntry(child, uniqueMap, accept, unreadable);
  }
}

// Drops already warned about. The window's drop handler and the element under
// the pointer can both walk one drop; the user hears about it once.
const _reportedDrops = new WeakSet();

/**
 * An `onUnreadable` for `extractSupportedImportFilesFromDataTransfer` that
 * shows one warning for the drop, however many handlers walk it.
 *
 * @param {DataTransfer} dataTransfer - the drop being walked.
 * @returns {(report: {count: number, examples: string[]}) => void}
 */
export function unreadableDropReporter(dataTransfer) {
  return ({ count }) => {
    if (dataTransfer && typeof dataTransfer === "object") {
      if (_reportedDrops.has(dataTransfer)) return;
      _reportedDrops.add(dataTransfer);
    }
    useNoticeStore().warning(
      count === 1
        ? "1 item could not be read and was skipped"
        : `${count.toLocaleString()} items could not be read and were skipped`,
      { key: "import-unreadable-items" },
    );
  };
}

/**
 * Every file in a drop that `accept` wants, directories walked.
 *
 * A file found by walking a folder carries its path in the drop, read with
 * `importRelativePath`.
 *
 * @param {DataTransfer} dataTransfer - the drop's payload.
 * @param {object} [options]
 * @param {(file: File) => boolean} [options.accept] - what to keep. Defaults to
 *   what the picture importer takes. A caller that ALSO wants something else
 *   out of the same drop must widen this and split the result itself rather
 *   than call twice: the walk is destructive on Safari, which empties the
 *   DataTransfer on the first `await`, so a second pass returns nothing.
 * @param {(report: {count: number, examples: string[]}) => void}
 *   [options.onUnreadable] - called once, after the walk, when any file or
 *   folder could not be read: how many, and up to five of their paths. The
 *   readable files are still returned. `unreadableDropReporter` makes one that
 *   warns the user.
 * @returns {Promise<File[]>}
 */
export async function extractSupportedImportFilesFromDataTransfer(
  dataTransfer,
  { accept = isSupportedImportFile, onUnreadable } = {},
) {
  if (!dataTransfer) return [];
  // A drag-out dropped back onto the window carries PixlStash's own share
  // copies. Every import path reads its files here, so this is the one place
  // that keeps them out of the library.
  if (isOwnShareCopyDrop(dataTransfer)) return [];

  const unique = new Map();
  const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];

  // IMPORTANT: Safari clears the DataTransfer object after the first `await`,
  // so all synchronous DataTransfer access must complete before any async work.
  // webkitGetAsEntry() is the primary method - it is synchronous, handles
  // directories, and is supported in all modern browsers (Chrome, Edge,
  // Firefox, Safari). getAsFile() serves as a per-item fallback.
  const webkitEntries = [];
  const fallbackFiles = [];

  for (const item of items) {
    if (!item || item.kind !== "file") continue;

    if (typeof item.webkitGetAsEntry === "function") {
      try {
        const entry = item.webkitGetAsEntry();
        if (entry) {
          webkitEntries.push(entry);
          continue;
        }
      } catch {
        // Fall through to getAsFile().
      }
    }

    if (typeof item.getAsFile === "function") {
      fallbackFiles.push(item.getAsFile());
    }
  }

  // Capture dataTransfer.files synchronously before any awaits (final fallback
  // for browsers that expose no items list at all).
  const directFiles = Array.from(dataTransfer.files || []);

  // --- All synchronous DataTransfer access is done. Now we can safely await.
  // ---

  const unreadable = { count: 0, examples: [] };
  for (const entry of webkitEntries) {
    await _collectFromWebkitEntry(entry, unique, accept, unreadable);
  }

  for (const file of fallbackFiles) {
    _addIfSupportedFile(file, unique, accept);
  }

  for (const file of directFiles) {
    _addIfSupportedFile(file, unique, accept);
  }

  if (unreadable.count > 0 && onUnreadable) onUnreadable(unreadable);
  return Array.from(unique.values());
}

export function MediaFormat(source) {
  if (!source) return "";
  if (typeof source === "string") {
    const trimmed = source.trim().toLowerCase();
    if (!trimmed) return "";
    const stripped = trimmed.split("?")[0].split("#")[0];
    if (!stripped) return "";
    const parts = stripped.split(".");
    return parts.length > 1 ? parts.pop() : stripped;
  }
  if (source.format) return MediaFormat(source.format);
  if (source.filename) return MediaFormat(source.filename);
  if (source.url) return MediaFormat(source.url);
  return "";
}

// Map a media extension (or a source with a derivable format) to a MIME type.
// Used for the `DownloadURL` drag-out hint so the OS file manager creates a
// correctly-typed file. Unknown formats fall back to a generic binary type,
// which still downloads correctly.
const MEDIA_MIME_TYPES = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  gif: "image/gif",
  webp: "image/webp",
  bmp: "image/bmp",
  tif: "image/tiff",
  tiff: "image/tiff",
  svg: "image/svg+xml",
  ico: "image/x-icon",
  cur: "image/x-icon",
  avif: "image/avif",
  heic: "image/heic",
  heif: "image/heif",
  mp4: "video/mp4",
  m4v: "video/mp4",
  webm: "video/webm",
  mov: "video/quicktime",
  avi: "video/x-msvideo",
  mkv: "video/x-matroska",
  flv: "video/x-flv",
  wmv: "video/x-ms-wmv",
};

export function mediaMimeType(source) {
  const ext = MediaFormat(source);
  if (!ext) return "application/octet-stream";
  return MEDIA_MIME_TYPES[ext] || "application/octet-stream";
}

// Sanitise a user-supplied name for use as a drag-out download filename. The
// `DownloadURL` dataTransfer hint is "<mime>:<filename>:<url>", so a name that
// carries path separators, a colon, or a newline can redirect or break the
// saved file. This matters in the Electron shell, which sanitises drag-out
// names less than a plain browser download does. Takes the basename (drops
// everything up to the last / or \), replaces control chars and ':' with '_',
// trims, and falls back to `fallback` when nothing usable remains.
export function safeDownloadName(name, fallback = "download") {
  const raw = typeof name === "string" ? name : "";
  // Basename: strip everything up to and including the last / or \.
  const slashIdx = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
  const base = slashIdx >= 0 ? raw.slice(slashIdx + 1) : raw;
  // Replace control chars (newlines, tabs, etc.) and the ':' separator with '_'.
  // eslint-disable-next-line no-control-regex
  const cleaned = base.replace(/[\u0000-\u001f\u007f:]/g, "_").trim();
  if (!cleaned) return fallback;
  // Cap at 255 chars (the common filesystem filename limit) so a pathological
  // multi-KB original_file_name can't produce an unbounded download name.
  // Keep a short real extension; otherwise hard-truncate.
  const MAX = 255;
  if (cleaned.length <= MAX) return cleaned;
  const dot = cleaned.lastIndexOf(".");
  const ext = dot > 0 && cleaned.length - dot <= 16 ? cleaned.slice(dot) : "";
  return cleaned.slice(0, MAX - ext.length) + ext;
}

export function getPictureId(id) {
  if (id === null || id === undefined) return null;
  return String(id);
}

/**
 * The `?v=` token for a picture's full-size media URL, or `""` for none.
 *
 * **Keyed on `orientation` alone, and that is the whole point.** An in-place
 * rotate rewrites the EXIF orientation tag and copies every pixel through, so
 * the content hash does not move - and the browser, which applies the tag
 * itself, goes on painting the bytes it already decoded. `pixel_sha` was the
 * token here and could not express the one edit that needs it.
 *
 * It has to be derivable from a *grid* record, because that is what the
 * lightbox opens on and what `prefetchFullImage` / `preloadAdjacentImages`
 * build their warm-up URLs from. All three must agree on the URL or the
 * lightbox reads a stale prefetch out of the memory cache, which is exactly
 * the bug this replaced: `pixel_sha` is not in the grid projection and only
 * arrives with `/pictures/{id}/metadata`, a beat after the `<img>` has loaded.
 * `orientation` is, so every builder produces the same URL from the first
 * paint and no pinning is needed. (See `Picture.grid_fields()`.)
 *
 * Orientation 1 - and a record that carries none - contributes nothing, so a
 * picture that has never been turned keeps the URL it has always had.
 *
 * @param {Object|null} image - a grid or metadata picture record.
 * @returns {string} `"o<orientation>"` or `""`.
 */
export function mediaVersion(image) {
  const orientation = Number(image?.orientation);
  return Number.isFinite(orientation) && orientation > 1
    ? `o${orientation}`
    : "";
}

/** Orientations that put the picture on its side - 90° either way. */
const QUARTER_TURNED_ORIENTATIONS = new Set([5, 6, 7, 8]);

/**
 * The aspect ratio a picture is DISPLAYED at, for justified packing.
 *
 * Two dimension pairs live on a picture record and only one of them is in
 * display space:
 *
 *   * `thumbnail_width`/`height` are the stored bitmap's, rendered from the
 *     EXIF-transposed decode - already turned, never turn them again;
 *   * `width`/`height` are the RAW stored ones and do **not** move when a
 *     picture is rotated in place (that rewrites one tag and copies every pixel
 *     byte through), so the quarter turns have to be applied here.
 *
 * The fallback is not a rare path for a turned picture: `apply_orientation`
 * NULLs the thumbnail dimensions to re-queue the bitmap, so every card sits on
 * `width`/`height` from the rotate until the regeneration sweep lands. Without
 * the swap the tile keeps its pre-rotate shape and then jumps when the
 * regenerated dimensions arrive - the reflow-then-repaint a rotate used to do
 * in two visible steps.
 *
 * @param {Object|null} image - a grid or metadata picture record.
 * @returns {number} width/height, or 1 when the record has no usable pair
 *   (unimported pictures, unprobed videos) so packing never divides by zero.
 */
export function displayedAspectRatio(image) {
  if (!image) return 1;
  const tw = Number(image.thumbnail_width);
  const th = Number(image.thumbnail_height);
  if (Number.isFinite(tw) && tw > 0 && Number.isFinite(th) && th > 0) {
    return tw / th;
  }
  const w = Number(image.width);
  const h = Number(image.height);
  if (Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0) {
    return QUARTER_TURNED_ORIENTATIONS.has(Number(image.orientation))
      ? h / w
      : w / h;
  }
  return 1;
}

export function buildMediaUrl({ backendUrl, image, format } = {}) {
  if (!backendUrl || !image || !image.id) return "";
  const ext = MediaFormat(format || image);
  // The extension selects the native-media route. Without it this URL points
  // at the JSON picture-detail resource, whose 200 response cannot be decoded
  // by <img>/<video> elements.
  if (!ext) return "";
  const version = mediaVersion(image);
  const cacheBuster = version ? `?v=${version}` : "";
  return `${backendUrl}/pictures/${image.id}.${ext}${cacheBuster}`;
}

export function getLightboxFormat(lightboxImage) {
  return MediaFormat(lightboxImage) || "png";
}

export function isFileDrag(dataTransfer) {
  if (!dataTransfer) return false;
  const types = dataTransfer.types ? Array.from(dataTransfer.types) : [];
  return types.includes("Files") || types.includes("application/x-moz-file");
}

/**
 * True when a drag originates inside the app (grid thumbnails dragged onto a
 * character / set / project), identified by our own `application/json` payload.
 *
 * This must be distinguished from an external OS file drop because the desktop
 * shell (Electron) populates `dataTransfer.files` with the dragged in-page image
 * as a real File - which the web does not - so a `files.length > 0` check alone
 * misreads an internal assign-drag as a file import. Only `types` is readable
 * during `dragover` (the payload itself is protected until `drop`), so key off
 * the type list, the same signal the drop handlers use.
 */
export function isInternalImageDrag(dataTransfer) {
  if (!dataTransfer) return false;
  const types = dataTransfer.types ? Array.from(dataTransfer.types) : [];
  // A desktop drag-out is an OS file drag with no JSON (utils/shareDrag.js).
  return (
    types.includes("application/json") || outgoingDragIds(dataTransfer) != null
  );
}

/**
 * The internal payload a drag carries: its JSON body, or for a desktop
 * drag-out (an OS file drag, utils/shareDrag.js) the pictures it is dragging.
 * Null when there is none, or the JSON does not parse.
 *
 * Every drop target reads its payload here, so the two kinds of picture drag
 * cannot be told apart by the targets that take them.
 */
export function readInternalDragPayload(dataTransfer) {
  if (!dataTransfer) return null;
  const raw = dataTransfer.getData?.("application/json");
  if (raw) {
    try {
      return JSON.parse(raw);
    } catch (err) {
      console.error("Could not parse the drag payload", { raw, err });
      return null;
    }
  }
  const ids = outgoingDragIds(dataTransfer);
  return ids ? { type: "image-ids", imageIds: ids } : null;
}

/**
 * Marker types that carry the *kind* of an internal drag payload.
 *
 * Every internal payload travels as `application/json`, whose body is protected
 * during `dragover` (getData() returns "" in Chrome and Firefox); only `types`
 * is readable. The discriminator therefore has to be the key, not a field in
 * the body, or a drop target cannot tell a picture drag from a face drag until
 * the drop has already happened.
 */
export const PICTURE_DRAG_MIME = "application/x-pixlstash-pictures";
export const FACE_DRAG_MIME = "application/x-pixlstash-faces";
/**
 * Payload `type` to its marker. A kind absent from this map gets no marker, so
 * no drop target accepts it - an unmapped payload must fail closed rather than
 * inherit the picture marker and be filed as a picture drag (issue #757 again,
 * one payload kind later).
 */
const DRAG_MARKERS = {
  "image-ids": PICTURE_DRAG_MIME,
  "face-bbox": FACE_DRAG_MIME,
};

/**
 * Write an internal drag payload: the JSON body every drop handler reads, plus
 * the marker type its kind is recognised by during dragover.
 */
export function setInternalDragPayload(dataTransfer, payload) {
  if (!dataTransfer || !payload) return;
  dataTransfer.setData("application/json", JSON.stringify(payload));
  const marker = DRAG_MARKERS[payload.type];
  if (!marker) {
    console.error(
      `Internal drag payload type "${payload.type}" has no marker in ` +
        "DRAG_MARKERS, so no drop target will accept it. Add one.",
    );
    return;
  }
  dataTransfer.setData(marker, payload.type);
}

/** True when the drag carries pictures (grid thumbnails, the open overlay). */
export function isPictureDrag(dataTransfer) {
  if (!dataTransfer) return false;
  const types = dataTransfer.types ? Array.from(dataTransfer.types) : [];
  return (
    types.includes(PICTURE_DRAG_MIME) || outgoingDragIds(dataTransfer) != null
  );
}

/**
 * The drop effect a target should set: `preferred` where the drag allows it,
 * else one it does allow. A dropEffect outside `effectAllowed` makes the
 * browser refuse the drop without a drop event, and a desktop drag-out
 * (utils/shareDrag.js) allows only copy and link, never move.
 */
export function dropEffectFor(dataTransfer, preferred = "move") {
  const allowed = String(dataTransfer?.effectAllowed || "all").toLowerCase();
  if (allowed === "all" || allowed === "uninitialized") return preferred;
  return (
    [preferred, "copy", "link", "move"].find((e) => allowed.includes(e)) ||
    "none"
  );
}

/** True when the drag carries face bounding boxes. */
export function isFaceDrag(dataTransfer) {
  if (!dataTransfer) return false;
  const types = dataTransfer.types ? Array.from(dataTransfer.types) : [];
  return types.includes(FACE_DRAG_MIME);
}

export function isVideo(img) {
  if (!img) return false;
  const format = MediaFormat(img);
  if (format) {
    return isSupportedVideoFile(`file.${format}`);
  }
  return isSupportedVideoFile(img.id || "");
}

/**
 * Drop repeated face or object detections: the same id, or for a row without
 * one the same box on the same frame. The lightbox and the grid's rail read
 * both lists through this.
 */
export function dedupeDetections(items) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const result = [];
  for (const item of items) {
    const id = item?.id;
    const bbox = Array.isArray(item?.bbox) ? item.bbox.join(",") : "";
    const frame = item?.frame_index ?? "";
    const key = id != null ? `id:${id}` : `bbox:${frame}:${bbox}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }
  return result;
}
