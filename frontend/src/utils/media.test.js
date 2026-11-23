import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useNoticeStore } from '../stores/useNoticeStore'
import {
  buildMediaUrl,
  displayedAspectRatio,
  dropEffectFor,
  extractSupportedImportFilesFromDataTransfer,
  importRelativePath,
  unreadableDropReporter,
  isFaceDrag,
  isFileDrag,
  isInternalImageDrag,
  isPictureDrag,
  setInternalDragPayload,
  isSupportedImportFile,
  ARCHIVE_EXTENSIONS,
  CAPTION_EXTENSIONS,
  FACE_DRAG_MIME,
  IMPORT_FILE_ACCEPT,
  PICTURE_DRAG_MIME,
} from './media.js'

// Minimal DataTransfer stand-in: only `types` (array) and `files` (array-like)
// are read by the drag predicates.
function dt({ types = [], files = [] } = {}) {
  return { types, files }
}

describe('isInternalImageDrag', () => {
  it('is true when the drag carries our application/json payload', () => {
    expect(isInternalImageDrag(dt({ types: ['application/json'] }))).toBe(true)
  })

  it('is false for an external OS file drag', () => {
    expect(isInternalImageDrag(dt({ types: ['Files'], files: [{}] }))).toBe(false)
  })

  // Regression: on the Electron desktop shell, dragging an in-page thumbnail
  // onto a character/set populates dataTransfer.files with the image as a real
  // File *in addition to* our marker. The marker must still win so the window
  // import handler doesn't import the picture instead of assigning it.
  it('is true even when the desktop shell also attaches the image as a file', () => {
    expect(
      isInternalImageDrag(dt({ types: ['application/json', 'Files'], files: [{}] })),
    ).toBe(true)
  })

  it('is false for null/empty data transfer', () => {
    expect(isInternalImageDrag(null)).toBe(false)
    expect(isInternalImageDrag(dt())).toBe(false)
  })
})

describe('isFileDrag', () => {
  it('detects an external file drag by type', () => {
    expect(isFileDrag(dt({ types: ['Files'] }))).toBe(true)
    expect(isFileDrag(dt({ types: ['application/x-moz-file'] }))).toBe(true)
  })

  it('is false for an internal-only drag', () => {
    expect(isFileDrag(dt({ types: ['application/json'] }))).toBe(false)
  })
})

// A drop target may only read `types` during dragover, so the payload kind has
// to be a key. Before this, a face drag and a picture drag were indistinguishable
// until the drop had already happened (issue #757).
describe('internal drag payload markers', () => {
  function writer() {
    const store = {}
    return {
      store,
      dataTransfer: {
        setData: (type, value) => {
          store[type] = value
        },
        get types() {
          return Object.keys(store)
        },
      },
    }
  }

  it('marks a picture payload so dragover can recognise it', () => {
    const { store, dataTransfer } = writer()
    setInternalDragPayload(dataTransfer, { type: 'image-ids', imageIds: [1] })

    expect(JSON.parse(store['application/json'])).toEqual({
      type: 'image-ids',
      imageIds: [1],
    })
    expect(isPictureDrag(dataTransfer)).toBe(true)
    expect(isFaceDrag(dataTransfer)).toBe(false)
    expect(isInternalImageDrag(dataTransfer)).toBe(true)
  })

  it('marks a face payload distinctly, despite it carrying imageIds too', () => {
    const { dataTransfer } = writer()
    setInternalDragPayload(dataTransfer, {
      type: 'face-bbox',
      faceIds: [9],
      imageIds: [1],
    })

    expect(isFaceDrag(dataTransfer)).toBe(true)
    expect(isPictureDrag(dataTransfer)).toBe(false)
    expect(isInternalImageDrag(dataTransfer)).toBe(true)
  })

  it('reports neither kind for an external file drag', () => {
    expect(isPictureDrag(dt({ types: ['Files'] }))).toBe(false)
    expect(isFaceDrag(dt({ types: ['Files'] }))).toBe(false)
    expect(isPictureDrag(null)).toBe(false)
    expect(isFaceDrag(null)).toBe(false)
  })

  it('keeps the two marker types apart', () => {
    expect(PICTURE_DRAG_MIME).not.toBe(FACE_DRAG_MIME)
  })

  it('leaves an unmapped payload kind unmarked rather than calling it a picture',
     () => {
       const {store, dataTransfer} = writer()
       const complained =
           vi.spyOn(console, 'error').mockImplementation(() => {})

       setInternalDragPayload(dataTransfer, {type: 'something-new', ids: [1]})

       expect(JSON.parse(store['application/json']).type).toBe('something-new')
       expect(isPictureDrag(dataTransfer)).toBe(false)
       expect(isFaceDrag(dataTransfer)).toBe(false)
       expect(complained).toHaveBeenCalled()
       complained.mockRestore()
     })
})

// A dropEffect the drag does not allow makes the browser drop nothing, silently.
describe('dropEffectFor', () => {
  it('keeps move where the drag allows it', () => {
    expect(dropEffectFor({ effectAllowed: 'move' })).toBe('move')
    expect(dropEffectFor({ effectAllowed: 'all' })).toBe('move')
    expect(dropEffectFor({ effectAllowed: 'uninitialized' })).toBe('move')
  })

  it('falls to copy for a desktop drag-out, which allows only copy and link', () => {
    expect(dropEffectFor({ effectAllowed: 'copyLink' })).toBe('copy')
  })

  it('never offers copy to a drag that only allows move', () => {
    expect(dropEffectFor({ effectAllowed: 'move' }, 'copy')).toBe('move')
  })

  it('is none when the drag allows nothing', () => {
    expect(dropEffectFor({ effectAllowed: 'none' })).toBe('none')
  })
})

describe('buildMediaUrl', () => {
  it('builds an extension-qualified native-media URL', () => {
    expect(
      buildMediaUrl({
        backendUrl: '/api/v1',
        image: { id: 7, format: 'PNG', pixel_sha: 'abc' },
      }),
    ).toBe('/api/v1/pictures/7.png')
  })

  // The buster is the EXIF orientation, not the content hash: an in-place
  // rotate copies every pixel through, so the hash cannot express the one edit
  // that needs busting - and `orientation` is in the grid projection, so the
  // lightbox and both full-image preloaders build the same URL from the same
  // record. A picture that has never been turned keeps its bare URL.
  it('busts the URL on the orientation, not the content hash', () => {
    expect(
      buildMediaUrl({
        backendUrl: '/api/v1',
        image: { id: 7, format: 'PNG', pixel_sha: 'abc', orientation: 6 },
      }),
    ).toBe('/api/v1/pictures/7.png?v=o6')
    expect(
      buildMediaUrl({
        backendUrl: '/api/v1',
        image: { id: 7, format: 'PNG', orientation: 1 },
      }),
    ).toBe('/api/v1/pictures/7.png')
  })

  it('does not turn an id-only placeholder into a JSON endpoint media URL', () => {
    expect(buildMediaUrl({ backendUrl: '/api/v1', image: { id: 7 } })).toBe('')
  })
})

describe('displayedAspectRatio', () => {
  it('takes the thumbnail bitmap as-is - it is already EXIF-transposed', () => {
    expect(
      displayedAspectRatio({
        thumbnail_width: 300,
        thumbnail_height: 200,
        width: 3000,
        height: 2000,
        orientation: 6,
      }),
    ).toBeCloseTo(1.5)
  })

  // The window an in-place rotate opens: apply_orientation NULLs the thumbnail
  // dimensions to re-queue the bitmap, so a turned card sits on the RAW
  // width/height - which do not swap - until the sweep lands. Unswapped here,
  // the tile keeps its pre-rotate shape and then jumps.
  it('swaps the raw dimensions for a quarter-turned picture', () => {
    expect(displayedAspectRatio({ width: 4000, height: 3000 })).toBeCloseTo(4 / 3)
    expect(
      displayedAspectRatio({ width: 4000, height: 3000, orientation: 1 }),
    ).toBeCloseTo(4 / 3)
    // 3 and 2 are the 180deg / mirrored-horizontal cases: still landscape.
    expect(
      displayedAspectRatio({ width: 4000, height: 3000, orientation: 3 }),
    ).toBeCloseTo(4 / 3)
    for (const orientation of [5, 6, 7, 8]) {
      expect(
        displayedAspectRatio({ width: 4000, height: 3000, orientation }),
      ).toBeCloseTo(3 / 4)
    }
  })

  it('falls back to square rather than dividing by zero', () => {
    expect(displayedAspectRatio(null)).toBe(1)
    expect(displayedAspectRatio({ width: 0, height: 0 })).toBe(1)
  })
})

describe('IMPORT_FILE_ACCEPT', () => {
  // The empty-library card shipped with `image/*,video/*` and nothing else, so
  // a zip or a caption file could only be reached through the picker's "All
  // Files" - an import route the app supports and its own dialog hid. These
  // assert the offer against the predicate rather than against a copy of the
  // string, so the two cannot drift apart silently again.

  it('offers every file extension the importer will actually take', () => {
    const offered = IMPORT_FILE_ACCEPT.split(',')
      .filter((entry) => entry.startsWith('.'))
      .map((entry) => entry.slice(1))

    expect(offered.length).toBeGreaterThan(0)
    for (const ext of offered) {
      expect(isSupportedImportFile({ name: `sample.${ext}` })).toBe(true)
    }
  })

  it('advertises every type the importer takes, the other direction', () => {
    // The check above is a subset check: it would stay green if a supported
    // type were dropped from the offer, which is the drift that actually
    // matters. So walk the predicate's own lists instead.
    //
    // Media is carried by the two wildcards rather than named extension by
    // extension, so those are asserted outright - removing either would hide
    // every picture and every video while leaving the subset check green.
    expect(IMPORT_FILE_ACCEPT).toContain('image/*')
    expect(IMPORT_FILE_ACCEPT).toContain('video/*')

    // Everything else matches neither wildcard and has to be named.
    for (const ext of [...ARCHIVE_EXTENSIONS, ...CAPTION_EXTENSIONS]) {
      expect(isSupportedImportFile({ name: `shoot.${ext}` })).toBe(true)
      expect(IMPORT_FILE_ACCEPT).toContain(`.${ext}`)
    }
  })
})

// A drop as the browser's entries API hands it over. A directory's reader
// answers with its children in one batch, then with an empty batch; `fail`
// makes a listing, or a file, report an error instead.
const MTIME = 1700000000000

function fileOf(name) {
  return new File(['x'], name, { lastModified: MTIME })
}

function fileEntry(fullPath, { fail = false } = {}) {
  const file = fileOf(fullPath.split('/').pop())
  return {
    isFile: true,
    isDirectory: false,
    name: file.name,
    fullPath,
    // Like a browser, an error goes to the error callback if there is one, and
    // nowhere at all if there is not.
    file: (ok, error) =>
      fail ? error?.(new DOMException('gone', 'NotFoundError')) : ok(file),
  }
}

function dirEntry(fullPath, children, { failAfter = null } = {}) {
  return {
    isFile: false,
    isDirectory: true,
    name: fullPath.split('/').pop(),
    fullPath,
    createReader: () => {
      let batches = 0
      return {
        readEntries(ok, error) {
          if (failAfter !== null && batches === failAfter) {
            error?.(new DOMException('denied', 'NotReadableError'))
            return
          }
          batches += 1
          ok(batches === 1 ? children : [])
        },
      }
    },
  }
}

// `files` lists the top level again, as browsers do: a folder as one opaque
// entry, a file as its own File object.
function entriesDrop(...entries) {
  return {
    types: ['Files'],
    files: entries.map((entry) =>
      entry.isFile ? fileOf(entry.name) : new File([], entry.name),
    ),
    items: entries.map((entry) => ({
      kind: 'file',
      webkitGetAsEntry: () => entry,
    })),
  }
}

const PENDING = 'still walking'

function settledWithin(promise, ms = 200) {
  return Promise.race([
    promise,
    new Promise((resolve) => setTimeout(() => resolve(PENDING), ms)),
  ])
}

describe('extractSupportedImportFilesFromDataTransfer: a dropped folder tree', () => {
  beforeEach(() => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps same-named files from two folders, each with its own path', async () => {
    const drop = entriesDrop(
      dirEntry('/a', [fileEntry('/a/x.png'), fileEntry('/a/x.txt')]),
      dirEntry('/b', [fileEntry('/b/x.png'), fileEntry('/b/x.txt')]),
    )

    const files = await extractSupportedImportFilesFromDataTransfer(drop)

    expect(files.map(importRelativePath)).toEqual([
      '/a/x.png',
      '/a/x.txt',
      '/b/x.png',
      '/b/x.txt',
    ])
  })

  it('counts a top-level file once, though the drop lists it twice', async () => {
    const drop = entriesDrop(fileEntry('/x.png'))

    const files = await extractSupportedImportFilesFromDataTransfer(drop)

    expect(files.map(importRelativePath)).toEqual(['/x.png'])
  })

  it('has no path for a file that did not come from a folder walk', () => {
    expect(importRelativePath(fileOf('x.png'))).toBe('')
  })

  it('finishes, and reports the folder, when a folder cannot be listed', async () => {
    const onUnreadable = vi.fn()
    const drop = entriesDrop(
      dirEntry('/locked', [fileEntry('/locked/x.png')], { failAfter: 0 }),
      fileEntry('/kept.png'),
    )

    const walk = extractSupportedImportFilesFromDataTransfer(drop, {
      onUnreadable,
    })
    const files = await settledWithin(walk)

    expect(files).not.toBe(PENDING)
    expect(files.map(importRelativePath)).toEqual(['/kept.png'])
    expect(onUnreadable).toHaveBeenCalledTimes(1)
    expect(onUnreadable).toHaveBeenCalledWith({
      count: 1,
      examples: ['/locked'],
    })
  })

  it('keeps what a folder listed before its listing failed', async () => {
    const onUnreadable = vi.fn()
    const drop = entriesDrop(
      dirEntry('/big', [fileEntry('/big/one.png')], { failAfter: 1 }),
    )

    const files = await settledWithin(
      extractSupportedImportFilesFromDataTransfer(drop, { onUnreadable }),
    )

    expect(files).not.toBe(PENDING)
    expect(files.map(importRelativePath)).toEqual(['/big/one.png'])
    expect(onUnreadable).toHaveBeenCalledWith({
      count: 1,
      examples: ['/big'],
    })
  })

  it('reports every file it could not read, once for the whole drop', async () => {
    const onUnreadable = vi.fn()
    const drop = entriesDrop(
      dirEntry('/a', [
        fileEntry('/a/gone-1.png', { fail: true }),
        fileEntry('/a/ok.png'),
        fileEntry('/a/gone-2.png', { fail: true }),
      ]),
    )

    const files = await extractSupportedImportFilesFromDataTransfer(drop, {
      onUnreadable,
    })

    expect(files.map(importRelativePath)).toEqual(['/a/ok.png'])
    expect(onUnreadable).toHaveBeenCalledTimes(1)
    expect(onUnreadable).toHaveBeenCalledWith({
      count: 2,
      examples: ['/a/gone-1.png', '/a/gone-2.png'],
    })
  })

  it('reports a folder whose reader cannot be opened', async () => {
    const onUnreadable = vi.fn()
    const broken = {
      ...dirEntry('/broken', []),
      createReader: () => {
        throw new DOMException('denied', 'SecurityError')
      },
    }

    await extractSupportedImportFilesFromDataTransfer(entriesDrop(broken), {
      onUnreadable,
    })

    expect(onUnreadable).toHaveBeenCalledWith({
      count: 1,
      examples: ['/broken'],
    })
  })

  it('names at most five examples however many items failed', async () => {
    const onUnreadable = vi.fn()
    const gone = Array.from({ length: 7 }, (_, i) =>
      fileEntry(`/a/gone-${i}.png`, { fail: true }),
    )

    await extractSupportedImportFilesFromDataTransfer(
      entriesDrop(dirEntry('/a', gone)),
      { onUnreadable },
    )

    const [{ count, examples }] = onUnreadable.mock.calls[0]
    expect(count).toBe(7)
    expect(examples).toEqual(gone.slice(0, 5).map((entry) => entry.fullPath))
  })

  it('says nothing when every item could be read', async () => {
    const onUnreadable = vi.fn()

    await extractSupportedImportFilesFromDataTransfer(
      entriesDrop(dirEntry('/a', [fileEntry('/a/x.png')])),
      { onUnreadable },
    )

    expect(onUnreadable).not.toHaveBeenCalled()
  })
})

describe('unreadableDropReporter', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('warns once, in the singular, for one item', () => {
    const notices = useNoticeStore()
    unreadableDropReporter({})({ count: 1, examples: ['/a'] })

    expect(notices.notices.map((n) => [n.level, n.text])).toEqual([
      ['warning', '1 item could not be read and was skipped'],
    ])
  })

  it('warns in the plural for several', () => {
    const notices = useNoticeStore()
    unreadableDropReporter({})({ count: 3, examples: [] })

    expect(notices.notices.at(-1).text).toBe(
      '3 items could not be read and were skipped',
    )
  })

  it('warns once per drop, however many handlers walk it', () => {
    const notices = useNoticeStore()
    const drop = {}
    unreadableDropReporter(drop)({ count: 2, examples: [] })
    unreadableDropReporter(drop)({ count: 2, examples: [] })

    expect(notices.notices).toHaveLength(1)
    // Not a coalesced "×2" card either: the second walk adds nothing.
    expect(notices.notices[0].count).toBe(1)
  })
})
