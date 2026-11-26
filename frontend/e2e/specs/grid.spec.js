// The shared fixture (not raw @playwright/test) so this context gets the
// production-network block registered on the browser fixture (issue #1213).
import { crc32, deflateSync } from 'node:zlib'
import { test, expect } from '../fixtures/test.js'

// Reference spec: drives the real app against the seeded test-data/ backend
// through an authenticated owner session (minted in global-setup). Assertions
// are resilient to fixture pruning — they check "at least one" rather than an
// exact count, so changing the number of seed images won't break them.

test.describe('image grid', () => {
  test('renders seeded thumbnails after login', async ({ page }) => {
    await page.goto('/')
    const cards = page.locator('.thumbnail-card')
    await expect(cards.first()).toBeVisible({ timeout: 15_000 })
    await expect.poll(() => cards.count()).toBeGreaterThan(0)
  })

  test('opens and closes the image lightbox', async ({ page }) => {
    await page.goto('/')

    const lightbox = page.locator('.lightbox')
    // The grid re-renders on picture websocket events (the e2e backend's
    // thumbnail-upgrade job emits a steady stream of them), which can swap the
    // card out between the actionability check and the click, swallowing it.
    // Retry until the lightbox is actually up. This spec deliberately avoids
    // page objects (the retry is inline rather than borrowing Lightbox),
    // which is unrelated to importing `test` from the shared fixtures file
    // above — that import is only for the production-network block.
    await expect(async () => {
      await page.locator('.thumbnail-card').first().dblclick()
      await expect(lightbox).toBeVisible({ timeout: 2_000 })
    }).toPass({ timeout: 30_000 })

    // Close via the toolbar's way back to the grid.
    await lightbox.locator('.lightbox-back').click()
    await expect(lightbox).toBeHidden()
  })

  test('focuses the search field with the F shortcut', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.thumbnail-card').first()).toBeVisible({
      timeout: 15_000,
    })

    // Search is the toolbar field itself. The "F" shortcut widens it, drops
    // its list and puts the caret in it.
    await page.keyboard.press('f')

    const panel = page.locator('.stf-drop')
    await expect(panel).toBeVisible()
    await expect(page.locator('input.stf-input')).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(panel).toBeHidden()
  })
})

// The fixture is smaller than the render window (100 pictures each side), so
// the window would never move; this imports enough pictures to make it move,
// and purges them again afterwards.
const SEEDED = 500
// Far from import.spec.js's seeds, so a picture here is never a duplicate of
// one there (the importer de-duplicates by pixel hash).
const SEED_BASE = 910_000

/** A 48px PNG of noise unique to `seed`. */
function noisePng(seed) {
  const size = 48
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
    const out = Buffer.alloc(body.length + 8)
    out.writeUInt32BE(data.length, 0)
    body.copy(out, 4)
    out.writeUInt32BE(crc32(body), body.length + 4)
    return out
  }
  const header = Buffer.alloc(13)
  header.writeUInt32BE(size, 0)
  header.writeUInt32BE(size, 4)
  header[8] = 8 // bit depth
  header[9] = 2 // truecolour
  const stride = size * 3 + 1
  const raw = Buffer.alloc(stride * size)
  let state = (seed * 2654435761) >>> 0
  for (let i = 0; i < raw.length; i++) {
    if (i % stride === 0) continue // filter byte: none
    state = (Math.imul(state, 1103515245) + 12345) >>> 0
    raw[i] = state >>> 24
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

async function pictureIds(apiContext) {
  const res = await apiContext.get('/api/v1/pictures', {
    params: { fields: 'grid' },
  })
  expect(res.ok(), await res.text()).toBe(true)
  return new Set((await res.json()).map((p) => p.id))
}

/** Import `count` new pictures through one staging run of 100-file batches. */
async function importNoise(apiContext, count) {
  for (let done = 0; done < count; done += 100) {
    const n = Math.min(100, count - done)
    const open = await apiContext.post('/api/v1/pictures/import/staging', {
      data: { total_files: n },
    })
    expect(open.ok(), await open.text()).toBe(true)
    const { staging_id } = await open.json()
    const form = new FormData()
    for (let k = 0; k < n; k++) {
      const seed = SEED_BASE + done + k
      form.append(
        'file',
        new Blob([noisePng(seed)], { type: 'image/png' }),
        `grid-rows-${seed}.png`,
      )
      form.append('key', `k${seed}`)
    }
    const upload = await apiContext.post(
      `/api/v1/pictures/import/staging/${staging_id}/files`,
      { multipart: form },
    )
    expect(upload.ok(), await upload.text()).toBe(true)
    const commit = await apiContext.post(
      `/api/v1/pictures/import/staging/${staging_id}/commit`,
    )
    expect(commit.ok(), await commit.text()).toBe(true)
    await expect
      .poll(
        async () => {
          const res = await apiContext.get(
            `/api/v1/pictures/import/staging/${staging_id}/status`,
          )
          return (await res.json()).stage
        },
        { timeout: 60_000 },
      )
      .toBe('completed')
  }
}

/** Soft delete, then purge from the Scrapheap, exactly these pictures. */
async function purgePictures(apiContext, ids) {
  if (!ids.length) return
  const moved = await apiContext.delete('/api/v1/pictures', {
    data: { picture_ids: ids },
  })
  expect(moved.ok(), await moved.text()).toBe(true)
  const preview = await apiContext.post(
    '/api/v1/pictures/scrapheap/delete-preview',
    { data: { ids } },
  )
  expect(preview.ok(), await preview.text()).toBe(true)
  const { confirm_token } = await preview.json()
  const purge = await apiContext.delete('/api/v1/pictures/scrapheap', {
    data: { picture_ids: ids, confirm_token },
  })
  expect(purge.ok(), await purge.text()).toBe(true)
}

/**
 * Runs in the page. Scrolls the grid in short bursts and records, for every
 * frame pair, how far each card that stayed mounted moved in content
 * coordinates (its screen motion plus the scroll motion). A rigid grid
 * moves no card at all.
 */
async function scrollAndMeasure({ bursts, steps, stepPx, pauseMs }) {
  const sc = document.querySelector('.grid-scroll-wrapper')
  const raf = () => new Promise((r) => requestAnimationFrame(() => r()))
  const ids = new WeakMap()
  let next = 0
  const spacers = () => {
    const grid = sc.querySelector('.image-grid')
    const first = grid.firstElementChild
    const last = grid.lastElementChild
    return `${first.style.height}|${last.style.height}`
  }
  const snap = () => {
    const top = sc.getBoundingClientRect().top
    const cards = new Map()
    for (const el of sc.querySelectorAll('.image-card')) {
      if (!ids.has(el)) ids.set(el, next++)
      cards.set(ids.get(el), el.getBoundingClientRect().top - top)
    }
    return { st: sc.scrollTop, sp: spacers(), cards }
  }
  const frames = [snap()]
  for (let b = 0; b < bursts; b++) {
    for (let s = 0; s < steps; s++) {
      sc.scrollBy({ top: stepPx, behavior: 'instant' })
      await raf()
      frames.push(snap())
    }
    const end = performance.now() + pauseMs
    while (performance.now() < end) {
      await raf()
      frames.push(snap())
    }
  }
  let windowMoves = 0
  const shifts = []
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1]
    const z = frames[i]
    if (a.sp !== z.sp) windowMoves++
    let worst = 0
    for (const [id, y] of z.cards) {
      if (!a.cards.has(id)) continue
      worst = Math.max(worst, Math.abs(y - a.cards.get(id) + (z.st - a.st)))
    }
    if (worst > 0.01) shifts.push({ frame: i, scrollTop: z.st, px: worst })
  }
  return { windowMoves, scrolled: sc.scrollTop, shifts }
}

test.describe('square grid scroll', () => {
  // Importing, scrolling and purging is more than the default 30 s per test.
  test.describe.configure({ mode: 'serial', timeout: 120_000 })

  let seededIds = []
  let previousConfig

  test.beforeAll(async ({ apiContext }) => {
    const res = await apiContext.get('/api/v1/users/me/config')
    expect(res.ok()).toBe(true)
    const config = await res.json()
    previousConfig = {
      thumbnail_mode: config.thumbnail_mode ?? 'square',
      thumbnail_size_level: config.thumbnail_size_level,
      compact_mode: config.compact_mode ?? false,
    }
    const before = await pictureIds(apiContext)
    await importNoise(apiContext, SEEDED)
    seededIds = [...(await pictureIds(apiContext))].filter(
      (id) => !before.has(id),
    )
    expect(seededIds).toHaveLength(SEEDED)
  })

  test.afterAll(async ({ apiContext }) => {
    if (previousConfig) {
      const res = await apiContext.patch('/api/v1/users/me/config', {
        data: previousConfig,
      })
      expect(res.ok(), await res.text()).toBe(true)
    }
    await purgePictures(apiContext, seededIds)
  })

  test('square rows keep their offset while scrolling', async ({
    page,
    apiContext,
    grid,
  }) => {
    const res = await apiContext.patch('/api/v1/users/me/config', {
      data: {
        thumbnail_mode: 'square',
        compact_mode: false,
        thumbnail_size_level: 3,
      },
    })
    expect(res.ok(), await res.text()).toBe(true)
    await grid.goto()
    await grid.waitForThumbnailLoaded()

    const result = await page.evaluate(scrollAndMeasure, {
      bursts: 20,
      steps: 10,
      stepPx: 20,
      pauseMs: 120,
    })
    // Not vacuous: the scroll went far enough for the window to move.
    expect(result.scrolled).toBeGreaterThan(3000)
    expect(result.windowMoves).toBeGreaterThan(2)
    expect(result.shifts).toEqual([])
  })
})
