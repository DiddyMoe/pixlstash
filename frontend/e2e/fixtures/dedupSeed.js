// The duplicate queue's seed script, `e2e/seed_dedup_fixture.py`, for the
// specs that seed duplicates or open the queue.
//
// The e2e planner never runs a duplicate scan, so the scan the queue asks for
// on opening stays pending for the rest of the run. A pending scan is work a
// person asked for, and a snapshot restore refuses while one waits rather than
// lose it, so a spec that opens the queue as the owner hands the backend on
// with `clearDedupState`, the way the duplicate specs undo their seed.

import { execFileSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const SEED_SCRIPT = resolve(__dirname, '../seed_dedup_fixture.py')
const REPO_ROOT = resolve(__dirname, '../../..')

/** Run the seed script and parse its JSON report. */
export function runSeedScript(args) {
  const python = process.env.PIXLSTASH_PYTHON || 'python'
  const stdout = execFileSync(python, [SEED_SCRIPT, ...args], {
    // The backend is launched from `frontend/`, so the repo is not on
    // `sys.path` there. The seed imports the same pixlstash the server runs.
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
    encoding: 'utf8',
  })
  return JSON.parse(stdout)
}

/**
 * Remove the seeded pictures, the stacks their verdicts made and all dedup
 * state, the scans the queue asked for included.
 */
export function clearDedupState() {
  return runSeedScript(['--cleanup'])
}
