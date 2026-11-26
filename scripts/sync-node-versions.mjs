// Single source-of-truth version sync: mirror the version from pyproject.toml
// into the Node package manifests so the desktop app, the frontend bundle and
// their lockfiles all advertise the same version.
//
// Updates four files in total, for each of `electron/` and `frontend/`:
//   - package.json        -> .version
//   - package-lock.json   -> .version and .packages[""].version
//
// Dependency versions nested deeper in the lockfiles are never touched. The
// script is idempotent; run it from anywhere. Pass directory names to limit the
// targets (e.g. `node scripts/sync-node-versions.mjs electron`); with no args it
// syncs both. Mirrors what .github/workflows/windows-installer.yml does for the
// Inno Setup build.
//
// `--check` writes nothing: it names every manifest that disagrees with
// pyproject.toml and exits 1, so a gate can ask without editing the tree. Any
// other option is refused (exit 2) rather than read as a directory name.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..');
const pyprojectPath = join(repoRoot, 'pyproject.toml');

const args = process.argv.slice(2);
const check = args.includes('--check');
const unknown = args.filter((arg) => arg.startsWith('-') && arg !== '--check');
if (unknown.length) {
  console.error(
    `sync-node-versions: unknown option ${unknown.join(' ')} ` +
      '(usage: sync-node-versions.mjs [--check] [electron] [frontend])',
  );
  process.exit(2);
}

const pyproject = readFileSync(pyprojectPath, 'utf8');
// The version lives in the [project] table: `version = "1.6.0"`. Match the
// first top-level version assignment.
const match = pyproject.match(/^\s*version\s*=\s*"([^"]+)"/m);
if (!match) {
  console.error('sync-node-versions: could not find version in pyproject.toml');
  process.exit(1);
}
const pep440 = match[1];

// PEP 440 (Python) versions like "1.6.0.dev1" or "1.6.0rc1" aren't valid semver,
// which electron-builder (>=26) and electron-updater require. Translate the base
// release plus any pre-release (aN/bN/rcN), dev (.devN) and post (.postN) segments
// into their semver spelling, e.g. 1.6.0.dev1 -> 1.6.0-dev.1, 1.6.0rc1 -> 1.6.0-rc.1.
function pep440ToSemver(v) {
  const base = v.match(/^(\d+)\.(\d+)(?:\.(\d+))?(.*)$/);
  if (!base) return v;
  const [, major, minor, patch = '0'] = base;
  let rest = base[4] || '';
  const pre = [];
  const phase = { a: 'alpha', alpha: 'alpha', b: 'beta', beta: 'beta', rc: 'rc', c: 'rc', preview: 'rc', pre: 'rc' };
  let g;
  if ((g = rest.match(/^[._-]?(alpha|a|beta|b|preview|pre|rc|c)[._-]?(\d+)/i))) {
    pre.push(phase[g[1].toLowerCase()], g[2]);
    rest = rest.slice(g[0].length);
  }
  if ((g = rest.match(/^[._-]?dev[._-]?(\d+)/i))) { pre.push('dev', g[1]); rest = rest.slice(g[0].length); }
  if ((g = rest.match(/^[._-]?post[._-]?(\d+)/i))) { pre.push('post', g[1]); }
  return pre.length ? `${major}.${minor}.${patch}-${pre.join('.')}` : `${major}.${minor}.${patch}`;
}

const version = pep440ToSemver(pep440);
const fromNote = version !== pep440 ? ` (from ${pep440})` : '';

// Rewrite a single JSON file via a mutator, preserving 2-space indentation and a
// trailing newline; under --check the file is only compared. Returns whether
// anything changed (or, under --check, would have).
function updateJson(path, mutate) {
  let raw;
  try {
    raw = readFileSync(path, 'utf8');
  } catch (err) {
    console.error(`sync-node-versions: cannot read ${path}: ${err.message}`);
    process.exit(1);
  }
  const data = JSON.parse(raw);
  const changed = mutate(data);
  if (changed && !check) {
    writeFileSync(path, JSON.stringify(data, null, 2) + '\n');
  }
  return changed;
}

const targets = args.filter((arg) => !arg.startsWith('-'));
const dirs = targets.length ? targets : ['electron', 'frontend'];

let anyChange = false;
for (const dir of dirs) {
  const pkgPath = join(repoRoot, dir, 'package.json');
  const lockPath = join(repoRoot, dir, 'package-lock.json');

  const pkgChanged = updateJson(pkgPath, (pkg) => {
    if (pkg.version === version) return false;
    pkg.version = version;
    return true;
  });

  // Update the lockfile's own version in both places it appears (the root
  // metadata and the self-referential packages[""] entry). Leave every other
  // "version" (dependencies) alone.
  const lockChanged = updateJson(lockPath, (lock) => {
    let changed = false;
    if (lock.version !== version) { lock.version = version; changed = true; }
    if (lock.packages && lock.packages[''] && lock.packages[''].version !== version) {
      lock.packages[''].version = version;
      changed = true;
    }
    return changed;
  });

  if (pkgChanged || lockChanged) {
    anyChange = true;
    if (check) {
      const stale = [pkgChanged && 'package.json', lockChanged && 'package-lock.json']
        .filter(Boolean)
        .map((name) => `${dir}/${name}`);
      console.error(
        `sync-node-versions: ${stale.join(' and ')} not at ${version}${fromNote}`,
      );
    } else {
      console.log(`sync-node-versions: ${dir} -> ${version}${fromNote}`);
    }
  } else {
    console.log(`sync-node-versions: ${dir} already at ${version}`);
  }
}

if (check && anyChange) {
  console.error('sync-node-versions: run it without --check to sync them');
  process.exit(1);
}
if (!anyChange) {
  console.log('sync-node-versions: nothing to do');
}
