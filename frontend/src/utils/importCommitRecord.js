// The imports whose commit has been sent and not yet answered, kept in
// sessionStorage for each library, so that a reload in that window does not
// lose the staging id. Once the commit is answered, the tasks store's
// own record (useTasksStore, "pixlstash:importRuns") takes over: it follows
// the import from there, including after a reload. sessionStorage does not
// outlive a closed tab, and a staging session belongs to the library that
// opened it, so the records are kept apart by library.
//
// The staging ids this page is committing, or picking back up, are also held
// in memory: two importers are mounted (the grid's and the sidebar's), and a
// view switch remounts the grid's while its commit is still unanswered, so
// without that one commit would be sent again by an importer that only read
// the record.

const KEY_PREFIX = "pixlstash:importCommit:";
const claimed = new Set();

function storageKey(libraryUuid) {
  return `${KEY_PREFIX}${libraryUuid}`;
}

function read(libraryUuid) {
  try {
    const raw = window.sessionStorage.getItem(storageKey(libraryUuid));
    const records = raw ? JSON.parse(raw) : [];
    return Array.isArray(records) ? records : [];
  } catch (error) {
    console.warn(
      "import: could not read the commits a reload interrupted; they are not picked up",
      libraryUuid,
      error,
    );
    return [];
  }
}

function write(libraryUuid, records) {
  try {
    if (records.length) {
      window.sessionStorage.setItem(
        storageKey(libraryUuid),
        JSON.stringify(records),
      );
    } else {
      window.sessionStorage.removeItem(storageKey(libraryUuid));
    }
  } catch (error) {
    console.warn(
      "import: could not save the commit in progress; a reload now loses the import",
      libraryUuid,
      error,
    );
  }
}

/**
 * Keep `record` until its commit is answered.
 *
 * @param {string} libraryUuid
 * @param {{runId: string, stagingId: string, total: number}} record
 */
export function saveCommitRecord(libraryUuid, record) {
  claimed.add(record.stagingId);
  const others = read(libraryUuid).filter(
    (entry) => entry?.stagingId !== record.stagingId,
  );
  write(libraryUuid, [...others, record]);
}

/** The commit of `stagingId` was answered: nothing is left to pick up. */
export function dropCommitRecord(libraryUuid, stagingId) {
  claimed.delete(stagingId);
  const records = read(libraryUuid);
  const kept = records.filter((entry) => entry?.stagingId !== stagingId);
  if (kept.length !== records.length) write(libraryUuid, kept);
}

/**
 * The commits of `libraryUuid` that nothing on this page is settling yet,
 * now claimed by the caller, which settles each and then drops its record.
 */
export function claimCommitRecords(libraryUuid) {
  const unclaimed = read(libraryUuid).filter(
    (entry) =>
      typeof entry?.stagingId === "string" &&
      typeof entry?.runId === "string" &&
      !claimed.has(entry.stagingId),
  );
  for (const entry of unclaimed) claimed.add(entry.stagingId);
  return unclaimed;
}
