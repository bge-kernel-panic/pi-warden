/**
 * Format-aware excerpts. The format is detected offline from the output's own markers (see detectFormat); code then keeps
 * the exact lines that matter for that format: failing tests with their assertion lines, compiler and linter errors with
 * file:line, changed files with counts, package manager warnings, and the final summary. Nothing is paraphrased. A parser
 * returns undefined when it does not recognise its markers, and the caller falls back to the generic head/diagnostic/tail.
 */
export type OutputFormat = "vitest_jest" | "node_test" | "tsc" | "eslint" | "pytest" | "git_diff" | "git_log" | "npm_install" | "other";

const LIMIT = 6000;

/** Header lines start a block; the following non-empty lines up to `depth` are kept with it. */
function capture(lines: readonly string[], header: RegExp, keep: RegExp, depth: number, summary: RegExp): string[] | undefined {
  const out: string[] = [];
  let headers = 0;
  let remaining = 0;
  for (const line of lines) {
    if (header.test(line)) { headers++; remaining = depth; out.push(line); continue; }
    if (summary.test(line)) { out.push(line); remaining = 0; continue; }
    if (remaining > 0 && line.trim()) {
      if (keep.test(line)) out.push(line);
      remaining--;
    }
  }
  return headers || out.length ? out : undefined;
}

function vitestJest(lines: readonly string[]): string[] | undefined {
  const summary = /^\s*(?:Test Files|Tests|Test Suites|Snapshots|Duration|Time|Start at|Ran all test suites)\b|^\s*(?:FAIL|PASS)\s{2,}\S/;
  const out = capture(lines, /^\s*(?:FAIL\s|[×✗✕]\s|●\s|❯\s.*\s\d+\s*$|AssertionError\b)/, /expected|received|actual|Error|assert|❯|›|\bat\b.*:\d+:\d+|[+-]\s|^\s*\d+\|/i, 8, summary);
  return out && lines.some(line => summary.test(line) || /[✓✗×✔✖]/.test(line)) ? out : undefined;
}

function nodeTest(lines: readonly string[]): string[] | undefined {
  const summary = /^(?:#|ℹ)\s*(?:tests|suites|pass|fail|cancelled|skipped|todo|duration_ms)\b/;
  if (!lines.some(line => summary.test(line) || /^TAP version/.test(line) || /^\s*[✔✖]\s/.test(line))) return undefined;
  return capture(lines, /^\s*(?:not ok\b|✖\s)/, /error|expected|actual|Error|failureType|code|at\b.*:\d+:\d+|^\s*[+-]\s|^\s*\|/i, 10, summary) ?? [];
}

function tsc(lines: readonly string[]): string[] | undefined {
  const header = /^(?:\S.*?)(?:\(\d+,\d+\)|:\d+:\d+)\s*[-:]\s*error TS\d+/;
  if (!lines.some(line => header.test(line) || /^Found \d+ errors?/.test(line))) return undefined;
  return capture(lines, header, /^\s{2,}\S/, 3, /^Found \d+ errors?|^error TS\d+/);
}

function eslint(lines: readonly string[]): string[] | undefined {
  const problem = /^\s+\d+:\d+\s+(?:error|warning)\s+/;
  if (!lines.some(line => problem.test(line)) || !lines.some(line => /^✖ \d+ problems?|^\d+ problems? \(/.test(line) || /\S+\.[cm]?[jt]sx?$|\.vue$|\.svelte$/.test(line))) return undefined;
  const out: string[] = [];
  let file: string | undefined;
  for (const line of lines) {
    if (/^\S.*\.(?:[cm]?[jt]sx?|vue|svelte|json|md)$/.test(line)) { file = line; continue; }
    if (problem.test(line)) { if (file) { out.push(file); file = undefined; } out.push(line); continue; }
    if (/^✖ \d+ problems?|^\d+ problems? \(|potentially fixable/.test(line)) out.push(line);
  }
  return out;
}

function pytest(lines: readonly string[]): string[] | undefined {
  const summary = /^=+ .*(?:passed|failed|error|skipped|no tests ran).* =+$|^=+ short test summary info =+$|^(?:FAILED|ERROR) \S+/;
  if (!lines.some(line => summary.test(line) || /^_{3,} .+ _{3,}$/.test(line))) return undefined;
  return capture(lines, /^_{3,} .+ _{3,}$/, /^E\s|^>\s|:\d+: |Error|assert/i, 12, summary) ?? [];
}

function gitDiff(lines: readonly string[]): string[] | undefined {
  const files: Array<{ name: string; added: number; removed: number; note?: string }> = [];
  for (const line of lines) {
    const match = /^diff --git a\/(.+?) b\/(.+)$/.exec(line);
    if (match) { files.push({ name: match[2]!, added: 0, removed: 0 }); continue; }
    const current = files.at(-1);
    if (!current) continue;
    if (/^(?:new file mode|deleted file mode|rename (?:from|to)|Binary files)/.test(line)) current.note = (current.note ? `${current.note}; ` : "") + line;
    else if (line.startsWith("+") && !line.startsWith("+++")) current.added++;
    else if (line.startsWith("-") && !line.startsWith("---")) current.removed++;
  }
  if (!files.length) return undefined;
  const added = files.reduce((sum, file) => sum + file.added, 0);
  const removed = files.reduce((sum, file) => sum + file.removed, 0);
  return [...files.map(file => `${file.name}  +${file.added} -${file.removed}${file.note ? `  (${file.note})` : ""}`), `${files.length} files changed, +${added} -${removed}`];
}

function gitLog(lines: readonly string[]): string[] | undefined {
  const out: string[] = [];
  let pendingSubject = false;
  for (const line of lines) {
    const commit = /^commit ([0-9a-f]{7,40})\b/.exec(line);
    if (commit) { out.push(commit[1]!.slice(0, 12)); pendingSubject = true; continue; }
    if (pendingSubject && /^\s{4}\S/.test(line)) { out[out.length - 1] += `  ${line.trim()}`; pendingSubject = false; continue; }
    if (/^[0-9a-f]{7,40} \S/.test(line) || /^\s*(?:modified|new file|deleted|renamed|both modified):|^\?\?\s|^[ MADRCU?!]{2}\s\S/.test(line) || /^(?:On branch|Your branch|Changes (?:to be committed|not staged)|Untracked files|nothing to commit)/.test(line)) out.push(line);
  }
  return out.length ? out : undefined;
}

function npmInstall(lines: readonly string[]): string[] | undefined {
  const keep = /^(?:added|removed|changed|audited|found|up to date|up-to-date)\b|\bvulnerabilit(?:y|ies)\b|^npm (?:warn|WARN|error|ERR!|notice)\b|\bdeprecated\b|^(?:warning|error)\s|^\s*Progress: resolved|^Done in|^Packages: /i;
  const out = lines.filter(line => keep.test(line));
  return out.some(line => /^(?:added|removed|changed|audited|up to date|up-to-date|Done in|Packages:)/i.test(line)) ? out : undefined;
}

const PARSERS: Record<Exclude<OutputFormat, "other">, (lines: readonly string[]) => string[] | undefined> = {
  vitest_jest: vitestJest, node_test: nodeTest, tsc, eslint, pytest, git_diff: gitDiff, git_log: gitLog, npm_install: npmInstall,
};

/**
 * Exact lines selected by the parser for `format`, joined and capped, with the last non-empty lines as a tail so the
 * final status is always present. Undefined when the format is `other` or its markers are absent.
 */
/**
 * The output format, detected offline from its own markers by trying each parser; the first whose markers are present
 * wins. Replaces asking the model "which tool produced this" — the markers are distinctive enough for regex. Undefined
 * means no known format (the caller keeps the generic head/diagnostic/tail excerpt).
 */
export function detectFormat(text: string): OutputFormat | undefined {
  const lines = text.split("\n");
  for (const [format, parser] of Object.entries(PARSERS)) {
    if (parser(lines) !== undefined) return format as OutputFormat;
  }
  return undefined;
}

export function formatExcerpt(text: string, format: OutputFormat): string | undefined {
  if (format === "other") return undefined;
  const lines = text.split("\n");
  const selected = PARSERS[format](lines);
  if (!selected) return undefined;
  const tail = lines.filter(line => line.trim()).slice(-3);
  const body: string[] = [];
  let size = 0;
  let dropped = 0;
  for (const line of selected) {
    const clipped = line.length > 400 ? `${line.slice(0, 400)}…` : line;
    if (size + clipped.length + 1 > LIMIT - 400) { dropped++; continue; }
    body.push(clipped);
    size += clipped.length + 1;
  }
  const parts = [`[${format} excerpt: ${selected.length} selected lines${dropped ? `, ${dropped} omitted for size` : ""}]`, ...body];
  // The final status line is the one the agent reads first; add the tail only when the parser did not already keep it.
  // Diffs and logs have no status line; their last lines are hunk or body text.
  const last = tail.at(-1);
  if (format !== "git_diff" && format !== "git_log" && last !== undefined && !body.includes(last)) parts.push("[tail]", ...tail.map(line => line.length > 400 ? `${line.slice(0, 400)}…` : line));
  return parts.join("\n");
}
