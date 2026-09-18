// Billable synthetic cases for the subagent triage question only. Build first.
// Run standalone: node scripts/subagent-cases.mjs  (or via live-smoke.mjs subagent for the cloud judge).
import { buildTriageRequest, defaultConfig } from '../dist/index.js';
import { makeJudge } from './judge.mjs';
import { calibrateGuard, loadCalibration } from './calibrate.mjs';

/**
 * One request per case, sent straight to the `wake` question with the request the guard builds, so the offline
 * pre-filter does not answer for the model. The interesting cases are the reports that mention trouble and still
 * need no wake: a measurement that only fed failures would prove nothing about the threshold.
 */
export async function runSubagentCases(judge, report, threshold) {
  const config = defaultConfig();
  const wakeAt = threshold ?? config.subagent.threshold;
  const task = 'Add a created_at column to the users table and make the migration safe to re-run';
  const cases = [
    {
      name: 'progress line', expect: 'silent', incremental: true, kind: 'subagent-incremental-child-notify',
      text: 'Background task progress: **explorer** is still reading src/config.ts (12 files inspected so far)',
    },
    {
      name: 'clean completion', expect: 'silent', incremental: false, kind: 'subagent-notify',
      text: 'Background tasks completed (1): **writer**\n\n1. writer\nResult: rewrote the parser and ran the test suite; all 12 tests pass',
    },
    {
      name: 'hard failure', expect: 'wake', incremental: false, kind: 'subagent-notify',
      text: 'Background tasks completed (1): **explorer**\n\n1. explorer\nResult: the migration step failed with exit code 1; nothing was migrated',
    },
    {
      name: 'blocked on a decision', expect: 'wake', incremental: false, kind: 'subagent-notify',
      text: '1. explorer\nBlocked: the fixture uses two field names and both are load-bearing. Which one is canonical? I need your decision before I continue.',
    },
    {
      name: 'failure already fixed', expect: 'silent', incremental: false, kind: 'subagent-notify',
      text: 'Background tasks completed (1): **writer**\n\n1. writer\nResult: 3 tests failed on the first run because the fixture was stale; I fixed the fixture and all 12 tests pass now',
    },
    {
      name: 'stopped, no result', expect: 'wake', incremental: false, kind: 'subagent-notify',
      text: '1. explorer was stopped by the watchdog after 900 s. No result was produced and the working tree is mid-edit.',
    },
    {
      name: 'child asks the parent', expect: 'wake', incremental: false, kind: 'subagent_supervisor_request',
      text: 'Child asks: the migration is written. Should I merge it into main now or open a draft PR for review?',
    },
    {
      name: 'completed, one check skipped', expect: 'silent', incremental: false, kind: 'subagent-notify',
      text: 'Background tasks completed (1): **writer**\n\n1. writer\nResult: implemented the parser and ran the unit tests (all pass). The end-to-end suite was skipped because it needs Docker.',
    },
  ];
  for (const item of cases) {
    try {
      const request = buildTriageRequest({ id: item.name, customType: item.kind, incremental: item.incremental, text: item.text }, task);
      const result = await judge.evaluate(request, { signal: AbortSignal.timeout(8000) });
      const answer = result.answers.wake;
      const probability = typeof answer.noul === 'number' ? answer.noul : 0;
      const wake = probability >= wakeAt;
      report(wake === (item.expect === 'wake'), item.name, `${wake ? 'wake' : 'silent'} wake=${probability.toFixed(2)} threshold=${wakeAt} (${result.elapsedMs} ms)`, { score: probability, label: item.expect === 'wake' });
    } catch (error) {
      report(false, item.name, `error ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const calibrating = process.env.CALIBRATE === '1';
  const threshold = calibrating || process.env.WARDEN_JUDGE !== 'laya' ? undefined : loadCalibration().subagent?.wake;
  let total = 0, miss = 0; const points = [];
  await runSubagentCases(makeJudge({ maxRequests: 20 }), (ok, name, detail, info) => {
    total++; if (!ok) miss++;
    if (info) points.push(info);
    console.log(`${ok ? 'ok  ' : 'MISS'} ${name}  ${detail}`);
  }, threshold);
  console.log(`\n${total - miss}/${total} matched`);
  if (calibrating) calibrateGuard('subagent', { wake: points });
}
