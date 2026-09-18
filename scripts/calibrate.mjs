// Threshold calibration for the local Laya judge. Each noul question emits a probability; the right cut point differs
// per question because Laya's scores are biased differently per question. Given labeled scores from a case set, pick the
// threshold that best separates violations from clean cases, and persist it so the case scripts (and later the extension)
// can load calibrated cut points instead of a single hand-picked 0.7.
//
// Note: fitting and testing on the same small case set is optimistic (no held-out split), so treat these as a sanity
// signal and a starting point, not a generalization guarantee.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CALIBRATION_FILE = join(dirname(fileURLToPath(import.meta.url)), '..', 'eval', 'laya-calibration.json');

const round = n => Math.round(n * 1000) / 1000;

/**
 * The threshold maximizing F0.5 over labeled points [{ score, label:boolean }]. F0.5 weights precision above recall
 * because a false positive here nags the agent (costly) while a miss is quiet; that matches what the guards want and
 * gives higher, safer cut points than balanced accuracy. Ties break toward the higher threshold. Returns metrics too.
 */
export function bestThreshold(points, beta = 0.5) {
  const positives = points.filter(p => p.label).length;
  const negatives = points.length - positives;
  if (!positives || !negatives) return { threshold: 1, precision: 0, recall: 0, acc: round(negatives / points.length), n: points.length, degenerate: true };
  const grid = [0, ...new Set(points.map(p => p.score))].sort((a, b) => a - b).concat(1);
  const b2 = beta * beta;
  let best = { f: -Infinity };
  for (const t of grid) {
    let tp = 0, fp = 0;
    for (const p of points) if (p.score >= t) (p.label ? tp++ : fp++);
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp / positives;
    const f = precision + recall ? (1 + b2) * precision * recall / (b2 * precision + recall) : 0;
    if (f >= best.f) best = { threshold: round(t), precision: round(precision), recall: round(recall), acc: round((tp + negatives - fp) / points.length), n: points.length, f };
  }
  delete best.f;
  return best;
}

export function loadCalibration() {
  try { return existsSync(CALIBRATION_FILE) ? JSON.parse(readFileSync(CALIBRATION_FILE, 'utf8')) : {}; } catch { return {}; }
}

/** Merge one guard's thresholds into the shared file, leaving other guards untouched. */
export function saveCalibration(guard, thresholds) {
  const all = loadCalibration();
  all[guard] = thresholds;
  mkdirSync(dirname(CALIBRATION_FILE), { recursive: true });
  writeFileSync(CALIBRATION_FILE, JSON.stringify(all, null, 2) + '\n');
  return CALIBRATION_FILE;
}

/** Fit and report a threshold per key from accumulated points; persist under `guard` when CALIBRATE is set. */
export function calibrateGuard(guard, pointsByKey) {
  const fitted = {};
  console.log(`\n# calibration for ${guard} (F0.5-optimal on the case set)`);
  for (const [key, points] of Object.entries(pointsByKey)) {
    const fit = bestThreshold(points);
    fitted[key] = fit.threshold;
    console.log(`  ${key.padEnd(34)} t=${fit.threshold.toFixed(2)}  acc=${(fit.acc * 100).toFixed(0)}%  P=${fit.precision.toFixed(2)} R=${fit.recall.toFixed(2)}  (n=${fit.n}${fit.degenerate ? ', one-class' : ''})`);
  }
  const path = saveCalibration(guard, fitted);
  console.log(`  saved -> ${path}`);
  return fitted;
}
