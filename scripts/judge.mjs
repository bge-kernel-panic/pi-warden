// Judge factory for the eval scripts. Cloud by default (the historical baseline); set WARDEN_JUDGE=laya to run the
// case sets and calibration against the local Laya model for an A/B comparison. Run `npm run build` first (reads dist/).
import { createTypeSafe } from 'pi-typesafe';
import { createLayaJudge } from '../dist/laya.js';

export function makeJudge(options = {}) {
  return process.env.WARDEN_JUDGE === 'laya' ? createLayaJudge() : createTypeSafe(options);
}
