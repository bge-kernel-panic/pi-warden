# Step 06 — Review mode: harvest on-domain safety labels from real sessions

**Status: design/implementation plan, not built.** This is the feature that
generates pi-warden's *own* training data — the on-domain, naturally long-context
data that off-the-shelf sets (DocNLI/QuALITY) can only approximate. It's the
inspiration behind kotoba-lang's `data-hermes` (agent tool-calls labeled by
outcome), which they never published.

File:line anchors below are from the working tree at authoring time (verify before
editing).

## Goal + phasing

Collect a `(tool call → "was this safe?" label)` dataset from real use, with a
fast keyboard review UI, so we can retrain Laya on data that actually matches how
pi-warden is used.

- **Phase 1 (this plan): shadow collection.** Review mode observes Bash tool
  calls, records Laya's safety verdict + leash's verdict + the user's label, and
  writes canonical-format JSONL. Guards stay **silent** (no steering/nagging).
  Nothing gates anything.
- **Phase 2 (later, gated on a trained+trusted model): enforcement.** pi-warden's
  action guard blocks/confirms Bash on the safety verdict, and leash is relaxed
  (pi-warden owns the vendored analyzer + lets Laya handle edge cases). NOT built
  until Phase 1 data produces a model that earns trust. Building enforcement on
  the current unvalidated checkpoint is explicitly rejected.

## The label: "was this a safe operation" (noul)

**Definition (reversibility-centered):** *Safe = reversible and non-destructive.
Unsafe = destroys/overwrites unrecoverable data, has irreversible effects
(force-push, deploy with no rollback, `rm` of unrecoverable data), or exfiltrates.*

Examples: `git push` = **safe** (revertible). `git push -f` / `--force-with-lease`
= **unsafe** (overwrites remote history). This is ~the inverse of pi-warden's
existing `irreversible` noul (`src/guard.ts:442`), so reuse/adapt that wording for
the captured question.

Only this one label in v1. The "would this command succeed?" label is **cut** —
trivial-environment failures (find vs gfind, missing deps) make it noisy and it's
not what we care about. Trivial to add later via a second `noul` if wanted.

## Scope + toggle

- **Bash tool calls only.** Highest-risk, clearest to judge, cleanest labels.
- **Off by default.** New `config.review.enabled` (default false); bump
  `CONFIG_SCHEMA` 6→7 (`src/config.ts:206`); add `/warden review on|off` to the
  command table (`src/extension.ts:890`), persisted via `setUserSetting` /
  `writeUserConfig` (owner-only, 0o600, `src/config.ts:499-512`).
- When on, **guards go silent**: we read the safety verdict without acting on it.

## Architecture (seams from recon)

### Capture — `pi.on("tool_call")` (`src/extension.ts:477`)
Preflight hook; fires once per proposed call with `event.toolName` / `event.input`.
When `config.review.enabled` and `toolName === "Bash"`:
1. Ask the "safe?" noul via the **extras-questions seam** — `EvaluateOptions.questions`
   (`src/guard.ts:135`), answered into `verdict.extra.safe` and *never acted on*
   (same mechanism `scripts/calibrate-action.mjs` uses to trial candidate
   questions). No new acting rule needed.
2. Run the **vendored leash analyzer** (see below) on the command → `{blocked, reason}`.
3. Queue `{ call, laya_prob: verdict.extra.safe, leash: {blocked, reason},
   state: buildRequest(...).state }`. Tie to the trace via the existing `traceOf`
   WeakMap / `HoldLedger.record()` (`src/holds.ts:108`) rather than a parallel store.

### Outcome — `pi.on("tool_result")` (`src/extension.ts:621`)
Attach the real outcome (`event.isError`, `resultFailed(...)` at line 627) to the
queued item. (Not the label — outcome is context, the label is safety — but useful
metadata.)

### Leash — vendored core analyzer
leash's decision engine is a self-contained class `CommandAnalyzer`
(`packages/core/command-analyzer.ts` in `sailthru/leash`) exposing a pure
`analyze(command) → { blocked, reason }` (verified against installed
`@sailthru/leash@1.0.19` factory bundle, method at line 636; ctor
`new CommandAnalyzer(workingDirectory, allowedDirectories)` wraps a `PathValidator`).
- **Vendor** that file (+ `PathValidator` dep) into e.g. `src/vendor/leash/`, pinned
  to a specific `sailthru/leash` commit; document the pin + sync in a header
  comment. It's first-party (Sailthru/Zeta) code.
- Instantiate once with pi-warden's cwd + allowed dirs; call `analyze()` per Bash
  command. This is **read-only in v1** — the real leash extension keeps enforcing
  independently; we only record its verdict. In Phase 2, pi-warden's copy becomes
  the gate and the standalone leash extension is retired.
- Seeds the default label: `leash.blocked → default "unsafe" (gold_index 0)`, user
  can flip it.

### Review UI — `ctx.ui.custom({ overlay:true, anchor:"center" })`
Copy the working `Component` + overlay pattern from `openTracePanel`
(`src/panel.ts:142`) and the focus-row keyboard input model from pi-atelier's
settings workspace (`~/.pi/agent/npm/node_modules/pi-atelier/src/settings-workspace.ts:423`,
mounted at `menu.ts:524`). Keyboard model: `handleInput(data)` + `matchesKey` /
bare-char compares; `this.requestRender()` after state changes; `done(result)` closes.
(`ctx.ui.confirm` exists for trivial yes/no — but we need multi-item + engagement
tracking, so a custom overlay is warranted.)

## Cadence + UX

- **Trigger:** at the agent hand-back boundary (`turn_end` / `agent_end`,
  `src/extension.ts:471`/`812`) — never mid-run. Only open if **≥10 items queued**.
- **Paginate 10 at a time** within the modal.
- **Keys:** `tab`/`shift-tab` move between items; `space` toggles the label
  (pre-filled to Laya's answer); `enter` dismisses/commits the page; **`s`
  postpones** the whole review to the next hand-back boundary.
- Items sorted **leash↔Laya disagreement first** (leash.blocked ≠ (laya_prob<0.5)),
  then borderline Laya by `|laya_prob − 0.5|` ascending.

## Label trust (engagement)

Record an `engagement` tier per item so retraining can trust the good labels:
- `overridden` — user toggled away from Laya's value. **Strongest.**
- `touched` — user tab-navigated to / dwelt on the item without changing it. Confirms.
- `untouched` — enter-through without navigating. **Weak** — store it, but flagged.

Store the **raw** signal (hard label = user's final value, `engagement`, and a
suggested `weight` e.g. overridden 1.0 / touched 0.8 / untouched 0.3). Decide the
*treatment* at training time, not now: either down-weight low-trust items in the
loss, or soften their `target` toward 0.5 (the training format has a soft `target`,
not just a hard label — so both are one-liners in the harvest loader). Keeping the
raw signal makes it tunable.

## Storage + redaction

- **Path:** `<agentDir>/pi-warden/harvest/<YYYY-MM-DD>-<session>.jsonl`
  (mirror `holdLogPath`, `src/holds.ts:281`). Owner-only (mkdir 0o700 / write 0o600).
  Use `appendFile` (append-only), unlike `HoldLog.save()` which rewrites.
- **Redaction is mandatory.** Run every captured command + state through
  `redact()` (`src/redact.ts:21`) — the canonical scrubber used wherever text
  leaves the machine — and `reportDigest()` (`src/subagent.ts:68`) for long output.
- **Policy note / privacy:** pi-warden *currently never writes commands or paths to
  disk* (the feedback log stores only scores/outcomes, `src/holds.ts:20`).
  Capturing real commands is a deliberate change. It stays local, owner-only,
  redacted — surface it in the consent/disclosure string (`src/extension.ts:40`)
  so enabling review mode is informed.

## Harvest record shape (canonical + extras)

Matches `training/README.md`'s canonical JSONL (extra fields are ignored by the
public-dataset scripts; a future `reformat_harvest.py` consumes them):

```json
{
  "qtype": "noul",
  "instructions": "<safety question, adapted from guard.ts:442>",
  "criteria": {"false": "unsafe / destructive / irreversible", "true": "safe / reversible"},
  "gold_index": 1,
  "state": "<redacted, UNTRUNCATED buildRequest().state via stateText()>",
  "source": "review",
  "engagement": "overridden|touched|untouched",
  "weight": 1.0,
  "laya_prob": 0.42,
  "leash": {"blocked": false, "reason": null},
  "outcome": {"failed": false}
}
```

Note: store `state` **untruncated** (do not apply `STATE_TOKEN_BUDGET`,
`src/laya.ts:22`) — the whole point is longer-window training. Build it via the
same `stateText()` (`src/laya.ts:43`) so it's byte-identical in shape to
inference-time input.

## Phase 2 (sketch, not built)
Once Phase 1 data yields a retrained + validated model: flip the action guard to
block/confirm Bash on `verdict.safe`, retire the standalone leash extension in
favour of the vendored analyzer + Laya override, and relax the analyzer's strictness
where Laya reliably catches the edge cases. Recalibrate first (see `05` §4).

## Open items
- Confirm cutting "would-succeed" (assumed cut).
- Pick the exact hand-back event (`turn_end` vs `agent_end`) during implementation.
- Decide untouched treatment (down-weight vs target-softening) at training time.
- `reformat_harvest.py` (future): harvest JSONL → canonical + apply engagement weights.
