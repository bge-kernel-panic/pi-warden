import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import type { EntryType, Evaluation, Judge, Question, Questions, SystemOneRequest } from "pi-typesafe";

/** Where the local model lives by default; override with PI_WARDEN_LAYA_MODEL. `~` expands to the home directory. */
export function defaultLayaModelPath(): string {
  const configured = process.env.PI_WARDEN_LAYA_MODEL?.trim();
  if (configured) return configured === "~" || configured.startsWith("~/") ? join(homedir(), configured.slice(1)) : configured;
  return join(homedir(), ".pi", "agents", "laya", "laya_int8.onnx");
}

// ModernBERT-large special-token ids. Transformers.js exposes some as undefined (e.g. cls_token_id), so these are the fallbacks.
const CLS_TOKEN_ID = 50281;
const SEP_TOKEN_ID = 50282;
const MASK_TOKEN_ID = 50284;
/**
 * Tokens of state kept per question. The checkpoint is trained at ~256 state / 512 total; raising it feeds more context
 * (the ONNX model accepts longer sequences) but goes out of the trained distribution and costs latency. Override with
 * PI_WARDEN_LAYA_STATE_TOKENS to experiment; recalibrate thresholds if you change it.
 */
const STATE_TOKEN_BUDGET = Number(process.env.PI_WARDEN_LAYA_STATE_TOKENS) > 0 ? Math.floor(Number(process.env.PI_WARDEN_LAYA_STATE_TOKENS)) : 256;
// Post-softmax calibration temperature per qtype, from the Laya model card. Override any one for iteration with
// PI_WARDEN_LAYA_T_CHOICE / _SCORE / _NOUL (a positive number); an unset or invalid var keeps the card default.
const CARD_TEMPERATURE = { choice: 1.637, score: 1.251, noul: 1.983 } as const;
const TEMP_ENV = { choice: "PI_WARDEN_LAYA_T_CHOICE", score: "PI_WARDEN_LAYA_T_SCORE", noul: "PI_WARDEN_LAYA_T_NOUL" } as const;
const tempOf = (name: keyof typeof CARD_TEMPERATURE): number => {
  const override = Number(process.env[TEMP_ENV[name]]);
  return Number.isFinite(override) && override > 0 ? override : CARD_TEMPERATURE[name];
};
const QTYPE = { choice: 0, score: 1, noul: 2 } as const;

/** Non-literal specifier so tsc treats these optional native deps as `any` and does not require them to be installed. */
const importModule = (name: string): Promise<any> => import(name);

const asText = (value: EntryType): string => (typeof value === "string" ? value : value == null ? "" : JSON.stringify(value));

/**
 * Render the request state as natural text rather than JSON. Laya reads prose far better than escaped JSON (and plain
 * text is more token-efficient under the 256-token state budget). Objects become `key:\n<value>` blocks; short scalars
 * stay inline; arrays are blank-line separated. This is a calibration lever — tweak the layout and re-run the case sets.
 */
function stateText(value: EntryType): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map(entry => stateText(entry as EntryType)).join("\n\n");
  return Object.entries(value)
    .map(([key, entry]) => {
      const text = stateText(entry as EntryType);
      return text.includes("\n") || text.length > 60 ? `${key}:\n${text}` : `${key}: ${text}`;
    })
    .join("\n");
}

/** A label carries its Jev description into the option token; the classifier reads the meaning, not just the bare key. */
const optionText = (key: string, description: EntryType): string => {
  const described = asText(description).trim();
  return described ? `${key}: ${described}` : key;
};

interface Prepared {
  qtypeName: keyof typeof QTYPE;
  /** One entry per [MASK] marker, in sequence order. */
  options: { key: string; text: string }[];
}

/** The three SDK question shapes map one-to-one onto Laya's qtype and its per-option markers. */
function prepare(question: Question): Prepared {
  if (question.type === "noul") {
    const criteria = question.criteria ?? undefined;
    return {
      qtypeName: "noul",
      options: [
        { key: "false", text: optionText("false", criteria?.false ?? null) },
        { key: "true", text: optionText("true", criteria?.true ?? null) },
      ],
    };
  }
  if (question.type === "choice") {
    return { qtypeName: "choice", options: Object.entries(question.criteria).map(([key, desc]) => ({ key, text: optionText(key, desc) })) };
  }
  return { qtypeName: "score", options: question.criteria.map((desc, index) => ({ key: String(index), text: optionText(String(index), desc) })) };
}

/** The reference span at the second marker when scoring one option on its own; masked out via marker_mask. */
const FILLER_OPTION = "none";

/**
 * Serialize one question into Laya's trained layout:
 * `[CLS] <qtype> question: <instructions> [SEP] [MASK] <optA> [MASK] <optB> [SEP] <state> [SEP]`.
 * The exported head is fixed to exactly 2 markers / 2 logits (see the model I/O), so an N-way question is scored one
 * option at a time (that option at marker 0, a masked filler at marker 1) and softmaxed across options by the caller.
 * ponytail: this serialization and prepare()/optionText() are the calibration surface — validate label agreement
 * against real Laya outputs (scripts/*-cases.mjs) before trusting it as a Jev replacement.
 */
function encodePair(tokenizer: any, qtypeName: string, instructions: string, textA: string, textB: string, state: string): { ids: number[]; markers: number[] } {
  const cls = tokenizer.cls_token_id ?? tokenizer.bos_token_id ?? CLS_TOKEN_ID;
  const sep = tokenizer.sep_token_id ?? tokenizer.eos_token_id ?? SEP_TOKEN_ID;
  const mask = tokenizer.mask_token_id ?? MASK_TOKEN_ID;
  const head = tokenizer.encode(`${qtypeName} question: ${instructions}`, { add_special_tokens: false });
  const optionIds = [mask, ...tokenizer.encode(` ${textA}`, { add_special_tokens: false }), mask, ...tokenizer.encode(` ${textB}`, { add_special_tokens: false })];

  const ids = [cls, ...head, sep];
  const markers = optionIds.flatMap((token, index) => (token === mask ? [ids.length + index] : []));
  ids.push(...optionIds, sep);
  ids.push(...tokenizer.encode(state, { add_special_tokens: false }).slice(0, STATE_TOKEN_BUDGET), sep);
  return { ids, markers };
}

function softmax(logits: number[], temperature: number): number[] {
  const scaled = logits.map(value => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(value => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

/** Assemble one SDK response from the per-option probabilities, matching the shapes validResult expects. */
function answerFor(question: Question, prepared: Prepared, probs: number[]): unknown {
  if (question.type === "noul") return { type: "noul", noul: probs[1] ?? 0 };
  const probabilities = Object.fromEntries(prepared.options.map((option, index) => [option.key, probs[index] ?? 0]));
  let best = 0;
  for (let index = 1; index < probs.length; index++) if ((probs[index] ?? 0) > (probs[best] ?? 0)) best = index;
  if (question.type === "choice") return { type: "choice", choice: prepared.options[best]!.key, confidence: probs[best] ?? 0, probabilities };
  const legend = Object.fromEntries(question.criteria.map((desc, index) => [String(index), desc]));
  const score = probs.reduce((sum, probability, index) => sum + index * probability, 0);
  return { type: "score", score, confidence: probs[best] ?? 0, legend, probabilities };
}

/**
 * A local {@link Judge} backed by the Laya ONNX classifier. Drop-in for the pi-typesafe client: same `evaluate`
 * contract, same answer shapes, no data leaves the machine. Fails fast (per repo config-error style) if the model
 * file is absent, rather than silently degrading.
 */
export function createLayaJudge(modelPath: string = defaultLayaModelPath()): Judge {
  if (!existsSync(modelPath)) {
    throw new Error(`Laya model not found at ${modelPath}. Download Mattepiu/laya-onnx (laya_int8.onnx + tokenizer.json) into that directory, or set PI_WARDEN_LAYA_MODEL.`);
  }
  const modelDir = dirname(modelPath);
  // Mirrors the cloud client's usage counters so the eval scripts (getUsage/getSpend) are a one-line swap.
  const usage = { requestsStarted: 0, requestsSucceeded: 0, requestsFailed: 0, inputTokens: 0, outputTokens: 0 };
  // Session and tokenizer are ~0.7-1 GB resident; load once per process on first use, then reuse.
  let ready: Promise<{ ort: any; session: any; tokenizer: any }> | undefined;
  const init = () =>
    (ready ??= (async () => {
      const ort = await importModule("onnxruntime-node");
      const { AutoTokenizer, env } = await importModule("@huggingface/transformers");
      env.allowRemoteModels = false;
      // Transformers.js resolves a model id under localModelPath; point it at the parent so the id is the model's own dir.
      env.localModelPath = dirname(modelDir);
      const tokenizer = await AutoTokenizer.from_pretrained(basename(modelDir));
      const session = await ort.InferenceSession.create(modelPath);
      return { ort, session, tokenizer };
    })());

  const evaluate = async <Q extends Questions>(request: SystemOneRequest<Q>, options?: { signal?: AbortSignal }): Promise<Evaluation<Q>> => {
    usage.requestsStarted++;
    const { ort, session, tokenizer } = await init().catch(error => { usage.requestsFailed++; throw error; });
    const state = stateText(request.state);
    const answers: Record<string, unknown> = {};
    let inputTokens = 0;
    const start = performance.now();
    // One forward pass over a fixed 2-marker sequence; markerActive says whether the second marker counts. Returns the 2 logits.
    const runPair = async (qtype: number, ids: number[], markers: number[], markerActive: boolean): Promise<number[]> => {
      inputTokens += ids.length;
      const output = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
        marker_pos: new ort.Tensor("int64", BigInt64Array.from(markers, BigInt), [1, 2]),
        marker_mask: new ort.Tensor("bool", Uint8Array.from([1, markerActive ? 1 : 0]), [1, 2]),
        qtype: new ort.Tensor("int64", BigInt64Array.from([BigInt(qtype)]), [1]),
      });
      return Array.from(output.logits.data as Float32Array).map(Number);
    };
    try {
      for (const [name, question] of Object.entries(request.questions)) {
        if (options?.signal?.aborted) throw new Error("Laya evaluation cancelled.");
        const prepared = prepare(question as Question);
        const instructions = asText((question as Question).instructions ?? null);
        const qtype = QTYPE[prepared.qtypeName];
        let probs: number[];
        if (prepared.qtypeName === "noul") {
          // Native 2-marker case: false vs true in one pass; probs[1] is P(true).
          const { ids, markers } = encodePair(tokenizer, "noul", instructions, prepared.options[0]!.text, prepared.options[1]!.text, state);
          probs = softmax(await runPair(qtype, ids, markers, true), tempOf("noul"));
        } else {
          // N-way: score each option on its own (marker 0, filler masked at marker 1), then softmax the per-option logits.
          const perOption: number[] = [];
          for (const option of prepared.options) {
            const { ids, markers } = encodePair(tokenizer, prepared.qtypeName, instructions, option.text, FILLER_OPTION, state);
            perOption.push((await runPair(qtype, ids, markers, false))[0]!);
          }
          probs = softmax(perOption, tempOf(prepared.qtypeName));
        }
        answers[name] = answerFor(question as Question, prepared, probs);
      }
    } catch (error) {
      usage.requestsFailed++;
      throw error;
    }
    usage.requestsSucceeded++;
    usage.inputTokens += inputTokens;
    return {
      model: `laya-${basename(modelPath, ".onnx")}`,
      answers,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
      elapsedMs: Math.round(performance.now() - start),
    } as unknown as Evaluation<Q>;
  };

  // getUsage/getSpend stubs keep the eval scripts' reporting a drop-in swap; Laya is free, so spend is always zero.
  const getUsage = () => ({ ...usage, estimatedUsd: 0 });
  const getSpend = () => ({
    session: getUsage(),
    today: { requestsStarted: usage.requestsStarted, inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, estimatedUsd: 0 },
    caps: {},
    usdPerMTok: 0,
  });
  return { evaluate, getUsage, getSpend } as unknown as Judge;
}
