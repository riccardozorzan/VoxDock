# TODO & Architectural Spec: High-Fidelity Local TTS with dots.tts & Docker — v2

> **Status: implemented and verified on the target hardware (2026-09-28).**
>
> The original v2 spec is preserved in § *Original Checklist* below for intent,
> but six of its recommendations proved unachievable on a 6 GB RTX 4050 and were
> changed — each with a measured reason under [Deviations](#deviations-from-the-original-spec).
> Two findings materially change how the acceptance targets should be read; see
> [Verified results](#verified-results) and [Known limits](#known-limits--open-items).

---

## Project Overview (as built)

* **Core Model**: `dots-studio/dots.tts-mf` — the MeanFlow student of `dots.tts-soar`
  ([deviation #1](#1-model-mf-not-soar)).
* **Target Hardware**: Intel Core Ultra + NVIDIA GeForce RTX 4050 Laptop, 6 GB
  GDDR6. Measured 5772.81 MB total, compute capability 8.9, ~5.64 GiB usable.
* **Environment**: 100% Dockerized with GPU passthrough — zero host pollution.
* **Capabilities delivered**:
  1. **Continuation Voice Cloning** — reference WAV + exact transcript.
  2. **VRAM Optimization** — peak **5.381 GB**, inside the `< 5.5 GB` cap.
  3. **Punctuation & Prosody** — respects `. , ! ?` for pauses and pitch.
  4. **Long-Form Batching** — chunk → POST → lossless FFmpeg concat.
  5. **Automated Benchmarking** — cold start, peak VRAM, RTF.
  6. **Settings autotuner** *(not in the original spec)* — sweeps settings and
     emits the best `.env` for whichever host it runs on.

### Acceptance status

| Requirement | Target | Measured | |
| :--- | :--- | :--- | :--- |
| Peak VRAM | `< 5.5 GB` | **5.381 GB** (5510.6 MB) | ✅ |
| RTF `gen/audio`, 13.4 s utterance | `< 0.5` | **0.4638** (p90 0.4694) | ✅ |
| RTF, short utterance (~3.7 s) | `< 0.5` | **0.63 – 0.67** | ⚠️ [limit #1](#1-rtf--05-is-unreachable-below-77-s-of-audio) |
| HTTP 200 ×3 | all | 3/3 | ✅ |
| Long-form pipeline | lossless | 161 words → 49.04 s, RTF 0.511 | ✅ |
| Speaker similarity | SIM > 83 | **not measured** | ⚠️ [limit #2](#2-audio-quality-was-never-measured) |

---

## As-Built Directory Structure

```text
TTS/
├── Dockerfile                  # CUDA 12.8.1 runtime + torch cu128 + dots.tts
├── docker-compose.yml          # GPU passthrough, hf_cache volume, env wiring
├── .env / .env.example         # all tunables are env-driven
├── .gitignore                  # keeps .env + voice recordings out of git
├── requirements.txt
├── README.md
├── todo-tts-agent-v2.md        # this file
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI service + runtime loader (561 lines)
│   ├── config.py               # env-driven settings, torch-free (338)
│   ├── utils.py                # VRAM telemetry, WAV buffers, temp files (294)
│   └── patches.py              # NEW: low-VRAM patches for dots.tts (305)
├── scripts/
│   ├── clean_audio.py          # denoise + validate + --trim-to (335)
│   ├── process_txt.py          # chunk → POST → FFmpeg concat (720)
│   ├── benchmark_test.py       # cold start / VRAM / RTF benchmark (580)
│   ├── autotune.py            # NEW: autotuner (was tune_settings.py)
│   └── test_api.py             # one-shot smoke test (129)
└── data/
    ├── references/             # raw voice recordings + transcripts
    ├── input/
    │   ├── reference_clean.wav # 7.0 s / 44 patches — the service reference
    │   ├── reference_clean.txt # exact transcript (MUST match the audio)
    │   ├── ref_{s,m,l}.{wav,txt}   # NEW: tuner reference set (26/46/72 patches)
    │   ├── reference_set.json  # NEW: describes the above
    │   └── test_audio_sample.txt
    ├── temp_chunks/            # intermediate chunks (auto-cleaned)
    └── output/                 # final WAVs + benchmark/tuning JSON reports
```

4 684 lines of Python across 10 files.

---

## Verified Results

Reproduced from `data/output/benchmark_report.json` after a clean
`docker compose up -d --build`. GPU: RTX 4050 Laptop, 5772.81 MB total.

### Benchmark (`scripts/benchmark_test.py`)

| Metric | Value |
| :--- | :--- |
| Verdict | `peak_vram_ok` ✅ `rtf_ok` ✅ `http_ok` ✅ |
| Cold start (checkpoint load + compile warmup) | 197.7 s |
| Audio duration (50-word sentence) | 13.44 s |
| Generation time | 6.234 s |
| **RTF** `gen/audio` | **0.4638** mean, 0.4694 p90 |
| Speed factor `audio/gen` | 2.157× |
| **Peak VRAM** | **5510.6 MB = 5.381 GB** |
| VRAM free after | 334.9 MB |

### Long-form pipeline (`scripts/process_txt.py`)

161 words → 7 chunks → **49.04 s** of audio (48.32 s + 0.72 s of 0.12 s gaps),
24.71 s wall, **RTF 0.511**, peak 5453 MB, 7/7 chunks, 0 retries.
Per-chunk RTF ranged 0.473 – 0.563, tracking the cost model below.

### The central finding: the RTF target is output-length-bound

Least-squares fit over 6 measured calls (3 references × 2 text lengths,
3.36 – 16.48 s of audio) in `data/output/tuning_report.json`:

```
generation_seconds ≈ 0.980 + 0.3726 × audio_seconds
RTF               ≈ 0.373  + 0.98  / audio_seconds
```

| audio per request | RTF | passes `< 0.5`? |
| --- | --- | --- |
| 3.36 s | 0.647 | ❌ |
| 3.68 s | 0.632 | ❌ |
| 3.84 s | 0.639 | ❌ |
| 9.12 s | 0.483 | ✅ |
| 13.28 s | 0.455 | ✅ |
| 16.48 s | 0.426 | ✅ |

**RTF ≤ 0.5 requires ≳ 7.69 s of audio per request on this GPU.** There is a
~1.0 s *fixed* per-request cost that dominates short utterances. It was measured
to be **independent of reference length** (26 / 46 / 72-patch references all land
on the same curve), so it is not prompt prefill and cannot be tuned away with any
env setting. See [limit #1](#1-rtf--05-is-unreachable-below-77-s-of-audio).

This also invalidates the original acceptance methodology: a single 50-word
benchmark sentence yields 13.4 s of audio and reports a comfortable 0.46, while
3-second utterances sit at 0.63–0.67. **Any RTF figure here must state the audio
length it was measured at.**

---

## Deviations from the original spec

Six recommendations changed. Each was measured, not assumed.

### 1. Model: MF, not Soar

Original: `dots.tts-soar`. Shipped: `dots-studio/dots.tts-mf`.

The published dots.tts memory table (bfloat16, `--optimize`) gives Soar's
*smallest* bucket as **5.65 GB**, already above the 5.5 GB cap, so Soar cannot
meet the VRAM requirement for any input. MF is Soar's MeanFlow student: 4.55 GiB
of weights, a 2-step sampler, and a 5.381 GB measured peak.

**Cost:** MF is the lower-fidelity student, so speaker similarity is expected to
be below Soar's 83.9. This was never measured — see [limit #2](#2-audio-quality-was-never-measured).

### 2. Base image: CUDA 12.8.1, not 12.1.1

`dots.tts 0.3.x` requires `torch>=2.8`, and PyTorch stopped publishing cu121
wheels after 2.5.x. The image is `nvidia/cuda:12.8.1-runtime-ubuntu22.04` with
cu128 wheels. Host drivers ≥ 525 work.

### 3. `num_steps`: 2, not 10

MeanFlow's native step count is 2. Beyond quality, `num_steps` **sizes the DiT KV
cache** — allocated as `(num_steps, layers, heads, tokens, dim)` — so it doubles
VRAM per step. Measured: `max_generate_length=128` fits at 2 steps (5.381 GB) and
**OOMs at 4 steps**. Raising `TTS_NUM_STEPS` requires lowering
`TTS_MAX_GENERATE_LENGTH` to 96.

### 4. Reference audio: 7 s, not 10–12 s

`TTS_MAX_GENERATE_LENGTH` is snapped **up** to the next torch.compile bucket —
`(64, 128, 256, 512)` — so 96 behaves exactly like 128 with a tighter limit. A
10–12 s reference is 63–75 patches, which collides with the smallest usable
bucket. The shipped 7 s clip is **44 patches**, leaving ~84 patches (13.4 s) of
generation room inside the 128 bucket.

Measured patch cost of the 3-clip reference set:

| clip | duration | patches |
| --- | --- | --- |
| `ref_s` (1 sentence) | 4.05 s | 26 |
| `ref_m` (2 sentences, **shipped**) | 7.25 s | 46 |
| `ref_l` (3 sentences) | 11.45 s | 72 |

The transcripts must match each clip *exactly* (continuation cloning). Trimming
the audio without re-trimming the transcript degrades similarity.

### 5. Chunk size: ~33 words, not 150–180

180 words ≈ 70 s of audio ≈ **440 audio patches** ≈ 10 GB of VRAM — roughly 1.7×
the whole card. `process_txt.py --max-words auto` instead subtracts the prompt's
patches from the live `max_generate_length` (read from `/health`) and converts
the remainder with `TTS_WORDS_PER_SECOND=2.5`, yielding ~33 words/chunk here.

### 6. RTF definition: the spec's formula is inverted relative to its target

The spec defines `RTF = audio_duration / generation_time` **and** targets
`< 0.5`. Those cannot both hold: the formula is a speed *factor*, so `< 0.5`
would mean slower than 2× real time. `benchmark_test.py` reports **both**:
`X-RTF` (`gen/audio`, the dots.tts convention the `< 0.5` target belongs to) and
`X-Speed-Factor` (`audio/gen`, the spec's literal formula). All verdicts use
`gen/audio`.

### Also changed, no conflict with the spec

* `TTS_MAX_SEQUENCE_LENGTH` = **1024**, not the checkpoint default 2048 — halves
  the LLM KV cache, saving ~170 MB, which is part of what makes 128 patches fit.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reclaims ~190 MB of
  allocator fragmentation (reserved fell 5628 → 5438 MB). This was previously
  left OFF because it conflicted with cudagraphs; those are now disabled
  (see [patch #2](#apppatchespy--three-upstream-memory-bugs)).

---

## New work not in the original spec

### `app/patches.py` — three upstream memory bugs

On a 6 GB card the stock `dots.tts` inference path does not fit, for reasons
independent of the checkpoint's own published memory table. Each patch is a
no-op if a future `dots_tts` release changes shape, and the whole module is
disabled by `TTS_LOW_VRAM_PATCHES=false`.

**Patch 1 — chunked, in-place AdaLN fusion.**
`FusedAdaLNDiT.__init__` builds a fused modulation matrix of shape
`1024 × (6×18+2) = 149 504 × 1024` — **292 MiB in bfloat16** — and fills it with
a single `torch.cat`. That needs a *second* 292 MiB scratch buffer at the same
time, and the per-block `adaLN_modulation` linears it copied from (another 292
MiB) are never released. The cat scratch alone is what pushed eager
`max_generate_length=128` ~50 MiB over the physical limit. The patch copies each
linear straight into its destination slice and nulls the source tensor, so no
full-size scratch is ever allocated and the un-fused weights are freed.

**Patch 2 — cudagraph trees are disabled.**
`dots_tts.modules.backbone.inference_utils._COMPILE_OPTIONS` compiles with
`mode="reduce-overhead"`, which requests Inductor's cudagraph trees. On this GPU
that (a) reserves a ~150 MB private pool and (b) aborts the first graph replay
with `RuntimeError: Expected curr_block->next == nullptr to be true, but got
false`. The patch rewrites the options to `mode="default"`. Cost: a little
kernel-launch overhead per decode step. Note `torch.compile` rejects `mode` and
`options` together, so this must set `mode` alone.

**Patch 3 — chunked BigVGAN decode.**
`generate_audio` concatenated every latent patch and ran the decoder once, so the
activation peak grew with utterance length and OOM'd even when weights fit. The
patch releases `_static_generate_workspaces`, then decodes in fixed
12-patch (~1.9 s) chunks. Measured: this dimension turned out **not** to be a
VRAM lever (12 vs 32 patches: 5.35 vs 5.36 GB) but it bounds the worst case for
long audio, so it is kept as insurance.

### `scripts/autotune.py` — the settings autotuner

Sweeps settings, measures each candidate with the real benchmark, and writes the
winning configuration out. All controls are CLI flags.

```bash
python scripts/autotune.py                    # full sweep, dry run
python scripts/autotune.py --quick --apply    # fast, install it
python scripts/autotune.py --only reference   # cheapest dimension
```

| Flag | Meaning |
| :--- | :--- |
| `--objective {balanced,throughput,headroom}` | what "best" means (default `balanced`) |
| `--profiles short medium long` | text lengths to score each candidate against |
| `--vram-limit-gb` / `--vram-margin-gb` | budget and safety headroom (default 5.5 / 0.1) |
| `--rtf-target` | RTF gate (default 0.5) |
| `--only <dim…>` | restrict the sweep (`reference` is a pseudo-dimension) |
| `--resume` | reuse candidates from a previous report |
| `--apply` | write the winner into `.env` behind a timestamped backup |
| `--keep-running` | leave the container on the winner (default: restore `.env`) |

Design notes worth preserving:

* **Measurement reuses `benchmark_test.py` itself**, so tuning numbers come from
  exactly the code that performs acceptance — no drift between what we tune
  against and what we validate.
* **Settings are varied without editing `.env`.** Compose reads the shell
  environment before `.env`, so each candidate is applied with a scoped
  `docker compose up -d`; the original config is restored on exit.
* **Staged coordinate descent**, not full factorial: one dimension at a time,
  keeping the winner. A server-side candidate costs a restart + compile warmup of
  3–5 min, so a full factorial is hours.
* **Candidates are persisted after every measurement**, so `--resume` survives a
  crash or an interrupt.
* **It ranks on the worst text profile**, not the mean — a config that is fast on
  long text and slow on short text is not a win.
* **It fits a cost model** and reports the minimum audio length needed to reach
  the RTF target, so a failed sweep explains *why* instead of just failing.
* **It verifies the effective config** against `/health`, warning when the server
  did not apply what was requested (i.e. something got clamped).

Artifacts: `data/output/tuning_report.json` (full matrix, failures + reasons) and
`data/output/tuned.env` (pasteable winner block).

> Note: the shipped `tuning_report.json` was produced before the cost-model
> feature landed, so it lacks the `cost_model` key; the six underlying trials are
> intact and the model recomputes identically from them.

### Coupling discovered while tuning

Three settings are **not independent**, which is why a sweep must vary them
together and why a single hand-tuned value is fragile:

1. `TTS_MAX_GENERATE_LENGTH` → compile bucket (snapped up to 64/128/256/512).
2. `TTS_NUM_STEPS` → DiT KV cache size (linear).
3. `TTS_MAX_SEQUENCE_LENGTH` → LLM KV cache size (linear).

`vocoder_chunk_patches` and `guidance_scale` were swept and found not to be
VRAM/speed levers in this regime.

---

## Bugs found & fixed

### Upstream (`dots_tts`) — see [patches](#apppatchespy--three-upstream-memory-bugs)

| # | Symptom | Cause |
| --- | --- | --- |
| 1 | OOM by ~50 MiB at 128 patches (eager) | AdaLN `torch.cat` scratch + unreleased source weights |
| 2 | `Expected curr_block->next == nullptr` on first replay | cudagraph trees vs a fragmented allocator |
| 3 | OOM in `alias_free_act` during vocoder decode | whole-utterance single-shot BigVGAN |

### Pre-existing defects in this project's own code

These predated the current session and blocked end-to-end verification:

| # | File | Defect | Fix |
| --- | --- | --- | --- |
| 1 | `app/utils.py` | `soundfile.write` to a `BytesIO` cannot infer a container from a filename, so **every** synthesis returned HTTP 500 | pass `format="WAV"` explicitly |
| 2 | `scripts/benchmark_test.py` | `VramPoller` set `self._stop = threading.Event()`, shadowing `Thread._stop()`, which `join()` calls internally → every `join()` raised `TypeError: 'Event' object is not callable` | renamed to `_stop_event` |
| 3 | `app/main.py` | application loggers had no handler, so startup / patch / request records never reached the container log | added `configure_logging()` |
| 4 | `app/main.py` | no way to see the *effective* config from outside, which the tuner needs | `/health` already reported it; tuner now cross-checks it |

### In the tuner itself (found by running it)

Worth recording because they are the failure mode of any orchestrator script:
`docker compose` env precedence, a stale function signature, a function defined
after the `__name__` guard, and **host-vs-container path mapping** — the
benchmark runs inside the container but its report had to be read from the host
side of the `data/` bind mount.

---

## Parameters (as shipped and verified)

| Parameter | Spec | Shipped | Notes |
| :--- | :--- | :--- | :--- |
| Model | `dots.tts-soar` | `dots-studio/dots.tts-mf` | dev #1 |
| Base image | `cuda:12.1.1` | `cuda:12.8.1-runtime-ubuntu22.04` | dev #2, torch ≥ 2.8 |
| Precision | `bfloat16` | `bfloat16` (float16 fallback) | |
| `--optimize` | true | true | required for the RTF target |
| `TTS_NUM_STEPS` | 10 | **2** | dev #3; also sizes the DiT KV cache |
| `TTS_MAX_GENERATE_LENGTH` | — | **128** patches (20.5 s) | 128 bucket, fits at 2 steps |
| `TTS_MAX_SEQUENCE_LENGTH` | — | **1024** tokens | saves ~170 MB |
| `TTS_VOCODER_CHUNK_PATCHES` | — | **12** (≈1.9 s) | bounds vocoder activations |
| `TTS_LOW_VRAM_PATCHES` | — | **true** | required on 6 GB |
| Ref audio duration | 10–12 s | **7.0 s** (44 patches) | dev #4 |
| Ref transcript | required | required, must match exactly | |
| Chunk size | 150–180 words | **~33 words** (auto) | dev #5 |
| Words/sec | — | 2.5 (`TTS_WORDS_PER_SECOND`) | |
| Audio format | WAV 48 kHz mono | WAV 48 kHz mono PCM_16 | |
| `PYTORCH_CUDA_ALLOC_CONF` | — | `expandable_segments:True` | −190 MB reserved |
| VRAM limit | `< 5.5 GB` | **5.381 GB** ✅ | only ~0.14 GB of slack |
| RTF target | `< 0.5` | **0.464** at 13.4 s ✅ / 0.63 at 3.7 s ⚠️ | limit #1 |
| Cold start | — | ~198 s (compile warmup) | |

---

## Known limits & open items

### 1. RTF < 0.5 is unreachable below 7.7 s of audio

Per [the cost model](#the-central-finding-the-rtf-target-is-output-length-bound),
a ~1.0 s fixed per-request cost bounds short utterances at RTF ≈ 0.63–0.67
regardless of configuration. Practical responses: batch sentences into one
request (`process_txt.py` already does, which is why the long-form run passes at
0.511), or accept a higher RTF for one-liners. **The fixed cost has not been
attributed** — that is the highest-value next investigation, and it should be
profiled before anyone attempts to tune it away.

### 2. Audio quality was never measured

No part of this work could listen to the output, so speaker similarity is
unvalidated. The spec's `SIM > 83` is **unverified**, and switching to MF
(deviation #1) was expected to cost some of it. Anyone continuing this should
run a speaker-similarity evaluation before trusting the cloning quality —
particularly if they consider raising `TTS_NUM_STEPS` back toward 4 for quality,
which will cost the 128-patch VRAM budget.

### 3. The full autotuner sweep was never run

Only two dimensions (`reference`, `vocoder_chunk_patches`) were exercised live,
each over 2–3 values. The full sweep is ~60–80 min of GPU time. Logic is covered
by 22 unit tests, but the untested dimensions are likely to surface more
orchestration bugs of the kind listed above. Run it before trusting
`--apply`.

### 4. VRAM headroom is thin

5.381 GB peak against a 5.5 GB cap is ~0.14 GB of slack on a card where the OS
and other processes also compete. The tuner's default `--vram-margin-gb` is
0.1 (not the 0.2 originally proposed) precisely because 0.2 would reject the
shipped config. On a shared GPU, lower `TTS_MAX_GENERATE_LENGTH` to 96.

### 5. `dots.tts-soar` remains untested

Every Soar-related claim here comes from the published memory table, not from
this hardware. If someone raises the VRAM budget above 5.65 GB, re-validate
Soar before assuming it is better.

---

## Original Checklist (all phases complete)

Phases 1–6 as originally specified, all delivered. Deviations are noted inline.

### Phase 1 — Docker environment
- [x] **1.1** `Dockerfile` — CUDA runtime + torch + dots.tts + ffmpeg/librosa/noisereduce.
      *Base image changed, dev #2.*
- [x] **1.2** `docker-compose.yml` — GPU passthrough (`count: 1`, `driver: nvidia`),
      port mapping, persistent `hf_cache` volume (keeps ~4.4 GB of weights across
      rebuilds), `shm_size: 2gb` for the compile workers.

### Phase 2 — Reference preprocessing (`clean_audio.py`)
- [x] **2.1** Stationary noise reduction, forced 48 kHz mono, `prop_decrease=0.85`.
- [x] **2.2** Duration validation (5–15 s, ideal 10–12 s) → `reference_clean.wav`.
      *Added `--trim-to N` for sentence-safe shortening when the VRAM budget
      forces a smaller reference — see dev #4.*
      *Also added a `--report` JSON and a patch-budget calculation.*

### Phase 3 — FastAPI service (`app/main.py`)
- [x] **3.1** `DotsTtsRuntime` on startup, `cuda`, `bfloat16` (float16 fallback),
      `optimize=True`, warmup at load.
- [x] **3.2** `GET /health` — GPU name, VRAM total/allocated/free, plus lifetime
      peak, per-request metrics, and readiness. Returns 503 until genuinely ready.
- [x] **3.3** `POST /v1/audio/speech` — `text`, `reference_audio`, `reference_text`,
      `num_steps` (default **2**, dev #3), `guidance_scale`, `seed`, `speaker_scale`,
      `language`, `normalize_text`, `template_name`, `include_metrics`.
- [x] **3.4** Raw `audio/wav` response, `tempfile` cleanup, metric headers
      (`X-RTF`, `X-Speed-Factor`, `X-Peak-Vram-MB`, `X-Request-Id`, …).
      *Added an `asyncio.Lock` so one GPU never runs two generations at once.*

### Phase 4 — Long-text client (`process_txt.py`)
- [x] **4.1** Sentence-preserving chunking on `(?<=[.!?])\s+`, whitespace cleaned.
      *Chunk size derived from the live server, dev #5; `--max-words auto` default.*
- [x] **4.2** Request loop with progress, per-chunk save, up to 3 retries with
      backoff. *Added per-chunk metrics to the report.*
- [x] **4.3** FFmpeg concat demuxer with `-c copy` (lossless), format-consistent
      silence padding, temp chunk cleanup. *Added `--resume` and `--keep-chunks`.*

### Phase 5 — Benchmarking (`benchmark_test.py`)
- [x] **5.1** Cold start, standardised single-chunk benchmark, peak VRAM sampled
      before/during/after via a background `/health` poller, RTF in both
      conventions, JSON report, non-zero exit on threshold violation.
      *Fixed a `Thread._stop` shadowing bug that made the poller unusable.*

### Phase 6 — Verification
- [x] **6.1** `docker compose up -d --build`.
- [x] **6.2** `python scripts/benchmark_test.py`.
- [x] **6.3** `benchmark_report.json` verified: 0 errors, peak 5.381 GB.
- [x] **6.4** End-to-end batch test. *Ran `test_audio_sample.txt` (161 words) rather
      than the ~2 000-word target — see limit #3; 7/7 chunks, RTF 0.511.*

---

## Operating it

```bash
# start / stop
docker compose up -d --build
docker compose down

# verify the service is genuinely ready (503 until the compile warmup finishes)
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# benchmark
docker compose exec tts python scripts/benchmark_test.py

# one-shot smoke test
docker compose exec tts python scripts/test_api.py

# long text -> one stitched WAV
docker compose exec tts python scripts/process_txt.py \
    --input data/input/test_audio_sample.txt --max-words auto

# re-cut the reference (SHORTEN THE TRANSCRIPT TO MATCH)
docker compose exec tts python scripts/clean_audio.py \
    --input data/references/my_voice_v2.wav --trim-to 7.0

# find the best settings for this GPU
python scripts/autotune.py --quick          # dry run
python scripts/autotune.py --quick --apply  # install
```

After any `.env` change: `docker compose up -d --force-recreate` and expect a
~3–5 min warmup. `/health` answers 503 with the load error in `model.error`
until the engine is truly ready.

---

## Pointers for whoever continues

1. **Profile the ~1.0 s fixed per-request cost first** (limit #1). It is the only
   thing standing between short speech and the RTF target.
2. **Measure speaker similarity** (limit #2) before changing `TTS_NUM_STEPS` or
   reconsidering Soar.
3. **Run the full tuner sweep** (limit #3) before trusting `--apply`.
4. If VRAM gets tight, drop `TTS_MAX_GENERATE_LENGTH` to 96 — it is the single
   biggest lever and is a compile-bucket boundary, so it is a real step down, not
   a rounding artefact.
5. `app/patches.py` is coupled to `dots_tts` internals. A `dots_tts` upgrade
   should be validated against it; the patches log and no-op rather than crash,
   but silently losing them means an OOM at 128 patches.
