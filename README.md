# Local High-Fidelity TTS — dots.tts + FastAPI + Docker (v2)

Containerised, GPU-accelerated text-to-speech with **continuation voice cloning**
on a laptop RTX 4050 (6 GB). Nothing is installed on the host except Docker.

* **Engine** — [`dots.tts`](https://github.com/studio-dots-ai/dots.tts) 2B continuous-AR TTS
  (`dots-studio/dots.tts-mf`, 48 kHz AudioVAE), bfloat16 + `torch.compile`.
* **Service** — FastAPI `POST /v1/audio/speech` (raw `audio/wav`) + `GET /health`
  with live VRAM telemetry.
* **Pipeline** — `clean_audio.py` → `process_txt.py` → `benchmark_test.py`.

---

## 1. Quick start

```bash
cp .env.example .env                 # optional: tune model / thresholds
docker compose up -d --build         # first build downloads ~8 GB (CUDA + torch)

# wait for the engine (first run also downloads ~4.4 GB of weights and runs a
# torch.compile warmup — several minutes on a laptop GPU)
docker compose logs -f tts

# 1) clean + validate the reference voice
docker compose exec tts python scripts/clean_audio.py \
    -i data/references/my_voice_v2.wav \
    --transcript data/references/my_voice_v2.txt

# 2) benchmark cold start / peak VRAM / RTF  (spec Phase 5)
docker compose exec tts python scripts/benchmark_test.py

# 3) long-form batch synthesis
docker compose exec tts python scripts/process_txt.py \
    -i data/input/test_audio_sample.txt
```

Then open <http://127.0.0.1:8000/docs> for the OpenAPI UI and
`data/output/benchmark_report.json` for the benchmark verdict.

---

## 2. Layout

```text
.
├── Dockerfile                 # CUDA 12.8 runtime + torch cu128 + dots.tts
├── docker-compose.yml         # GPU reservation, port 8000, hf_cache volume
├── .env.example               # every tunable (copied to .env)
├── requirements.txt
├── app/
│   ├── config.py              # env-driven settings (torch-free)
│   ├── patches.py             # low-VRAM patches for dots.tts (6 GB cards)
│   ├── utils.py               # GPU telemetry, WAV buffers, temp files
│   └── main.py                # FastAPI service + dots.tts runtime loader
├── scripts/
│   ├── clean_audio.py         # denoise + validate the reference clip
│   ├── process_txt.py         # chunk -> POST -> FFmpeg concat
│   ├── benchmark_test.py      # cold start / peak VRAM / RTF benchmark
│   ├── tune_settings.py       # autotuner: sweep settings, emit the best .env
│   └── test_api.py            # one-shot sanity check
└── data/
    ├── references/            # raw reference .wav + transcript .txt
    ├── input/                 # input .txt + reference_clean.wav + reference_set.json
    ├── temp_chunks/           # intermediate chunk WAVs + file_list.txt
    └── output/                # final WAVs + benchmark/tuning reports
```


---

## 3. API

### `GET /health`

`200` once the checkpoint is resident, `503` while loading (so the Docker health
check waits for the *engine*, not just the ASGI process).

```json
{
  "status": "ok",
  "ready": true,
  "model": {
    "id": "dots-studio/dots.tts-mf",
    "precision": "bfloat16",
    "optimize": true,
    "max_generate_length": 128,
    "sample_rate": 48000,
    "audio_patch_seconds": 0.16,
    "cold_start_seconds": 412.7
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "vram_total_mb": 6141,
    "vram_allocated_mb": 5310.4,
    "vram_free_mb": 726.9,
    "vram_used_mb": 5414.1,
    "vram_reserved_mb": 5450.0,
    "vram_peak_allocated_mb": 5450.0
  },
  "vram_limit_gb": 5.5,
  "vram_within_limit": true,
  "lifetime_peak_vram_mb": 5450.0
}
```

### `POST /v1/audio/speech` (`multipart/form-data`)

| field | type | required | default |
|---|---|---|---|
| `text` | str | yes | — |
| `reference_audio` | file | yes | — |
| `reference_text` | str | yes | — (exact transcript, continuation cloning) |
| `num_steps` | int | no | `2` (MeanFlow native; also sizes the DiT KV cache) |
| `guidance_scale` | float | no | `1.2` |
| `seed` | int | no | artifact default |
| `speaker_scale` | float | no | `1.5` |
| `language` | str | no | `none` (`IT`, `EN`, `auto_detect`, …) |
| `normalize_text` | bool | no | `false` |
| `template_name` | str | no | `tts` |
| `include_metrics` | bool | no | `false` → JSON envelope with base64 audio |

Returns **`audio/wav`** plus metric headers:

```
X-RTF: 0.4410            # generation / audio  (lower is better)
X-Engine-RTF: 0.4382
X-Speed-Factor: 2.27     # audio / generation  (the spec's literal formula)
X-Audio-Seconds, X-Generation-Seconds, X-Peak-VRAM-MB, X-Sample-Rate, X-Request-Id
```

```bash
curl -s -o out.wav -D - -X POST http://127.0.0.1:8000/v1/audio/speech \
  -F "text=Ciao, questo è un test di sintesi vocale." \
  -F "reference_audio=@data/input/reference_clean.wav" \
  -F "reference_text=$(cat data/input/reference_clean.txt)"
```

Every route is also exercised programmatically by `scripts/test_api.py` and
`scripts/benchmark_test.py` (which is the reference multipart client).

---

## 4. Two spec numbers that are physically impossible on a 6 GB card

The v2 spec contains mutually exclusive requirements. They are documented here
rather than silently ignored.

1. **Model / VRAM.** The spec asks for `dots.tts-soar` **and** peak VRAM
   `< 5.5 GB`. The official dots.tts memory table (bfloat16, `--optimize`) shows
   SOAR's *smallest* bucket already costs **5.65 GB** — it cannot fit a 5.5 GB
   budget for any input. `dots.tts-mf` (the MeanFlow student of SOAR) costs
   **5.30 GB** (`<64` patches) / **5.47 GB** (`<128` patches), so
   `TTS_MODEL_NAME` defaults to `dots-studio/dots.tts-mf`. Set
   `TTS_MODEL_NAME=dots-studio/dots.tts-soar` if you accept ~5.65 GB (and a
   slower RTF) in exchange for the highest speaker similarity.

2. **Chunk size.** The spec asks for 150–180-word chunks. 180 words ≈ 70 s of
   audio ≈ **440 audio patches**, which needs ~10 GB of VRAM. On 6 GB the hard
   ceiling is `max_generate_length` = 128 patches ≈ **20.5 s of audio per
   request, reference prompt included**. `process_txt.py --max-words auto`
   therefore derives the word budget from the live server: it subtracts the
   prompt's patches from `max_generate_length` and converts the remainder to
   words (`TTS_WORDS_PER_SECOND`, default 2.5).

   Consequence: a ~12 s reference leaves ~8 s of generation per call (~20 words);
   the shipped 7 s `data/input/reference_clean.wav` allows ~13 s of generation per
   call, which `--max-words auto` reports as ~33 words. `clean_audio.py --trim-to`
   exists for exactly this reason — remember to shorten the transcript to match.

4. **`num_steps` is coupled to the memory budget.** The DiT KV cache is allocated
   as `(num_steps, layers, heads, tokens, dim)`, so every extra sampling step
   doubles it. `dots.tts-mf` is a MeanFlow model whose native step count is **2**,
   and the shipped configuration uses 2: 128 patches then peak at 5.36 GB. At 4
   steps the same 128-patch budget OOMs — drop `TTS_MAX_GENERATE_LENGTH` to 96 if
   you raise `TTS_NUM_STEPS`.

5. **Three upstream memory bugs had to be patched.** See `app/patches.py`. On a
   6 GB card the stock path does not fit at all, for reasons independent of the
   checkpoint's own published memory table:

   * `FusedAdaLNDiT.__init__` builds a 292 MiB fused modulation matrix with a
     single `torch.cat`, which needs a second 292 MiB scratch buffer *and* never
     releases the per-block linears it copied from.
   * `torch.compile(mode="reduce-overhead")` (what `dots_tts` uses) reserves a
     ~150 MiB cudagraph pool and, on this GPU, aborts the first graph replay with
     `Expected curr_block->next == nullptr to be true`.
   * `generate_audio` decodes the whole utterance through BigVGAN in one shot, so
     the decoder's activation peak grows with the audio length.

   All three patches are no-ops if a future `dots_tts` release changes shape, and
   `TTS_LOW_VRAM_PATCHES=false` disables them.

3. **Base image.** The spec asked for `nvidia/cuda:12.1.1` + torch cu121.
   `dots.tts 0.3.x` requires `torch>=2.8`, and PyTorch stopped publishing cu121
   wheels after 2.5.x. The image therefore uses
   `nvidia/cuda:12.8.1-runtime-ubuntu22.04` with cu128 wheels — the stack
   dots.tts itself benchmarks on. Host drivers ≥ 525 work (this machine reports
   CUDA 13.2 and passes the GPU through fine).



---

## 5. Configuration

Everything is environment driven; see `.env.example` for the full annotated list.
The most useful knobs:

| variable | default | meaning |
|---|---|---|
| `TTS_MODEL_NAME` | `dots-studio/dots.tts-mf` | checkpoint |
| `TTS_PRECISION` | `bfloat16` | falls back to `float16` if BF16 is unsupported |
| `TTS_OPTIMIZE` | `true` | `torch.compile` (warmup at load, faster steady state) |
| `TTS_WARMUP_ON_OPTIMIZE` | `true` | run the compile warmup during startup |
| `TTS_MAX_GENERATE_LENGTH` | `128` | **total** prompt + generated audio patches (coupled to `TTS_NUM_STEPS`) |
| `TTS_MAX_SEQUENCE_LENGTH` | `1024` | LLM KV-cache length in tokens |
| `TTS_NUM_STEPS` | `2` | sampling steps; also sets the DiT KV-cache size |
| `TTS_LOW_VRAM_PATCHES` | `true` | enable the `app/patches.py` memory patches (required on 6 GB) |
| `TTS_VOCODER_CHUNK_PATCHES` | `12` | latent patches per BigVGAN call |
| `TTS_VRAM_LIMIT_GB` | `5.5` | benchmark acceptance threshold |
| `TTS_RTF_TARGET` | `0.5` | benchmark acceptance threshold (`gen/audio`) |
| `TTS_CHUNK_MAX_WORDS` | `35` | fallback chunk size when `--max-words auto` cannot resolve |

---

## 6. Finding the best settings for *your* GPU

`scripts/tune_settings.py` sweeps the runtime settings, measures each candidate
with the real benchmark, and writes the winning configuration out.

```bash
python scripts/tune_settings.py                    # full sweep, dry run
python scripts/tune_settings.py --quick --apply    # fast, and install it
python scripts/tune_settings.py --only reference   # cheapest dimension (~5 min)
```

Everything is configurable: `--objective {balanced,throughput,headroom}`,
`--profiles`, `--vram-limit-gb`, `--vram-margin-gb`, `--rtf-target`,
`--only`, `--resume`. Artifacts land in `data/output/`:

* `tuning_report.json` — every candidate, failures and reasons included.
* `tuned.env` — the winner as a pasteable block (`--apply` writes it into `.env`
  behind a timestamped backup).

**What it sweeps.** The reference clip, then `TTS_MAX_GENERATE_LENGTH`,
`TTS_MAX_SEQUENCE_LENGTH`, `TTS_NUM_STEPS`, `TTS_VOCODER_CHUNK_PATCHES`,
`TTS_OPTIMIZE`, `TTS_LOW_VRAM_PATCHES`, `TTS_GUIDANCE_SCALE`. It uses staged
coordinate descent (one dimension at a time, keeping the winner) rather than a
full factorial, because each server-side candidate costs a restart plus a
`torch.compile` warmup of 3–5 min. Candidates are persisted after every
measurement, so `--resume` picks up where a crashed run stopped.

Two things it deliberately guards against:

* **It measures across several text lengths and ranks on the worst one.** RTF
  depends strongly on output length, so a config tuned on one sentence can look
  fine and be terrible for short utterances.
* **It fits a cost model** `generation = fixed + rate × audio_seconds` and
  reports the minimum audio length needed to hit the RTF target. When a sweep
  fails, this says whether a better setting exists or the target is simply out
  of reach for the workload — which is usually the real answer.

Measurement reuses `scripts/benchmark_test.py` itself, so tuning numbers come
from exactly the code that performs acceptance. Varying settings does not touch
`.env`: Compose reads the shell environment first, so each candidate is applied
with a scoped `docker compose up -d`, and the original configuration is restored
at the end unless `--keep-running` is passed.

### Measured behaviour on the RTX 4050 (6 GB)

The tuned model is `generation ≈ 0.98 s + 0.373 × audio_seconds`, i.e.
`RTF ≈ 0.373 + 0.98 / audio_seconds`:

| audio length | RTF |
|---|---|
| 3.4 s | 0.65 |
| 9.1 s | 0.48 |
| 13.3 s | 0.45 |
| 16.5 s | 0.43 |

So **RTF ≤ 0.5 needs ≳ 7.7 s of audio per request on this GPU.** The ~1 s
per-request fixed cost is independent of the reference length (measured across
26/46/72-patch references), so it cannot be tuned away — short utterances are
bounded by it. For short text either accept a higher RTF or batch sentences into
one request; `process_txt.py` already batches by default.

---

## 7. Operational notes

* **Cold start** is dominated by `torch.compile`: the warmup walks every reachable
  DiT bucket and vocoder chunk size. Expect minutes, not seconds. The Docker
  health check allows up to an hour (`--start-period=3600s`) and keeps returning
  `503` until the engine is genuinely ready.
* **The `hf_cache` volume** (`local_tts_hf_cache`) keeps the ~4.4 GB of weights
  across rebuilds — only delete it if you want to re-download.
* **Changing `TTS_MAX_GENERATE_LENGTH`** needs `docker compose up -d` (it is a
  load-time parameter) and triggers a fresh compile warmup.
* **Concurrency** is serialised by an `asyncio.Lock`: one 6 GB GPU cannot serve
  two generations at once without spilling into system RAM.
* **VRAM numbers** come from `torch.cuda.mem_get_info()` (whole device) and
  `torch.cuda.memory_allocated()` / `max_memory_allocated()` (this process),
  exactly as the spec requests.
* **FFmpeg stitching** uses `-c copy` over format-consistent PCM_16 chunks, so it
  is lossless; a small silence gap (`--gap`, default 0.12 s) is inserted between
  chunks to avoid clipped prosody at the seams.
* **`benchmark_test.py`** reports the RTF in both conventions
  (`generation/audio` for the `< 0.5` target and `audio/generation` for the spec's
  literal formula) and exits non-zero when a threshold is violated.

---

## 8. Troubleshooting

| symptom | fix |
|---|---|
| `could not select device driver "nvidia"` | install `nvidia-container-toolkit`, then `sudo systemctl restart docker` |
| `CUDA out of memory` | lower `TTS_MAX_GENERATE_LENGTH` (e.g. 96), use a shorter `--trim-to` reference, keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `/health` stays `loading` for a long time | expected on first run (weight download + compile warmup); watch `docker compose logs -f tts` |
| `HTTP 400 … max_generate_length must exceed prompt audio patch count` | reference too long → `clean_audio.py --trim-to 6.5` |
| audio sounds truncated | the request hit `max_generate_length`; lower `--max-words` or raise the cap after measuring real headroom |
| `process_txt.py` reports 0 usable chunks | the input `.txt` is empty after whitespace normalisation |

