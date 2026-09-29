# VoxDock — containerised high-fidelity local TTS

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
#    Record 5-15 s of your own voice and put it here:
#      cp ~/my_voice.wav data/references/my_voice.wav
docker compose exec tts python scripts/clean_audio.py \
    -i data/references/my_voice.wav \
    --transcript data/references/my_voice.txt

# 2) benchmark cold start / peak VRAM / RTF
docker compose exec tts python scripts/benchmark_test.py

# 3) long-form batch synthesis
docker compose exec tts python scripts/process_txt.py \
    -i data/input/test_audio_sample.txt

# 4) does it actually sound like the speaker?
#    RTF and VRAM cannot answer this; ECAPA-TDNN cosine similarity can.
docker compose exec tts python scripts/speaker_similarity.py \
    --reference data/references/my_voice_v2.wav \
    --generated data/output/test_audio_sample.wav
```

The transcripts under `data/references/` and `data/input/` are committed, but
**the `.wav` files are not** — ignored on purpose, because a voice is personal
and large. Record your own reference; everything else works unchanged. For
continuation cloning the transcript must match the audio word for word.

Then open <http://127.0.0.1:8000/docs> for the OpenAPI UI,
**<http://127.0.0.1:8000/ui> for a usable web interface**, and
`data/output/benchmark_report.json` for the benchmark verdict.

### The web UI (`/ui`)

A single self-contained HTML page — no build step, no Node, no second service —
that drives the same public API. It lets you register a voice, pick it from a
dropdown (so the clip is not re-uploaded per request), type text and play the
result, and watch VRAM, RTF and request count live. The VRAM bar is drawn
against the **configured budget**, not the physical total, because keeping
allocations under that line is what `app/patches.py` exists for.

Long text is refused rather than silently truncated: the page mirrors
`resolve_max_words()` from `process_txt.py` and points you at the CLI, which
chunks and stitches properly.

It has **no authentication of its own** — it is a client for an API that also
has none. The port is published on `127.0.0.1` by default for that reason; do
not change `TTS_BIND_HOST` without putting something in front of it.

> The page is verified by hand, not by the test suite. `tests/test_ui.py` checks
> what breaks *silently* — endpoint paths, DOM ids, escaping — but exercising
> the actual flows needs a browser and a GPU, which the host-side suite has
> neither of.

`process_txt.py` needs no flags to get the full v3 behaviour: it registers the
voice once and reuses the id, inserts 300 ms between chunks, normalises the
finished file to −16 LUFS (EBU R128), halves any chunk that hits an OOM and
retries it, and takes its chunk size from `best_config.json` when one exists.
Use `--no-speaker-cache`, `--no-normalize`, `--lufs` or `--max-words` to opt out
or override.

---

## 2. Layout

Project identity is declared once, in `app/__init__.py`:

| constant | value | used for |
| :--- | :--- | :--- |
| `__title__` | `VoxDock` | FastAPI `/docs` title |
| `__package_name__` / `IMAGE_NAME` | `voxdock-tts` | Docker image, `voxdock-tts:latest` |
| `CONTAINER_NAME` | `voxdock-tts` | container name (`docker logs voxdock-tts`) |
| `COMPOSE_PROJECT` | `voxdock` | compose project namespace |
| `LOG_NAMESPACE` | `voxdock` | our logger namespace (`voxdock.api`, …) |
| `HF_CACHE_VOLUME` | `local_tts_hf_cache` | the weight cache volume |
| `INDUCTOR_CACHE_VOLUME` | `voxdock_inductor_cache` | the torch.compile kernel cache volume |

Docker Compose cannot read Python, so `docker-compose.yml` repeats these as
inline `${VAR:-default}` fallbacks. To catch drift:

```bash
python scripts/check_identity.py     # exits non-zero if the two disagree
```

`HF_CACHE_VOLUME` is deliberately *not* renamed: Docker cannot rename a volume,
and this one holds ~4.4 GB of checkpoint weights that would otherwise be
re-downloaded.

```text
.
├── Dockerfile                 # CUDA 12.8 runtime + torch cu128 + dots.tts
├── docker-compose.yml         # GPU reservation, non-root user, 3 volumes
├── .env.example               # every tunable (copied to .env)
├── requirements.txt
├── pytest.ini
├── app/
│   ├── config.py              # settings: env -> best_config.json -> defaults
│   ├── patches.py             # low-VRAM patches for dots.tts (6 GB cards)
│   ├── speakers.py            # speaker registry (register once, reuse by id)
│   ├── utils.py               # GPU telemetry, WAV buffers, temp files
│   ├── static/index.html      # the /ui web page (self-contained, no build)
│   └── main.py                # FastAPI service + dots.tts runtime loader
├── scripts/
│   ├── clean_audio.py         # denoise + validate the reference clip
│   ├── process_txt.py         # chunk -> POST -> FFmpeg concat -> loudnorm
│   ├── benchmark_test.py      # cold start / peak VRAM / RTF benchmark
│   ├── autotune.py            # sweep settings -> best_config.json
│   ├── speaker_similarity.py  # ECAPA-TDNN cosine similarity vs a reference
│   ├── profile_request.py     # attribute the per-request fixed cost
│   ├── latency_report.py      # cold start / first request / steady state
│   ├── check_identity.py      # assert compose matches app/__init__.py
│   └── test_api.py            # smoke test; --assert for CI
├── tests/                     # 117 host-side tests, no GPU needed (~0.1 s)
│   ├── conftest.py            # fixtures: isolated env, reloadable config
│   ├── test_config.py         # the three-tier settings chain
│   ├── test_speakers.py       # registry CRUD + path-traversal defence
│   ├── test_process_txt.py    # chunking, OOM split ordering, loudnorm parsing
│   ├── test_autotune.py       # the winning-profile artefact
│   ├── test_patches.py        # the low-VRAM patch guard
│   ├── test_api_smoke.py      # the --assert smoke test
│   ├── test_ui.py             # /ui asset: endpoints, DOM ids, escaping
│   └── test_identity.py       # compose vs app/Dockerfile identity drift guard
└── data/
    ├── references/            # raw reference .wav + transcript .txt
    ├── speakers/              # registered voices (id -> .wav + .txt), gitignored
    ├── input/                 # input .txt + reference_clean.wav + reference_set.json
    ├── temp_chunks/           # intermediate chunk WAVs + file_list.txt
    └── output/                # final WAVs + benchmark/tuning/best_config JSON
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
| `reference_audio` | file | yes\* | — |
| `reference_text` | str | yes\* | — (exact transcript, continuation cloning) |
| `speaker_id` | str | no | reuse a voice from `/v1/speakers/register` |
| `num_steps` | int | no | `2` (MeanFlow native; also sizes the DiT KV cache) |
| `guidance_scale` | float | no | `1.2` |
| `seed` | int | no | artifact default |
| `speaker_scale` | float | no | `1.5` |
| `language` | str | no | `none` (`IT`, `EN`, `auto_detect`, …) |
| `normalize_text` | bool | no | `false` |
| `template_name` | str | no | `tts` |
| `include_metrics` | bool | no | `false` → JSON envelope with base64 audio |

\* `reference_audio` / `reference_text` become optional when `speaker_id`
names a registered voice; `speaker_id` wins if both are supplied.

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

A CUDA out-of-memory is reported as **HTTP 503** with
`{"error": "out_of_memory", "retryable": true}` rather than a generic 500, so
a client can tell "retry smaller" apart from "this is broken".

### Registering a voice once (`POST /v1/speakers/register`)

`dots.tts` already caches prompt latents internally, so the cost that actually
repeats is the client re-uploading a ~656 KB WAV with every chunk. Registering
the voice once removes that:

```bash
# register once -> 201, returns an id
curl -s -X POST http://127.0.0.1:8000/v1/speakers/register \
  -F "reference_audio=@data/input/reference_clean.wav" \
  -F "reference_text=$(cat data/input/reference_clean.txt)" \
  -F "speaker_id=riccardo"

# then every request carries ~20 bytes instead of 656 KB
curl -s -o out.wav -X POST http://127.0.0.1:8000/v1/audio/speech \
  -F "text=Questo non carica alcun audio." -F "speaker_id=riccardo"
```

| route | purpose |
|---|---|
| `POST /v1/speakers/register` | store a voice, returns its id |
| `GET /v1/speakers` | list voices with a per-voice reuse counter |
| `DELETE /v1/speakers/{id}` | forget a voice |
| `POST /v1/admin/clear-cache` | drop prompt latents + empty the CUDA allocator |

`scripts/process_txt.py` does this automatically, and falls back to per-chunk
upload if the server is older or registration fails.

Every route is also exercised programmatically by `scripts/test_api.py` and
`scripts/benchmark_test.py` (which is the reference multipart client).

---

## 4. Spec numbers that are physically impossible on a 6 GB card

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
   the 7 s `data/input/reference_clean.wav` that `clean_audio.py` writes allows
   ~13 s of generation per call, which `--max-words auto` reports as ~33 words.
   `clean_audio.py --trim-to` exists for exactly this reason — remember to
   shorten the transcript to match. (The WAV is generated locally, not
   committed: `data/**/*.wav` is gitignored.)

3. **`num_steps` is coupled to the memory budget.** The DiT KV cache is allocated
   as `(num_steps, layers, heads, tokens, dim)`, so every extra sampling step
   doubles it. `dots.tts-mf` is a MeanFlow model whose native step count is **2**,
   and the tuned configuration uses 2.

   Earlier revisions of this document said 4 steps OOMs at 128 patches. **That is
   no longer true**: the tuned `vocoder_chunk_patches=32` (up from 12) freed
   enough headroom that 4 steps now fits, measured at 5518 MB peak. It costs
   +17 % generation time and buys +0.021 speaker similarity — see §6. The
   coupling is real; whether it binds at 4 steps on this card no longer does.

4. **`model_variant` is a startup setting, not a per-request parameter.** The
   spec asks for a per-request model/variant switch. Two checkpoints cannot
   co-exist in 6 GB — the weights alone are 4.55 GiB each — so switching variant
   would mean evicting and reloading the model inside a request.
   `TTS_MODEL_NAME` is therefore read once at startup:

   ```bash
   TTS_MODEL_NAME=dots-studio/dots.tts-soar docker compose up -d
   ```

5. **Speaker similarity has no absolute pass/fail threshold.** The spec cites
   "SIM > 83", a percentage on a different model's scale. The cosine similarity
   measured in §6 is a different metric on a different scale, meaningful only
   against its two controls, so this project asserts no threshold for it.


6. **Three upstream memory bugs had to be patched.** See `app/patches.py`. On a
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

   Each patch is a no-op if a future `dots_tts` release changes shape, so startup
   now *asserts* they applied rather than logging and continuing: on this card a
   missing patch means the configuration no longer fits, and a silent warning
   would surface minutes later as a confusing OOM. `TTS_LOW_VRAM_PATCHES=false`
   disables them and logs a warning saying to expect an OOM.

7. **Base image.** The spec asked for `nvidia/cuda:12.1.1` + torch cu121.
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

### Where each value comes from

Every setting resolves in this order, and the first hit wins:

1. **environment variable** — always, so you can override for a single run
2. **`data/output/best_config.json`** — the profile `scripts/autotune.py`
   measured on *this* host
3. **built-in default** — the values verified on the 6 GB RTX 4050

A missing, unreadable or wrongly-typed `best_config.json` is never fatal: the
service falls back and logs a warning, because a tuning artefact should not be
able to stop the service from starting. `GET /health` reports which tier is in
force under `config_source`, so you can always tell whether a measured profile
is actually being applied.

---

## 6. Finding the best settings for *your* GPU

`scripts/autotune.py` sweeps the runtime settings, measures each candidate
with the real benchmark, and writes the winning configuration out.

```bash
python scripts/autotune.py                    # full sweep, dry run
python scripts/autotune.py --quick --apply    # fast, and install it
python scripts/autotune.py --only reference   # cheapest dimension (~5 min)
```

> **Note.** The sweep restarts the container many times (3–5 min per
> server-side candidate, mostly `torch.compile` warmup), and now also downloads
> `dots.tts-soar` (~2 GB) to measure it. Expect the full run to take well over
> an hour, and expect `soar` to be ruled out on a 6 GB card.

Everything is configurable: `--objective {balanced,throughput,headroom}`,
`--profiles`, `--vram-limit-gb`, `--vram-margin-gb`, `--rtf-target`,
`--only`, `--resume`. Artifacts land in `data/output/`:

* `tuning_report.json` — every candidate, failures and reasons included.
* `best_config.json` — the winner as the service reads it (typed values, host
  description, constraints, and the model comparison). Consumed by
  `app/config.py` at the next container start.
* `tuned.env` — the winner as a pasteable block (`--apply` writes it into `.env`
  behind a timestamped backup).

**What it sweeps.** `TTS_MODEL_NAME` (`dots.tts-mf` vs `dots.tts-soar`) and
`TTS_PRECISION` first — which checkpoint is feasible decides the envelope
everything else is tuned inside — then the reference clip,
`TTS_MAX_GENERATE_LENGTH`, `TTS_MAX_SEQUENCE_LENGTH`, `TTS_NUM_STEPS`,
`TTS_VOCODER_CHUNK_PATCHES`, `TTS_OPTIMIZE`, `TTS_LOW_VRAM_PATCHES`,
`TTS_GUIDANCE_SCALE`. It uses staged coordinate descent (one dimension at a
time, keeping the winner) rather than a full factorial, because each
server-side candidate costs a restart plus a `torch.compile` warmup of 3–5 min.
Candidates are persisted after every measurement, so `--resume` picks up where a
crashed run stopped.

The **soar-vs-mf comparison table** keeps the checkpoints that were ruled out
along with the reason, because "soar cannot run on this GPU" is the most
useful single result of the sweep:

```
[tune] model comparison (v3 Task 4.3)
  checkpoint                          feasible  best RTF    min peak
  -------------------------------- ----------- --------- -----------
  dots-studio/dots.tts-mf          1/1            0.4638     5381 MB   <- selected
  dots-studio/dots.tts-soar        0/1                 -           -
  ruled out: dots-studio/dots.tts-soar never fit in the 5.5 GB peak-VRAM budget
```

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

The tuned model is `generation ≈ 0.79 s + 0.413 × audio_seconds`, i.e.
`RTF ≈ 0.413 + 0.79 / audio_seconds` (fitted over 34 calls during the sweep):

| audio length | RTF |
|---|---|
| 3.5 s | 0.64 |
| 9.1 s | 0.50 |
| 13.4 s | 0.47 |

So **RTF ≤ 0.5 needs ≳ 9.1 s of audio per request on this GPU.** The 0.79 s
per-request fixed cost is independent of the reference length (measured across
26/46/72-patch references), so it cannot be tuned away — short utterances are
bounded by it. For short text either accept a higher RTF or batch sentences into
one request; `process_txt.py` already batches by default.

#### Where the fixed cost actually goes

`scripts/profile_request.py` profiles one request with `torch.profiler` and
attributes the CUDA time by op kind (short text, 2 steps, 7 s reference):

| phase | self CUDA time | share |
|---|---:|---:|
| other (elementwise, reductions, unclassified) | 1374.5 ms | 39.6% |
| linear layers (LLM + DiT) | 1216.7 ms | 35.0% |
| vocoder — BigVGAN convolution | 461.7 ms | 13.3% |
| **GPU idle (host launch gap)** | **224.1 ms** | **6.5%** |
| attention (LLM + DiT) | 183.6 ms | 5.3% |
| normalisation | 11.6 ms | 0.3% |
| host↔device transfer | 1.1 ms | 0.0% |

The 6.5% `Command Buffer Full` row is the interesting one: time when the GPU
had nothing queued and was waiting on the host to launch work. That suggests
the fixed cost is partly **host-bound**, not GPU-bound — so raising
`OMP_NUM_THREADS`, or re-enabling CUDA graphs, would attack it where lowering
`num_steps` or shortening the reference would not.

Treat it as a lead rather than a conclusion: it is only 6.5%, and the bulk is
real compute spread across many small elementwise ops. Run the script for the
full operator table rather than trusting the summary above.

```bash
docker compose stop tts
docker compose run --rm --entrypoint python tts \
    scripts/profile_request.py --report data/output/profile_report.json
docker compose up -d
```

The service must be stopped first: the script loads its own copy of the
checkpoint, and two copies of a 4.5 GiB model do not fit in 6 GB.

The tuned configuration (`best_config.json`, from `autotune.py --quick`) raises
`TTS_VOCODER_CHUNK_PATCHES` from 12 to 32, which improved the long-form pipeline
from RTF 0.690 to 0.469 with no change in peak VRAM (5451.9 MB).

### Speaker similarity

RTF and VRAM say nothing about whether the output sounds like the speaker, so
`scripts/speaker_similarity.py` measures it directly with ECAPA-TDNN. Measured
on this host, reference `my_voice_v2.wav`:

| clip | cosine | what it is |
|---|---:|---|
| `my_voice_v1.wav` | 0.7146 | a **real** second recording of the same speaker |
| `test_audio_sample.wav` | **0.7583** | 48.5 s of TTS output, long-form |
| `test_api.wav` | 0.6392 | 2.9 s of TTS output, short |
| VoxCeleb control | −0.005 / 0.019 | a **different** speaker (the floor) |

The long-form synthesis scores *above* the real second recording of the same
person. With the impostor floor at ≈0, that is a genuine result rather than a
metric that scores everything highly: on this scale ≈0.71 is same-speaker
speech and 0.76 is the clone clearing it.

The short 2.9 s clip scores lower simply because a 3-second sample gives the
speaker encoder less to work with — a limitation of measuring short audio, not of
the synthesis.

The two VoxCeleb controls are the example clips shipped with the
`speechbrain/spkrec-ecapa-voxceleb` model repo (id10003 and id10004). Two
independent speakers both landing within ±0.02 of zero is what makes the floor
trustworthy. Re-fetch them to reproduce:

```bash
docker compose exec tts python - <<'PY'
import urllib.request, pathlib
base = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/"
for name in ("example1.wav", "example2.flac"):
    out = pathlib.Path("/app/data/references/controls") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(urllib.request.urlopen(base + name, timeout=60).read())
    print("saved", out)
PY
```

These are cosine similarities, so they are only interpretable relative to each
other and to the two controls. The script always reports a self-similarity
ceiling (1.0000 here) and accepts `--impostor` for the floor; without both, an
absolute threshold would be meaningless.

#### What 2 sampling steps actually costs

`TTS_NUM_STEPS=2` was chosen for memory reasons alone — the DiT KV cache is
sized by step count — so it is worth knowing what it trades. Both rows below
are the same text, same reference and the same 128-patch budget; only the step
count differs:

| `TTS_NUM_STEPS` | similarity | long-form RTF | peak VRAM |
|---|---:|---:|---:|
| 2 (tuned default) | 0.7645 | **0.469** | **5458 MB** |
| 4 | **0.7859** | 0.550 | 5518 MB |
| — real same-speaker recording | 0.7146 | — | — |

So doubling the steps buys **+0.021 cosine** for **+17 % generation time** and
**+60 MB of VRAM**. Four steps does still fit at `max_generate_length=128` —
the tuned `vocoder_chunk_patches=32` freed enough headroom that the KV cache
doubles without an OOM, which the earlier notes assumed would not happen.

Both configurations clear the real same-speaker recording, so the tuned default
is not a quality compromise worth reversing. If voice fidelity matters more
than throughput for a particular job, `TTS_NUM_STEPS=4` is now a viable choice
on this card:

```bash
TTS_NUM_STEPS=4 docker compose up -d --force-recreate
```

Treat the +0.021 as indicative rather than precise: it is a single generation
per configuration, and a difference this small is within the range where more
samples would be needed to claim significance.

#### Loudness normalisation, measured

`process_txt.py` targets **−16 LUFS with a −1.5 dBTP ceiling**. Single-pass
`loudnorm` guesses the loudness before analysing, so it normalised against an
estimate; the two-pass version measures with `ebur128` first and applies the
computed gain. Measured on real output:

| | loudness error | true peak |
|---|---:|---:|
| single-pass `loudnorm` | −0.50 LU | −1.40 dBFS |
| two-pass, limiter at the ceiling | −0.50 LU | **−0.80 dBFS — over** |
| two-pass, limiter backed off 1 dB | −0.90 LU | **−1.80 dBFS — held** |

The middle row is the trap: `alimiter` caps *sample* peak, not oversampled true
peak, so setting it to −1.5 dB still let the true peak reach −0.8. The limit is
therefore set 1 dB below the ceiling (`TRUE_PEAK_MARGIN_DB`).

This material has a **~16 dB crest factor**, so −16 LUFS and −1.5 dBTP are close
to mutually exclusive. The limiter costs ~0.4 LU of loudness accuracy to hold
the ceiling, and `process_txt.py` says so on stderr rather than hiding it. If
loudness accuracy matters more than the peak ceiling, pass `--no-normalize` and
handle it downstream.

---

## 7. Operational notes

* **The first request after a restart is 2-3x the steady cost, not 20x.** It
  used to look far worse (RTF 3.3-8.7) because the warmup compiles a 10.5 s
  utterance while a short first request hits a shape it has not seen. Measured
  on a fresh container, the first *short* request runs at RTF 0.74 and settles
  to 0.66-0.69 by the fourth. A first *long* request (~13 s) is essentially
  free, because that is close to the warmup shape. The earlier "RTF 3.3 on the
  first request" figures in this document came from measuring against a
  different audio length than the service had been warmed on, not from a real
  regression.
* **The 6.5% host-idle figure did not translate into a meaningful speedup.**
  `OMP_NUM_THREADS=16` (this host's physical core count) measured engine RTF
  0.4322 against 0.4361 at 8 — about 0.9%, inside run-to-run noise. So that
  idle time is not CPU thread starvation; it is more likely kernel-launch
  overhead. If you want the last percent, set `OMP_NUM_THREADS` to `nproc`;
  it is exposed as a variable for that reason. The bigger lever would be
  re-enabling CUDA graphs, which `app/patches.py` disables for VRAM reasons.
* **Cold start is `torch.compile`, and it is cached.** The warmup walks every
  reachable DiT bucket and vocoder chunk size, and Inductor spends ~105 s
  generating CUDA kernels before the GPU does any useful work. Those kernels go
  to `/tmp` by default, which is *not* a volume, so they were discarded on every
  restart. `TORCHINDUCTOR_CACHE_DIR` now points at the `inductor_cache` volume:

  | | cold start |
  |---|---:|
  | first run (populating the cache) | 219 s |
  | every later restart | **69 s** |

  Losing that volume costs one slow startup and nothing else. The Docker health
  check still allows up to an hour (`--start-period=3600s`) and returns `503`
  until the engine is genuinely ready.
* **The container runs as you, not root.** `user: "${VOXDOCK_UID:-1000}:${VOXDOCK_GID:-1000}"`
  keeps everything written into `./data` owned by you, so nothing needs `sudo`
  to fix. If your uid is not 1000, set `VOXDOCK_UID`/`VOXDOCK_GID` in `.env`
  **and rebuild** — the uid must match the `/etc/passwd` entry baked into the
  image, or startup dies with `getpwuid(): uid not found`. A non-root uid cannot
  write to `/root`, which is why `HF_HOME` points at a tmpfs at `/cache`.
* **The `hf_cache` volume** (`local_tts_hf_cache`) keeps the ~4.4 GB of weights
  across rebuilds — only delete it if you want to re-download. It was created
  under the pre-rename `local-tts-docker` project, so compose prints a harmless
  warning about it on every `up`; renaming the volume would orphan the weights.
* **Changing `TTS_MAX_GENERATE_LENGTH`** needs `docker compose up -d` (it is a
  load-time parameter) and triggers a fresh compile warmup.
* **Concurrency** is serialised by an `asyncio.Lock`: one 6 GB GPU cannot serve
  two generations at once without spilling into system RAM.
* **VRAM numbers** come from `torch.cuda.mem_get_info()` (whole device) and
  `torch.cuda.memory_allocated()` / `max_memory_allocated()` (this process),
  exactly as the spec requests.
* **FFmpeg stitching** uses `-c copy` over format-consistent PCM_16 chunks, so it
  is lossless; a small silence gap (`--gap`, default 0.30 s) is inserted between
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
| `getpwuid(): uid not found: 1000` | `VOXDOCK_UID`/`VOXDOCK_GID` do not match the image's `/etc/passwd` entry — set them in `.env` and `docker compose up -d --build` |
| `PermissionError` on `data/.speechbrain/…` | the similarity model cache holds symlinks into `/root`, which a non-root uid cannot read. Delete `data/.speechbrain/` and re-run the script to rebuild it as real files |
| `low-VRAM patches did not apply: …` at startup | `dots.tts` was upgraded and renamed a symbol a patch needs. Pin `dots.tts==0.3.1`, or set `TTS_LOW_VRAM_PATCHES=false` and lower `TTS_MAX_GENERATE_LENGTH` |
| every restart takes ~3.5 min again | the `inductor_cache` volume was deleted; it repopulates on the next start |
| `volume "local_tts_hf_cache" was created for project "local-tts-docker"` | expected and harmless — the volume predates the rename and keeps the ~4.4 GB of weights |
| first request after a restart is slow (RTF > 2) | expected: it pays any compile the warmup did not cover. The second request is representative |
| `speechbrain is not installed` | it is in `requirements.txt`; rebuild the image |

