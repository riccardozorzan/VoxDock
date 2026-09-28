# TODO v4 — Status, Where Things Live, and What's Left

Written after the v3 implementation (6 commits, `f994d22`). This is the
**handover document**: what is done, where each artefact lives, what is still
open. v2 and v3 stay in this folder as the original specifications; this one
supersedes them for status purposes.

Read `README.md` for *how to use* the system. Read this for *what is left*.

---

## 1. Where everything is stored

### 1.1 Code (all git-tracked)

| Path | Lines | Role |
| :--- | ---: | :--- |
| `app/main.py` | 717 | FastAPI service: routes, runtime loader, OOM classification |
| `app/config.py` | 446 | Settings: env → `best_config.json` → defaults |
| `app/patches.py` | 305 | Three low-VRAM patches for `dots_tts` internals |
| `app/utils.py` | 343 | GPU telemetry, WAV buffers, OOM/cache helpers |
| `app/speakers.py` | 205 | Speaker registry (register once, reuse by id) |
| `scripts/autotune.py` | 1606 | Sweeps model/precision/settings → `best_config.json` |
| `scripts/process_txt.py` | 1049 | Long-form: chunk → POST → concat → loudnorm |
| `scripts/benchmark_test.py` | 580 | Cold start / peak VRAM / RTF benchmark |
| `scripts/clean_audio.py` | 335 | Reference denoise / validate / trim |
| `scripts/test_api.py` | 129 | One-shot endpoint sanity check |

### 1.2 Specifications

| Path | State |
| :--- | :--- |
| `TODO/todo-tts-agent-v2.md` | superseded; kept for its measured-results tables |
| `TODO/todo-tts-agent-v3.md` | implemented, Phases 1–5 + 6.1/6.4 complete |
| `TODO/todo-tts-agent-v4.md` | this file |

### 1.3 Data — tracked vs. ignored

**Tracked** (small, and the evidence behind the documented numbers):

| Path | What |
| :--- | :--- |
| `data/input/reference_set.json` | the 3 reference clips the autotuner may choose between |
| `data/input/*.txt` | transcripts — kept, since exact-match transcripts are what make cloning reproducible |
| `data/references/*.txt` | raw reference transcripts |
| `data/output/benchmark_report.json` | VRAM / RTF verdict |
| `data/output/tuning_report.json` | full tuning matrix, failures included |
| `data/output/test_audio_sample.report.json` | per-chunk long-form report |

**Ignored** (large, personal, or regenerable): every `*.wav`, `.env`,
`data/speakers/`, `data/temp_chunks/`, `__pycache__/`.

### 1.4 Runtime stores (gitignored, on disk only)

| Path | Contents |
| :--- | :--- |
| `data/references/` | raw voice WAVs + matching transcripts |
| `data/input/reference_clean.wav` | the 7.00 s denoised reference actually used |
| `data/speakers/<id>.{wav,txt,json}` | registered voices; `.json` holds metadata + reuse counter |
| `data/temp_chunks/` | intermediate per-chunk WAVs (auto-cleaned) |
| `data/output/*.wav` | final long-form audio |
| `data/output/best_config.json` | **does not exist yet** — see §4.1 |
| HF cache | `hf_cache` Docker volume, so weights are not re-downloaded |

### 1.5 Docker

Container `dots-tts`, port `8000`, GPU passthrough, project bind-mounted at
`/app`. `data/speakers/` and `data/output/` are the same files from host and
container.

---

## 2. Git state

Branch `main`, 6 commits, each one logical change:

```
f994d22 docs: record v3 status, spec deviations, and the new API surface
3633f64 process_txt: speaker reuse, OOM auto-split, EBU R128, tuned chunk size
4f186fe autotune: rename to spec name, sweep model + precision, emit best_config.json
ea4ef5e api: speaker registry with on-disk prompt reuse (v3 Tasks 3.3, 3.5)
30711af config: resolve settings from env -> best_config.json -> defaults (v3 Task 3.1)
756b8dd Baseline: containerized dots.tts TTS service on a 6 GB RTX 4050
```

No remote is configured — nothing has been pushed anywhere.

---

## 3. What is done and verified

| Requirement | Target | Measured | |
| :--- | :--- | :--- | :--- |
| Peak VRAM | `< 5.5 GB` | **5.381 GB** benchmark / 5.452 GB long-form | ✅ |
| RTF, 13.4 s utterance | `< 0.5` | **0.4638** | ✅ |
| RTF, ~3.7 s utterance | `< 0.5` | 0.63 – 0.67 | ⚠️ see §4.3 |
| Long-form pipeline | lossless | 161 words → **51.24 s** 48 kHz mono, RTF 0.54 | ✅ |
| EBU R128 | −16 LUFS | −17.5 LUFS, true peak −1.5 dBFS | ⚠️ see §4.5 |
| Speaker similarity | SIM > 83 | **never measured** | ❌ see §4.2 |
| v3 Phases 1–5 + 6.1, 6.4 | — | implemented, live-tested | ✅ |

New in v3, all verified live: `best_config.json` resolution, speaker registry
(register/list/delete), `speaker_id` on the speech route, cache-eviction
endpoint, server-side OOM classification, model + precision sweep,
`best_config.json` emission, soar-vs-mf table, OOM auto-split, EBU R128,
300 ms inter-chunk silence, tuned chunk sizing.

---

## 4. What is still to do

Ordered by risk. Item 4.2 is the one that could invalidate a design decision.

### 4.1 Run the autotuner — the last unimplemented v3 task

**Status:** v3 Tasks 6.2 / 6.3. Not code — an *operation*.

`data/output/best_config.json` has never been produced by a real sweep, so the
`app/config.py` profile path is only proven by a synthetic round-trip test.

```bash
python scripts/autotune.py --quick --apply     # ~20-40 min
python scripts/autotune.py                     # full, 60-80 min
```

Before running: it restarts the container many times (3–5 min each, mostly
`torch.compile` warmup) and downloads `dots.tts-soar` (~2 GB). Expect `soar` to
be ruled out on this card.

- [ ] Run `--quick --apply`
- [ ] Confirm `data/output/best_config.json` exists and peak VRAM `< 5.5 GB`
- [ ] Restart the container, confirm `/health` → `config_source.best_config`
      is non-null (proves the profile is actually in force)
- [ ] Re-run the benchmark against the tuned profile

### 4.2 Measure speaker similarity — the biggest unknown

**Status:** never measured, in v2 or v3. Every quality claim so far is inferred
from RTF/VRAM, not from the audio.

This gates a decision already made: `dots.tts-mf` was chosen over `soar` purely
on VRAM grounds, and `TTS_NUM_STEPS=2` purely on memory grounds. If `mf` at 2
steps clones poorly, the answer may be a different checkpoint, or more steps on
a bigger card.

- [ ] Pick a metric: speaker-embedding cosine similarity, or WavLM-SV
- [ ] Score the reference, each reference-set clip, and the generated output
- [ ] Compare `num_steps` 2 vs 4 for similarity (not just RTF)
- [ ] Record in `data/output/similarity_report.json`; compare to SIM > 83

### 4.3 Profile the ~1.0 s fixed per-request cost

**Status:** open since v2. Generation time fits `0.98 s + 0.373 × audio_seconds`,
so **RTF < 0.5 needs ≥ 7.7 s of audio per request** and cannot be tuned away —
the constant does not scale with output length. Short utterances sit at
0.63–0.67.

Suspects: prompt prefill, the LLM→latent→BigVGAN two-stage hop, host↔device
transfers, Python overhead.

- [ ] `torch.profiler` one request; attribute the 0.98 s
- [ ] Record where it goes: prefill / decode / vocoder / IO
- [ ] If it is prefill, note `reference_set.json` exists to test exactly this —
      a shorter reference should shrink the constant

### 4.4 Fix the git identity before pushing

`git config user.email 'your-email@example.com'` — still the global placeholder.

Nothing is pushed and no remote is set, so nothing is leaked yet — but this must
be fixed before the first push, or every commit carries a fake address.

- [ ] Set a real `user.email` (repo-local is fine)
- [ ] Decide on a remote, or confirm the repo stays local-only

### 4.5 Close the loudnorm accuracy gap

Single-pass `loudnorm` lands at **−17.5 LUFS** against a −16 LUFS target (true
peak is correct at −1.5 dBFS). Closing it needs ffmpeg's two-pass mode: measure,
compute the offset, apply it.

- [ ] Implement two-pass `loudnorm` in `normalize_loudness()`
- [ ] Re-verify with `ffmpeg -af ebur128`


### 4.6 Add an automated test suite

Everything so far was verified by ad-hoc commands during development. There is
no `pytest` suite, so the next change can silently regress any of it.

- [ ] `tests/test_config.py` — the three-tier chain, including malformed and
      wrong-typed `best_config.json`
- [ ] `tests/test_speakers.py` — registry CRUD, id validation, **path traversal
      rejection** (`../evil` is already handled; keep it that way)
- [ ] `tests/test_process_txt.py` — OOM split ordering, 500-is-not-split,
      chunk splitting
- [ ] Promote `scripts/test_api.py` to an assertion set

### 4.7 Operational hardening

- [ ] **Container/host ownership.** `data/speakers/test_audio_sample.wav` is
      `root`-owned; earlier runs also left root-owned files in `data/output/`
      that broke host-side report writes with `PermissionError`. Either run
      scripts on the host only, or add a `user:` mapping to `docker-compose.yml`.
- [ ] **Long-run VRAM soak.** Peak is 5.381 GB against a 5.5 GB cap — only
      0.119 GB of headroom. A 20+ minute job has never run to completion.
- [ ] **`app/patches.py` upgrade guard.** Coupled to `dots_tts` internals, and it
      currently *logs and no-ops* rather than crashing. Silently losing the
      patches means an OOM at 128 patches — make failure loud, or assert at
      startup that each patch actually applied.
- [ ] **`model_variant` is restart-scoped.** Two checkpoints cannot co-exist in
      6 GB, so it is a startup setting, not a per-request parameter (v3 asks for
      the latter). Documented as a deliberate deviation.

---

## 5. Known physical limits — not fixable on this hardware

Recorded so they are not re-litigated as bugs.

1. **Chunk size 100–180 words is impossible here.** 1 patch = 160 ms. A 100-word
   chunk is ~17 s ≈ 150 patches *plus* ~44 prompt patches ≈ 194 patches ≈ 10 GB.
   The ceiling is 128 patches ≈ 20.5 s per request, reference included;
   `--max-words auto` settles at 33 words/chunk. The spec's 180 words assumes a
   far larger budget than the 5.5 GB the same document mandates.
2. **`dots.tts-soar` cannot meet the 5.5 GB cap.** Its smallest published bucket
   is 5.65 GB. The comparison table keeps this as a recorded result, not a gap.
3. **RTF < 0.5 for short speech** — see §4.3.
4. **Single-pass loudnorm accuracy** — see §4.5.

---

## 6. Quick reference

```bash
# start / stop
docker compose up -d --build
docker compose logs -f tts

# health (503 while the model loads, 200 when ready)
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# synthesize once
curl -s -o out.wav -X POST http://127.0.0.1:8000/v1/audio/speech \
  -F "text=Ciao, questo è un test." \
  -F "reference_audio=@data/input/reference_clean.wav" \
  -F "reference_text=$(cat data/input/reference_clean.txt)"

# long-form (all v3 behaviour is on by default)
python scripts/process_txt.py -i data/input/test_audio_sample.txt

# benchmark / autotune
python scripts/benchmark_test.py
python scripts/autotune.py --quick
```

Routes: `GET /health` · `GET /v1/models` · `POST /v1/audio/speech` ·
`POST /v1/speakers/register` · `GET /v1/speakers` ·
`DELETE /v1/speakers/{id}` · `POST /v1/admin/clear-cache`

Interactive API docs: <http://127.0.0.1:8000/docs>

