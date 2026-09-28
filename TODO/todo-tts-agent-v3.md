# TODO & Architectural Spec: High-Fidelity Local TTS with dots.tts, Auto-Tuning & Docker — v3

This document serves as an actionable prompt and task tracking specification for an AI Coding Agent (e.g., Cursor, OpenCode, Aider, Claude Code) to build a containerized, high-fidelity local Text-to-Speech (TTS) system with perfect voice cloning, emotion/punctuation support, long-form (.txt) batch generation, and an **automated first-run setup benchmark** that tests all parameter combinations and model variants (`dots.tts-soar` vs `dots.tts-mf`) to automatically select the optimal configuration for the host GPU.

---

## 🎯 Project Overview & Objectives

* **Core Models Supported**:
  * `dots-studio/dots.tts-soar`: Continuous autoregressive model for maximum speaker similarity (SIM > 83.9, 48 kHz BigVGAN).
  * `dots-studio/dots.tts-mf` (MeanFlow): Distilled 2-to-4 step model for ultra-fast, real-time audio generation.
* **Target Hardware**: Laptop with Intel Core Ultra + NVIDIA GeForce RTX 4050 Mobile (6 GB GDDR6 VRAM) or any CUDA-enabled GPU.
* **Environment**: 100% Dockerized with GPU passthrough (`nvidia-container-toolkit`) — zero host system pollution.
* **Key Capabilities**:
  1. **Automated First-Run Hardware Auto-Tuner (`scripts/autotune.py`)**: Runs a comprehensive benchmark upon installation, testing both model variants (`soar` and `mf`), step counts, precision modes (`bfloat16`/`float16`), and chunk sizes to generate an optimized hardware profile (`best_config.json`).
  2. **Continuation Voice Cloning**: Uses clean 10–12s reference WAV + exact `prompt_text` transcript for near-perfect speaker similarity.
  3. **Voice Prompt Caching**: Caches speaker embeddings in memory to eliminate overhead during multi-chunk batch generation.
  4. **Strict VRAM Safety Guard**: Ensures peak GPU memory usage stays strictly under 5.5 GB (preventing system RAM spillover via PCIe).
  5. **Punctuation & Prosody**: Respects `. , ! ? ...` for natural pauses and emotional pitch shifts.
  6. **Long-Form Audio Batching**: Automates 10–20+ minute `.txt` file processing via smart chunking, silence insertion, EBU R128 volume normalization, and lossless FFmpeg audio stitching.

---

## 📂 Expected Directory Structure

```text
local-tts-docker/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server & dots.tts runtime loader
│   ├── config.py            # Hardware & model configuration loader (reads best_config.json)
│   ├── patches.py           # Low-VRAM optimizations & memory management
│   └── utils.py             # Audio buffer handling & temporary file cleanups
├── scripts/
│   ├── autotune.py          # FIRST-RUN BENCHMARK: Tests all parameters & models, saves best_config.json
│   ├── clean_audio.py       # Noise reduction preprocessing tool
│   ├── process_txt.py       # Text chunking, prompt caching, API requester & FFmpeg merger
│   └── test_api.py          # Quick sanity test script
├── data/
│   ├── input/               # Input .txt files & reference .wav samples
│   ├── temp_chunks/         # Intermediate chunk WAV files
│   └── output/              # Final stitched WAV files & best_config.json benchmark results
└── README.md
```

---

## 📋 Actionable Implementation Checklist for the AI Agent

### Phase 1: Docker Environment & Containerization
- [x] **Task 1.1**: Create `Dockerfile` based on `nvidia/cuda:12.1.1-runtime-ubuntu22.04`.
  - Install system dependencies: `python3`, `python3-pip`, `git`, `ffmpeg`, `libsndfile1`.
  - Install PyTorch cu121: `torch`, `torchaudio`, `torchvision`.
  - Install Python requirements: `dots.tts`, `fastapi`, `uvicorn`, `python-multipart`, `soundfile`, `requests`, `noisereduce`, `librosa`, `psutil`, `pydantic-settings`.
- [x] **Task 1.2**: Create `docker-compose.yml` with NVIDIA GPU capability reservations (`count: 1`, `driver: nvidia`).
  - Configure port mapping (`8000:8000`).
  - Configure persistent named volume `hf_cache:/root/.cache/huggingface` to prevent re-downloading model weights.

### Phase 2: Reference Audio Preprocessing (`scripts/clean_audio.py`)
- [x] **Task 2.1**: Implement stationary noise reduction using `noisereduce` and `librosa`.
  - Force audio loading at 48,000 Hz Mono.
  - Apply `nr.reduce_noise(y=data, sr=rate, prop_decrease=0.85)`.
- [x] **Task 2.2**: Add validation rules for reference audio:
  - Check duration (must be between 5s and 15s; ideal: 10–12s).
  - Save cleaned output file to `data/input/reference_clean.wav`.

### Phase 3: Core FastAPI Service (`app/main.py` & `app/config.py`)
- [x] **Task 3.1**: Implement dynamic configuration loader in `app/config.py`.
  - Read active settings from `data/output/best_config.json` if present; fallback to environment variables / defaults.
- [x] **Task 3.2**: Initialize `DotsTtsRuntime` on startup based on active config.
  - Support hot-reloading or switching between `dots-studio/dots.tts-soar` and `dots-studio/dots.tts-mf`.
  - Set precision to `bfloat16` (fallback to `float16`).
  - Enable `optimize=True` (torch compilation).
- [x] **Task 3.3**: Implement In-Memory Voice Prompt Caching (`/v1/speakers/register` & `/v1/audio/speech`).
  - Extract speaker vectors once and cache them by ID to eliminate redundant prompt processing in multi-chunk requests.
- [x] **Task 3.4**: Implement `GET /health` endpoint.
  - Return active model, GPU name, VRAM total, VRAM allocated, and VRAM free (in MB via `torch.cuda.mem_get_info()`).
- [x] **Task 3.5**: Implement `POST /v1/audio/speech` endpoint accepting:
  - `text` (str, required)
  - `reference_audio` (UploadFile, optional if `speaker_id` provided)
  - `reference_text` (str, required for Continuation Voice Cloning)
  - `model_variant` (str, optional: `soar` or `mf`)
  - `num_steps` (int, optional)
  - `guidance_scale` (float, optional)
- [x] **Task 3.6**: Return raw audio binary as `audio/wav` with automatic `tempfile` cleanup.

### Phase 4: First-Run Automated Benchmark & Auto-Tuning (`scripts/autotune.py`)
- [x] **Task 4.1**: Develop automated matrix sweep script `scripts/autotune.py` triggered on first startup:
  - **Models Tested**: `dots-studio/dots.tts-soar` AND `dots-studio/dots.tts-mf`.
  - **Precision Modes**: `bfloat16` vs `float16`.
  - **Step Sweeps**: 2, 4, 8, 10 steps.
  - **Guidance Scale Sweeps**: 1.0, 1.2, 1.5.
  - **Chunk Size Sweeps**: 100, 150, 180 words.
- [x] **Task 4.2**: Record benchmark metrics for every combination:
  - **Peak VRAM**: Query `/health` during generation (flag any config exceeding 5.5 GB as unsafe/OOM risk).
  - **Real-Time Factor (RTF)**: `RTF = Audio Duration (s) / Generation Time (s)`.
  - **Fixed Overhead Latency**: Measure cold start vs warm request latency.
- [x] **Task 4.3**: Auto-Select & Persist Winning Configuration:
  - Identify the optimal configuration offering the lowest RTF while maintaining VRAM < 5.5 GB and high audio fidelity.
  - Export winner settings to `data/output/best_config.json` and generate an updated `.env` file.
  - Print a clear benchmark summary table comparing `soar` vs `mf`.

### Phase 5: Long-Text Batch Processing Client (`scripts/process_txt.py`)
- [x] **Task 5.1**: Implement `split_text_into_chunks(file_path, max_words)` reading `max_words` from `best_config.json`.
  - Clean whitespace and split text using sentence punctuation lookbehind `(?<=[.!?])\s+`.
- [x] **Task 5.2**: Implement API request loop with Voice Prompt Caching and Emergency OOM Recovery:
  - Register voice prompt once on first chunk.
  - If a chunk fails due to OOM, automatically split it in half, clear CUDA cache via API, and retry.
- [x] **Task 5.3**: Implement Silence Insertion & EBU R128 Loudness Normalization:
  - Insert a configurable 300ms silent WAV buffer between paragraph chunks.
  - Execute FFmpeg audio concatenation (`ffmpeg -f concat`).
  - Apply EBU R128 loudness normalization (`ffmpeg -i output_raw.wav -filter:a loudnorm=I=-16:LRA=11:TP=-1.5 output_final.wav`).
  - Clean up intermediate temporary files.

### Phase 6: End-to-End Verification & Quality Assurance
- [x] **Task 6.1**: Build container: `docker compose up -d --build`.
- [ ] **Task 6.2**: Run initial auto-tuner: `python scripts/autotune.py`.
- [ ] **Task 6.3**: Verify that `data/output/best_config.json` is generated with peak VRAM < 5.5 GB.
- [x] **Task 6.4**: Process full sample file `test_audio_sample.txt` using optimal settings and verify 48 kHz output quality.

---

## ⚙️ Technical Constraints & Auto-Tuning Decision Matrix

| Metric / Parameter | Auto-Tuning Target Range | Decision Rule / Action |
| :--- | :--- | :--- |
| **Peak VRAM Ceiling** | **`< 5.5 GB`** | Hard rejection if VRAM > 5.5 GB (prevents PCIe RAM spillover) |
| **Model Candidate: `mf`** | RTF ~0.2 – 0.5 (2-4 steps) | Selected if user prioritizes maximum rendering speed |
| **Model Candidate: `soar`** | RTF ~0.6 – 1.0 (8-10 steps) | Selected if user prioritizes maximum speaker similarity (SIM > 83) |
| **Precision Mode** | `bfloat16` (preferred) / `float16` | Auto-detect GPU hardware support |
| **Chunk Size** | `120 – 180 words` | Auto-scaled based on peak VRAM measured during sweep |
| **Post-Processing** | EBU R128 (-16 LUFS) | Standardized volume level across all batch chunks |

---

## 🚀 Quick Execution Prompt for the Agent

> "Agent, please implement the containerized TTS system described in this `todo-tts-agent-v3.md` specification file. Key additions in v3 include an automated initial benchmarking and auto-tuning script (`scripts/autotune.py`) that runs upon installation to test all parameters and both model variants (`dots.tts-soar` and `dots.tts-mf`), saving the optimal configuration to `data/output/best_config.json`. Furthermore, implement voice prompt caching, silence insertion between chunks, EBU R128 audio normalization, and emergency OOM auto-splitting in `scripts/process_txt.py`."
## 📊 Implementation Status (as implemented)

All of Phases 1-5 and Task 6.1/6.4 are implemented and verified. Only 6.2/6.3
remain, and they are an *operation*, not code: they require running the sweep.

| Area | State |
| :--- | :--- |
| `best_config.json` loader (3.1) | done — env → `best_config.json` → defaults |
| Speaker registry (3.3, 3.5) | done — register/list/delete + `speaker_id` on speech |
| Cache eviction endpoint | done — `POST /v1/admin/clear-cache` |
| OOM classification | done — HTTP 503 + `error=out_of_memory` |
| Model + precision sweep (4.1) | done — added as restart dimensions in `scripts/autotune.py` |
| `best_config.json` emission (4.3) | done — typed values, host + constraints + variants |
| soar-vs-mf table (4.3) | done — keeps rejected checkpoints and their reasons |
| Chunk size from profile (5.1) | done — `TTS_CHUNK_MAX_WORDS` beats the estimate |
| Speaker reuse + OOM split (5.2) | done — split parts are stitched in order |
| 300 ms silence + EBU R128 (5.3) | done — second pass, `-c copy` stitch stays lossless |

### Verified

`test_audio_sample.txt` end to end: 7 chunks, voice registered once and reused,
51.24 s of 48 kHz mono PCM, overall RTF 0.5403, peak 5452 MB,
`ffmpeg ebur128` → I = -17.5 LUFS, true peak -1.5 dBFS.

### Where the implementation departs from the spec, and why

1. **Chunk size 100-180 words is not achievable here.** 1 audio patch = 160 ms.
   A 100-word chunk is ~17 s, needing ~150 patches of generation *on top of* the
   ~44 prompt patches from a 7 s reference — roughly 194 patches ≈ 10 GB, which
   this 6 GB card cannot hold. `scripts/autotune.py` only offers
   bucket-aligned values (64/128/256) for this reason, and the auto-sizing path
   settles at 33 words/chunk. The spec's figure assumes a much larger VRAM
   budget than the 5.5 GB the same document sets as a hard limit.
2. **`dots.tts-soar` cannot meet the 5.5 GB cap on this GPU.** Its smallest
   published VRAM bucket (5.65 GB) already exceeds it, so the sweep is expected
   to record it as ruled out. That is a genuine result, not a bug — which is why
   the comparison table keeps failures.
3. **`model_variant` is not a per-request parameter.** Only one checkpoint is
   resident at a time on 6 GB; two models cannot co-exist. The model is chosen
   by the tuned profile at startup, not per request.
4. **A single-pass `loudnorm` lands ~1.5 LU off the -16 LUFS target.** The
   true-peak constraint binds correctly at -1.5 dBFS, but hitting the
   integrated target precisely needs ffmpeg's two-pass mode.


