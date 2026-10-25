# Lapisan naratif (SLM lokal) — opsional

Lapisan ini mengubah **fakta diagnosis P-DINA** menjadi laporan berbahasa Indonesia untuk guru dan siswa, memakai
**Qwen3-4B-Instruct-2507** yang disajikan oleh server OpenAI-compatible lokal (Ollama, llama.cpp, atau vLLM).

> SLM **hanya mengomunikasikan** fakta yang sudah dihitung P-DINA. SLM tidak menghitung, mengubah, atau memprediksi
> mastery, learning gap, rekomendasi, maupun metrik. Matematika, checkpoint, dan evaluasi P-DINA tidak disentuh.

## Cara kerja

```
diagnose() ──► NarrativeFacts (ringkas, ID dipseudonimkan) ──► prompt (aturan + heading wajib + 1 contoh)
           ──► SLM lokal (1 request/laporan) ──► validator ──ok──► teks + cache
                                                     └─gagal─► repair (≤ max_retries) ──► template deterministik
```

| Komponen | Isi |
|---|---|
| `cdacdm/narrative.py` | `NarrativeConfig`, `NarrativeFacts`/`build_facts`, `build_prompt`, `validate_narrative`, `fallback_narrative`, `NarrativeGenerator` (cache, audit log, retry, circuit breaker) |
| `CDMService.generate_narrative(result, audience, student_id, generator)` | mengembalikan `{"text", "meta", "validation", "fallback"}`; `report_json()` tidak berubah |
| `configs/slm.json` | konfigurasi (model, URL, batas token, cache, log) |
| `configs/narrative_examples.jsonl` | contoh few-shot (1 guru, 1 siswa; hanya contoh yang sesuai audiens yang dikirim) |
| `app.py` | tombol **Buat ringkasan naratif** (Dashboard, Learning Gap Analysis), **Buat feedback siswa** (Student Assessment), **Jelaskan rekomendasi** (Intervention Recommendations) |

Laporan **tidak** dibuat otomatis saat halaman dibuka; hanya lewat tombol.

## Menjalankan

### 1. Server model (Ollama, Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_slm.ps1          # serve + pull + uji singkat
# atau manual:
ollama pull qwen3:4b-instruct
ollama serve
```

Ollama menerbitkan Qwen3-4B-Instruct-2507 dengan tag **`qwen3:4b-instruct`** (Q4_K_M, ±2,5 GB). Tag
`qwen3:4b-instruct-2507` tidak ada di Ollama, jadi `configs/slm.json` memakai `qwen3:4b-instruct`. Untuk llama.cpp/vLLM,
isi `base_url` dan `model` sesuai nama yang dipakai server tersebut.

**Konteks server.** Endpoint OpenAI-compatible Ollama mengabaikan `num_ctx` per request; jendela konteks diatur di
server (default 4096 token). Prompt sudah dibuat ringkas agar muat di 4096, tetapi disarankan `OLLAMA_CONTEXT_LENGTH=8192`
(sudah diatur oleh `scripts/run_slm.ps1` bila server dijalankan dari skrip). Jika Ollama berjalan sebagai aplikasi tray,
set variabel lingkungan pengguna `OLLAMA_CONTEXT_LENGTH=8192` lalu restart Ollama. Output yang terpotong
(`finish_reason=length`) ditolak dan dicatat sebagai alasan *output SLM terpotong*.

### 2. Aplikasi

```bash
streamlit run app.py
```

Aplikasi tetap berfungsi tanpa server SLM: tombol menghasilkan laporan **template** dengan peringatan
"Laporan dibuat dengan template karena layanan SLM tidak tersedia." (tanpa error, tanpa hang: connect timeout 5 detik).

### 3. Uji

```bash
pytest -q                                   # semua test, tanpa jaringan
# uji opsional ke server nyata:
#   PowerShell: $env:CDACDM_SLM_TEST_URL="http://localhost:11434/v1"; pytest -q tests/test_narrative.py -k real
```

Smoke test manual (Python):

```python
from cdacdm.service import CDMService
import pandas as pd
svc = CDMService.load("artifacts")
demo = pd.read_csv("artifacts/demo/test_split_responses.csv", dtype={"student_id": str})
res = svc.diagnose(demo[demo["student_id"].isin(demo["student_id"].drop_duplicates().head(30))])
out = svc.generate_narrative(res, "teacher")
print(out["fallback"], out["meta"]["latency_ms"], out["meta"]["reason"]); print(out["text"])
```

## Konfigurasi (`configs/slm.json`)

| Kunci | Default | Catatan |
|---|---|---|
| `enabled` | `true` | `false` → selalu template, tanpa request |
| `base_url` | `http://localhost:11434/v1` | wajib localhost/127.0.0.1 kecuali `allow_remote` |
| `model` | `qwen3:4b-instruct` | tag server |
| `api_key_env` | `CDACDM_SLM_API_KEY` | nama env berisi API key (key **tidak** boleh ditulis di config; loader menolaknya) |
| `temperature` / `top_p` / `max_tokens` | 0.2 / 0.9 / 700 | tanpa streaming (validator butuh teks utuh) |
| `timeout_sec` / `connect_timeout_sec` | 120 / 5 | read / connect |
| `max_retries` | 2 | repair setelah gagal validasi; HTTP 429/5xx di-retry dengan backoff 1 s / 4 s; koneksi gagal tidak di-retry |
| `prompt_version` | `narrative-v2` | ikut kunci cache |
| `max_student_concepts` / `max_class_concepts` / `max_gaps` / `max_recommendations` | 12 / 11 / 5 / 5 | batas ukuran fakta |
| `max_examples` | 1 | contoh few-shot per request (maks. 2) |
| `cache_dir` / `cache_ttl_hours` | `artifacts/reports/cache` / 168 | kunci = sha256(fakta + audiens + prompt_version + model) |
| `log_path` | `artifacts/reports/narrative_log.jsonl` | audit log |
| `allow_remote` | `false` | lihat Privasi |

Variabel lingkungan: `CDACDM_SLM_CONFIG` (path config lain), `CDACDM_SLM_ALLOW_REMOTE=1`, `CDACDM_SLM_API_KEY`.
Generator di aplikasi di-cache (`st.cache_resource`): setelah mengubah `slm.json`, restart Streamlit.

## Validator (hard fail)

Teks SLM hanya ditampilkan bila lolos semua pemeriksaan; jika tidak, diminta perbaikan (maks. `max_retries`), lalu
jatuh ke template deterministik (`fallback`), yang juga lolos validator.

1. **Angka**: setiap angka di teks harus berasal dari fakta JSON (bilangan bulat apa adanya; 1–2 desimal dengan titik
   atau koma; proporsi 0–1 sebagai persen, dengan `%` atau "persen"; pemisah ribuan `28.003`). Daftar bernomor ditolak.
   **Angka per konsep**: pada baris yang menyebut nama konsep, angkanya harus milik konsep itu (atau fakta global: τ, n_min,
   jumlah siswa/konsep/respons). Contoh yang ditolak: "Linear Equations … 60%" padahal 60% milik konsep lain (fakta: 69%).
2. **Heading wajib** per audiens (guru: Ringkasan kelas, Temuan utama, Konsep prioritas, Rekomendasi tindakan,
   Keterbatasan dan catatan; siswa: Ringkasan, Yang sudah dikuasai, Yang perlu dilatih, Langkah belajar, Catatan).
3. **Placeholder**: jika ada rekomendasi placeholder, teks wajib memuat "placeholder" atau "perlu validasi".
4. **Klaim terlarang**: terbukti, pasti mastered, dijamin, efektif meningkatkan, menyembuhkan, siswa malas, guru gagal.
5. **Konsep**: nama konsep checkpoint yang tidak ada di fakta ditolak.
6. **Panjang** 120–900 kata; tanpa HTML.

## Privasi

* Default **hanya localhost**. `base_url` lain ditolak kecuali `allow_remote: true` **dan** env `CDACDM_SLM_ALLOW_REMOTE=1`.
* Fakta tidak memuat matriks respons, `UserId` asli, nama, atau sekolah. Siswa selalu disebut `Siswa-<sha256[:8]>`
  (lokal maupun remote); laporan guru hanya memuat agregat kelas (kecuali tombol *Jelaskan rekomendasi*, yang memuat
  satu siswa terpilih dengan ID pseudonim).
* Audit log berisi waktu, audiens, model, prompt_version, hash input, cache_hit, latensi, jumlah percobaan, status
  validasi, fallback, alasan, dan usage token. **Tidak** ada teks laporan maupun ID siswa.
* Cache disk menyimpan teks laporan tervalidasi (dengan ID pseudonim) di `artifacts/reports/cache`; hapus folder ini untuk
  membersihkannya.

## Troubleshooting

| Gejala | Penyebab / solusi |
|---|---|
| Peringatan template, alasan *SLM server tidak dapat dihubungi* | server tidak jalan: `ollama serve` atau buka aplikasi Ollama |
| Alasan *SLM HTTP 404* | tag model salah: `ollama list`, samakan `model` di `slm.json` |
| Alasan *output SLM terpotong* atau heading terakhir hilang | konteks server terlalu kecil: `OLLAMA_CONTEXT_LENGTH=8192`, restart Ollama |
| Alasan *circuit breaker open* | 3 kegagalan berturut-turut; request dilewati 60 detik, lalu dicoba lagi |
| Alasan *validation failed: angka …* | model mengarang/membulatkan angka; template ditampilkan. Pantau dengan `scripts/eval_narrative.py` |
| Sangat lambat | GPU 4 GB → model terbagi CPU/GPU. Tutup aplikasi lain yang memakai VRAM, atau pakai laptop terhubung charger |
| Alasan *SLM timeout (ReadTimeout)* | > 120 detik. Biasanya model Ollama lain sedang dimuat (cek `ollama ps`; mis. `qwen2.5-coder` dari opencode) sehingga Ollama menukar model di VRAM 4 GB, atau laptop di baterai. Hentikan model lain (`ollama stop <nama>`) atau naikkan `timeout_sec` |
| Ubah config tidak berpengaruh | restart Streamlit (generator di-cache) |

## Batasan

* Kualitas bahasa SLM 4B terbatas; validator mencegah angka/klaim yang tidak berdasar, tetapi tidak menjamin
  ketepatan pedagogis. Laporan adalah draf untuk guru.
* Mastery adalah estimasi model; rekomendasi placeholder perlu validasi guru; tidak ada klaim efektivitas intervensi.
* Pada laptop dengan GPU 4 GB, satu laporan dapat memakan puluhan detik hingga beberapa menit (lihat bagian hasil uji
  di bawah); repair menambah waktu.
* Validator memeriksa angka dan pasangan angka–konsep, tetapi tidak semua makna kalimat (mis. konsep yang ditaruh di bagian
  yang keliru). Guru tetap meninjau laporan sebelum dibagikan.

## Hasil uji model (EEDI, 30 siswa, laptop i5-12450H + RTX 2050 4 GB, Ollama 0.40)

| Model (tag Ollama) | Ukuran | Waktu per laporan | Hasil |
|---|---|---|---|
| **`qwen3:4b-instruct`** (dipakai) | 2,5 GB | guru ±80–105 s, siswa ±80 s (tanpa model lain di GPU) | lolos validator pada percobaan pertama; angka benar |
| `qwen2.5:3b` | 1,9 GB | ±25 s per percobaan (100% GPU) | 4 dari 5 persentase konsep tertukar/salah → ditolak → template |
| `gemma3:1b` | 0,8 GB | ±10–40 s | mengarang angka, menyalin angka contoh, menerjemahkan nama konsep → sebagian besar template |

Kesimpulan: model < 4B lebih cepat tetapi tidak cukup akurat untuk laporan yang memuat angka. Untuk membandingkan model
baru secara sistematis, pakai `scripts/eval_narrative.py --models <tag> ...`.

## Phase 2 (kerangka, belum dijalankan)

* `scripts/build_narrative_dataset.py` — JSONL `(fakta → narasi)` dari artefak; teks target adalah **draf template** yang
  wajib diedit guru (`needs_teacher_edit: true`).
* `scripts/train_lora.py` + `configs/lora.json` — resep QLoRA (Unsloth/PEFT); menolak baris yang belum diedit guru.
  Tidak diimpor aplikasi; dependensinya tidak masuk `requirements.txt`.
* `scripts/eval_narrative.py` — membandingkan model (mis. `qwen3:4b-instruct`, `qwen3:8b`, Sahabat-AI via tag GGUF lokal):
  pass rate percobaan pertama vs setelah repair, jenis pelanggaran, latensi, token.
