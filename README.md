# CDA-CDM: deteksi & intervensi learning gap (pengembangan dengan EEDI NeurIPS 2020 Task 3/4)

Implementasi end-to-end yang dapat dijalankan ulang: audit artefak data preparation → Q-matrix & pemetaan konsep target →
**probabilistic DINA (variational EM)** + baseline yang terikat Q-matrix → pemilihan hyperparameter & kebijakan diagnosis di
validation → evaluasi **satu kali** di test → CDA (form tetap, parallel forms, adaptif) → profil mastery, learning gap,
rekomendasi intervensi → analisis pre-test/post-test → aplikasi Streamlit (`README_streamlit.md`).
Opsional: laporan naratif berbahasa Indonesia dari SLM lokal (Qwen3-4B-Instruct-2507 via Ollama) yang hanya
mengomunikasikan fakta diagnosis dan selalu divalidasi — lihat `README_slm.md`.

> **Status data.** EEDI Task 3/4 dipakai sebagai **dataset pengembangan dan evaluasi awal**, bukan data pre/post kelas penelitian.
> Populasinya mayoritas siswa sekolah menengah pertama di UK (hanya ±5–6 siswa berusia 16–17 tahun menurut proxy metadata), dan
> tidak ada konsep kalkulus. Tidak ada klaim bahwa model sudah tervalidasi pada kelas SMA penelitian.

## Struktur direktori

```
finalIL/
├── eedi_task34_cda_cdm_pipeline.ipynb   # notebook data preparation yang diaudit (asli, tidak diubah)
├── cda_cdm_model_training.ipynb         # notebook training & evaluasi (dijalankan, berisi output)
├── app.py                               # aplikasi Streamlit
├── requirements.txt
├── README.md / README_streamlit.md / README_slm.md
├── configs/
│   ├── experiment.json                  # path, grid hyperparameter, aturan evaluasi/diagnosis/assessment
│   ├── slm.json                         # lapisan naratif opsional: server/model SLM lokal, batas, cache, log
│   ├── narrative_examples.jsonl         # contoh few-shot laporan guru & siswa
│   ├── lora.json                        # Phase 2 (kerangka): resep QLoRA, tidak dipakai aplikasi
│   ├── target_concept_map.csv           # 57 konsep EEDI → cakupan target (draft, perlu review peneliti)
│   ├── target_curriculum.csv            # topik aljabar & kalkulus SMA (draft) + konsep EEDI yang memetakan
│   ├── focus_item_content_audit.csv     # audit visual sampel 22 soal fokus
│   └── intervention_map.csv             # materi/aktivitas per konsep (PLACEHOLDER sampai divalidasi guru/dosen)
├── cdacdm/                              # paket Python
│   ├── config.py        path & deteksi otomatis artefak
│   ├── data.py          pemuatan, validasi skema, mask missing, alignment npz↔csv, audit leakage
│   ├── reconstruct.py   rekonstruksi artefak lokal dari arsip resmi (bila Drive tidak tersedia)
│   ├── qmatrix.py       analisis Q-matrix, peta konsep target, audit soal fokus, laporan cakupan
│   ├── models.py        PDINA (VEM) + ItemMean / ConceptScore / QLogistic baseline
│   ├── evaluation.py    metrik, kalibrasi, bootstrap, konsistensi split-half, simulasi recovery, pemilihan τ
│   ├── diagnosis.py     profil, label mastery, learning gap, agregasi kelas
│   ├── assessment.py    kelayakan soal, fixed form, parallel forms, sesi adaptif (mutual information), replay
│   ├── intervention.py  rekomendasi berbasis aturan R1–R5, placeholder berlabel
│   ├── prepost.py       perbandingan pre/post (uji berpasangan, effect size, CI, McNemar, Holm)
│   ├── responses.py     parsing & validasi CSV respons (panjang/lebar/answer_option)
│   ├── synthetic.py     data pre/post SINTETIS untuk demo
│   ├── service.py       pemuatan checkpoint & inferensi tanpa retraining (+ generate_narrative)
│   ├── narrative.py     lapisan naratif opsional: fakta, prompt, validator, template, klien SLM lokal
│   ├── api.py           train_model, predict_mastery, select_assessment, detect_learning_gaps,
│   │                    recommend_intervention, compare_pre_post
│   ├── pipeline.py      orkestrasi (CLI: python -m cdacdm.pipeline)
│   └── plots.py
├── scripts/
│   ├── build_notebook.py                # membangun ulang notebook training
│   ├── run_slm.ps1                      # menjalankan Ollama + pull model untuk lapisan naratif
│   └── build_narrative_dataset.py, train_lora.py, eval_narrative.py   # Phase 2 (kerangka)
├── tests/                               # 60 pytest (+1 uji opsional ke server SLM nyata)
├── artifacts_calculus/                  # checkpoint kalkulus kelas XI (DUMMY sintetis) untuk demo aplikasi
├── artifacts/                           # HASIL (dibuat pipeline)
│   ├── audit/        data_audit.json, leakage_audit.json, q_matrix_analysis.json, concept_item_counts.csv,
│   │                 multi_concept_items.csv, q_matrix_target.csv, focus_item_audit.csv, coverage_report.csv,
│   │                 unmapped_target_topics.csv
│   ├── model/        CHECKPOINT: pdina_params.npz, pdina_config.json, diagnosis_policy.json, item_parameters.csv,
│   │                 concept_metadata.csv, q_matrix.csv, target_concept_map.csv, intervention_map_resolved.csv,
│   │                 item_bank.csv, experiment_config_used.json, model_card.json
│   ├── evaluation/   hp_search_validation.csv, threshold_selection_validation.csv, consistency_*.csv, test_metrics.csv,
│   │                 test_model_differences.csv, calibration_test.csv, per_concept_test_main.csv,
│   │                 simulation_mastery_recovery.csv, adaptive_replay_*.csv, figures/*.png
│   ├── diagnosis/    test_mastery_profiles.csv, test_learning_gaps.csv, test_class_gap_summary.csv, test_student_summary.csv
│   ├── assessment/   fixed_form.csv, parallel_form_A.csv, parallel_form_B.csv, parallel_forms_balance.csv,
│   │                 concept_item_adequacy.csv, response_template_fixed_form.csv
│   ├── intervention/ test_intervention_recommendations.csv
│   ├── prepost/      SYNTHETIC_*.csv, SYNTHETIC_prepost_report.json   (SIMULASI)
│   ├── demo/         test_split_responses.csv (respons nyata siswa test, untuk demo app)
│   ├── reports/      (dibuat saat laporan naratif pertama) cache/ + narrative_log.jsonl
│   └── run_summary.json
└── ILCDA/                               # data lokal (arsip resmi + rekonstruksi); di Colab: MyDrive/ILCDA
```

## Menjalankan

### Ringkas (sudah pernah setup)

```powershell
cd D:\finalIL
.venv\Scripts\activate
streamlit run app.py                  # http://localhost:8501
```

### Setup lengkap di Windows (dari nol)

Diuji pada Windows 11, Python 3.12.10, laptop HP Victus (i5-12450H, RAM 8 GB, RTX 2050 4 GB). Semua perintah di bawah
dijalankan di **PowerShell** dari folder proyek (`D:\finalIL`).

**Langkah 1 — Prasyarat**

| Kebutuhan | Keterangan |
|---|---|
| Python 3.12 (64-bit) | https://www.python.org/downloads/ — centang *Add python.exe to PATH*. Cek: `python --version` |
| Ruang disk | ±1,5 GB untuk `.venv`; +2,5 GB bila memakai lapisan naratif (model Qwen3-4B) |
| Ollama (opsional) | hanya untuk laporan naratif; lihat Langkah 7 |

> **Smart App Control (Windows 11).** Bila aktif, Windows dapat memblokir file `.pyd` pandas/NumPy hasil `pip` dengan galat
> `ImportError: DLL load failed ... An Application Control policy has blocked this file`. Proyek tidak bisa berjalan sama
> sekali sampai ini diatasi. Cek di *Windows Security → App & browser control → Smart App Control*. Solusi paling andal:
> matikan Smart App Control (di banyak build Windows 11 tidak dapat diaktifkan lagi tanpa reset; Microsoft Defender tetap
> aktif). Alternatif yang tidak dijamin berhasil: `pip install --force-reinstall pandas==2.2.3`, atau memakai Miniforge/conda.

**Langkah 2 — Virtual environment & dependensi**

```powershell
python -m venv .venv
.venv\Scripts\activate                       # prompt berubah menjadi (.venv)
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import pandas, numpy, scipy, sklearn, streamlit, requests; print('OK')"
```

Bila `activate` ditolak (*running scripts is disabled*): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu ulangi.
Tanpa aktivasi pun bisa: ganti `python`/`pytest`/`streamlit` dengan `.venv\Scripts\python.exe -m ...`.

**Langkah 3 — Checkpoint model**

Aplikasi hanya **memuat** checkpoint; tidak pernah melatih ulang. Proyek ini sudah berisi dua checkpoint siap pakai:

| Folder | Isi |
|---|---|
| `artifacts/model/` | P-DINA EEDI Task 3/4 (aljabar SMP, data nyata) |
| `artifacts_calculus/model/` | kalkulus kelas XI (DUMMY sintetis) |

Cek: `Test-Path artifacts\model\pdina_params.npz` → `True`. Bila checkpoint EEDI tidak ada, bangun ulang (sekali saja):

```powershell
# hanya bila artefak data preparation tidak ada: rekonstruksi dari arsip resmi
python -m cdacdm.reconstruct --zip ILCDA/data.zip --out ILCDA/processed_task34_cda_cdm/final_local_reconstruction
python -m cdacdm.pipeline              # ±9 menit CPU; atau jalankan cda_cdm_model_training.ipynb
```

Pipeline mengevaluasi test set **satu kali**; jangan jalankan ulang hanya untuk "memperbarui" angka.

**Langkah 4 — Verifikasi dengan test**

```powershell
pytest -q
```

Hasil yang diharapkan: `60 passed, 1 skipped` (±1 menit). Test memakai checkpoint sintetis kecil yang dibuat di folder
sementara, tidak menyentuh `artifacts/`, dan tidak memakai jaringan. Yang di-*skip* adalah uji opsional ke server SLM nyata.

**Langkah 5 — Jalankan aplikasi**

```powershell
streamlit run app.py
```

Browser terbuka di http://localhost:8501. Pilih checkpoint di sidebar, lalu ikuti `README_streamlit.md` untuk tiap halaman.
Folder artefak lain: `$env:CDACDM_ARTIFACT_DIR="D:\path\ke\artifacts"` sebelum `streamlit run`.

**Langkah 6 — Pakai aplikasi tanpa SLM (default aman)**

Semua halaman berjalan tanpa Ollama. Tombol laporan naratif (*Buat ringkasan naratif*, *Buat feedback siswa*,
*Jelaskan rekomendasi*) tetap menghasilkan laporan **template** deterministik dengan peringatan
"Laporan dibuat dengan template karena layanan SLM tidak tersedia". Untuk mematikan SLM sepenuhnya: set
`"enabled": false` di `configs/slm.json`.

**Langkah 7 — (Opsional) Laporan naratif dengan Qwen3-4B lokal**

1. Instal Ollama: https://ollama.com/download (Windows). Cek: `ollama --version`.
2. Unduh model (±2,5 GB). Di Ollama, Qwen3-4B-Instruct-2507 bertag **`qwen3:4b-instruct`**:
   ```powershell
   ollama pull qwen3:4b-instruct
   ollama list                          # harus memuat qwen3:4b-instruct
   ```
3. Jalankan server. Cara termudah (mengatur konteks 8192 token dan bind ke localhost):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_slm.ps1
   ```
   Bila memakai aplikasi tray Ollama: tambahkan variabel lingkungan pengguna `OLLAMA_CONTEXT_LENGTH=8192`
   (*Settings → System → About → Advanced system settings → Environment Variables*), lalu keluar dari Ollama dan buka lagi.
   Default 4096 token juga cukup untuk prompt saat ini, tetapi tanpa ruang cadangan.
4. Cek server: `curl.exe http://localhost:11434/v1/models` → daftar model termasuk `qwen3:4b-instruct`.
5. Restart Streamlit (generator SLM di-cache saat aplikasi berjalan), lalu klik tombol laporan naratif.
6. (Opsional) uji ke server nyata:
   ```powershell
   $env:CDACDM_SLM_TEST_URL="http://localhost:11434/v1"; pytest -q tests/test_narrative.py -k real
   Remove-Item Env:CDACDM_SLM_TEST_URL
   ```

Konfigurasi default (`configs/slm.json`) sudah benar untuk langkah di atas; detail, privasi, dan validator ada di
`README_slm.md`.

**Waktu & hasil yang diharapkan (laptop uji, GPU 4 GB):** laporan guru 30 siswa ±80–105 detik, laporan siswa ±80 detik;
model terbagi ±34% CPU / 66% GPU karena VRAM 4 GB. Laporan yang sama berikutnya diambil dari cache (< 1 detik). Bila server
mati, template muncul dalam ±2 detik. Batas baca 120 detik; lewat dari itu → template.

**Agar SLM cepat dan tidak timeout:**

* Laptop terhubung ke charger (pada baterai, CPU dibatasi dan model jauh lebih lambat).
* Jangan memakai model Ollama lain bersamaan (mis. opencode dengan `qwen2.5-coder`): VRAM 4 GB tidak cukup untuk dua model,
  Ollama akan menukar model dan laporan bisa melewati 120 detik. Cek dengan `ollama ps`; hentikan model lain dengan
  `ollama stop <nama>`.
* Setelah selesai, `ollama stop qwen3:4b-instruct` membebaskan GPU (Ollama menahan model 5 menit setelah request terakhir).

**Model lain yang sudah diuji (tidak disarankan):** `gemma3:1b` dan `qwen2.5:3b` 3–5× lebih cepat, tetapi mengarang atau
salah menempatkan angka (mis. 4 siswa ber-gap padahal 18; persentase konsep tertukar) sehingga ditolak validator dan jatuh
ke template. `qwen3:4b-instruct` adalah model terkecil yang lolos dengan angka benar.

### Masalah umum

| Gejala | Solusi |
|---|---|
| `ImportError: DLL load failed ... Application Control policy has blocked this file` | Smart App Control memblokir pandas: lihat Langkah 1 |
| `streamlit` / `pytest` tidak dikenal | venv belum aktif: `.venv\Scripts\activate`, atau pakai `.venv\Scripts\python.exe -m streamlit run app.py` |
| Aplikasi: "Checkpoint model belum tersedia" | folder `artifacts\model` tidak lengkap atau `CDACDM_ARTIFACT_DIR` salah: lihat Langkah 3 |
| Laporan naratif selalu template, alasan *tidak dapat dihubungi* | Ollama belum jalan: `ollama serve` atau buka aplikasi Ollama |
| Alasan *SLM HTTP 404* | nama model salah: `ollama list` dan samakan `model` di `configs/slm.json` |
| Alasan *SLM timeout (ReadTimeout)* | model lain memakai GPU atau laptop di baterai: lihat "Agar SLM cepat" di atas |
| Alasan *output SLM terpotong* | konteks server terlalu kecil: `OLLAMA_CONTEXT_LENGTH=8192`, restart Ollama |
| Alasan *validation failed: …* | model menulis angka/format yang tidak sesuai fakta; template ditampilkan (ini perilaku yang disengaja) |
| Perubahan `configs/slm.json` tidak berpengaruh | restart Streamlit |
| Port 8501 dipakai | `streamlit run app.py --server.port 8502` |

### Google Colab

1. Upload folder proyek ini ke `MyDrive/ILCDA/cda_cdm_project/` (tanpa `.venv*`, `ILCDA/`).
2. Pastikan artefak data preparation ada di `MyDrive/ILCDA/processed_task34_cda_cdm/final/` (kandidat path pertama di
   `configs/experiment.json`; bila lain, set `paths.data_dir` atau env `CDACDM_DATA_DIR`; bila tidak diset, pipeline mencari
   `q_matrix.csv` secara rekursif di `MyDrive/ILCDA`).
3. Buka `cda_cdm_model_training.ipynb` di Colab dan *Run all*. Cell pertama me-mount Drive, `cd` ke folder proyek, dan
   `pip install -r requirements.txt`. Output tersimpan di `cda_cdm_project/artifacts/` di Drive.
4. Aplikasi: lihat `README_streamlit.md` (bagian Colab).

## Audit data preparation (temuan)

* `eedi_task34_cda_cdm_pipeline.ipynb` memanggil `run_pipeline()` di cell 4 **tanpa definisinya** (juga `pd`, `np`, `hardest`
  tidak didefinisikan di notebook). Data preparation tidak dapat direproduksi dari notebook saja.
* Artefak Drive tidak dapat diakses dari lingkungan pengembangan ini, sehingga `cdacdm/reconstruct.py` membangun ulang artefak
  dengan nama file dan skema yang terlihat pada output notebook dari arsip resmi (MD5 arsip cocok dengan header Azure). Hasilnya
  cocok dengan output notebook: 4.918 siswa, 948 soal, 1.382.727 interaksi, 57 konsep (SubjectId level-3), 14 soal multi-konsep,
  densitas Q 0,0178, 144 soal fokus (= soal di bawah hierarki *Maths > Algebra*), split 3.934/492/492, dan class weight serta
  train correct rate **identik hingga 16 digit** (indikasi kuat split yang sama). Proxy usia 16–17: 5 (notebook: 6; definisi usia
  berbeda). Semua output menyimpan `artifact_source` sehingga hasil rekonstruksi tidak tertukar dengan artefak Drive.
* **Verifikasi terhadap artefak asli.** Salinan sebagian artefak Drive di `processed_task34_cda_cdm/final/` (q_matrix,
  interactions_train/val, eda_summary; tanpa npz, concept_metadata, item bank, interactions_test) cocok dengan rekonstruksi:
  nilai Q-matrix identik, keanggotaan siswa train/val identik, 1.112.450 + 129.242 interaksi tanpa satu pun selisih `IsCorrect`.
  Q-matrix asli memakai label kolom `c_0…c_56` (indeks konsep); loader mendukungnya.
* Loader **tidak berasumsi**: mendeteksi kunci npz, skema missing (`-1`, NaN, mask eksplisit; bila hanya {0,1} tanpa mask → ditolak
  sebagai ambigu), menyelaraskan kolom dengan `question_ids`, dan membandingkan matriks npz dengan `interactions_*.csv`
  (sumber utama). Q-matrix divalidasi (948 × 57 + QuestionId, biner, unik), kolom konsep dicocokkan ke `concept_metadata.csv`.
* Leakage: hanya `UserId, QuestionId, IsCorrect` yang dibaca untuk model; `question_pass_rate*`, `question_difficulty*`,
  `prior_accuracy*`, `student_mean_confidence`, `student_accuracy`, `question_attempts`, `question_confidence`, `option_entropy` dicatat sebagai *excluded_leakage_prone* (`audit/leakage_audit.json`). Statistik
  `cda_item_bank.csv` diverifikasi berasal dari train saja.

## Konsep target & cakupan materi

* 11 konsep aljabar EEDI menjadi target (`configs/target_concept_map.csv`); 46 konsep lain (Number, Geometry) **tidak dimasukkan**.
* Audit visual 22 soal fokus (2 per konsep): 17 cocok dengan label, 5 parsial — mis. dua soal berlabel *Solving Linear
  Inequalities* sebenarnya membandingkan panjang ruas dengan simbol >/<, bukan menyelesaikan pertidaksamaan; soal *Gradient*
  juga menguji panjang ruas. Semua sampel setara SMP. 122 soal fokus lain berstatus `not_reviewed`.
* Konsep EEDI hanya **prasyarat** aljabar SMA (satu overlap parsial: barisan aritmetika). **Tidak terpetakan**: eksponen &
  logaritma, polinomial, matriks, limit, turunan, integral (`audit/unmapped_target_topics.csv`). *Gradient Between Two
  Co-ordinates* hanya 2 soal → keterbatasan diagnosis.

## Model

Persamaan lengkap ada di docstring `cdacdm/models.py` dan notebook. Ringkasnya:
$P(X_{ij}=1\mid\alpha)=(1-s_j)^{\eta_{ij}}g_j^{1-\eta_{ij}}$, $\eta_{ij}=\prod_k\alpha_{ik}^{q_{jk}}$; keluarga variasional
$q(\alpha_i)=\prod_k\mathrm{Bern}(\pi_{ik})$ memberi ELBO tertutup tanpa enumerasi $2^{57}$; update koordinat eksak untuk soal
satu-konsep; M-step tertutup dengan prior Beta pada $g,s$; prior mastery *independent* atau *higher-order*. Likelihood hanya
menjumlahkan respons teramati (mask). Siswa baru = E-step saja dengan parameter beku. Baseline: item-mean, concept-score
(proporsi benar per konsep, di-smoothing), Q-logistic kompensatori. NeuralCDM tidak diimplementasikan: model yang dipakai sudah
memberi prediksi kompetitif dan setiap parameternya dapat ditafsirkan.

Kebijakan diagnosis (dari validation): τ via log-likelihood held-out prediksi DINA berlabel keras + aturan *one-standard-error*;
n_min dan δ via konsistensi klasifikasi split-half ≥ 0,85. Label: `mastered`, `not_mastered` (= learning gap), `uncertain`,
`insufficient_evidence` (bukan gap — butuh data). Form CDA otomatis memberi tiap konsep ≥ n_min soal.

## Hasil (dihitung oleh pipeline; lihat `artifacts/`)

Semua angka di bawah berasal dari run notebook `cda_cdm_model_training.ipynb` (24-10-2026, artefak `local_reconstruction`,
510 detik di CPU). Bila notebook dijalankan pada artefak Drive, angkanya dihitung ulang dari data tersebut.

**Seleksi (validation):** P-DINA prior *higher-order*, kekuatan prior Beta g/s = 2, tanpa class weighting (class weighting
memperburuk log-loss validation di semua konfigurasi). Kebijakan: τ = 0,5; n_min = 5 respons per konsep; δ = 0.

**Prediksi respons — test (492 siswa, 28.003 respons held-out, dievaluasi sekali):**

| Model | ROC-AUC [95% CI] | Log-loss | Brier | Accuracy | Precision | Recall | F1 | ECE |
|---|---|---|---|---|---|---|---|---|
| **P-DINA (utama)** | 0,768 [0,759–0,779] | 0,568 | 0,194 | 0,699 | 0,743 | 0,669 | 0,704 | 0,008 |
| P-DINA prior independen (ablasi) | 0,761 [0,751–0,772] | 0,575 | 0,197 | 0,693 | 0,733 | 0,672 | 0,701 | 0,006 |
| Q-logistic (baseline) | 0,775 [0,766–0,786] | 0,567 | 0,193 | 0,704 | 0,721 | 0,731 | 0,726 | 0,036 |
| Concept-score (baseline) | 0,718 [0,707–0,733] | 0,611 | 0,213 | 0,658 | 0,682 | 0,681 | 0,681 | 0,016 |
| Item-mean (baseline) | 0,672 [0,661–0,682] | 0,644 | 0,226 | 0,627 | 0,642 | 0,689 | 0,665 | 0,005 |

Selisih berpasangan (bootstrap per siswa): P-DINA vs Q-logistic AUC −0,007 [−0,011; −0,003], log-loss +0,001 [−0,003; +0,007]
(tidak berbeda bermakna); P-DINA jauh lebih baik dari concept-score dan item-mean. In-sample train (optimistis): AUC 0,800.

**Kualitas diagnosis (proksi; tidak ada ground truth mastery):**
* Konsistensi klasifikasi split-half pada siswa test (konsep target, ≥ 5 respons per paruh): **0,876** (880 sel).
* SIMULASI dari model terlatih: akurasi atribut 0,957 bila ada bukti, 0,851 tanpa bukti; recovery parameter soal r(g) = 0,965,
  r(s) = 0,982. Ini batas atas optimistis (model diasumsikan benar).

**CDA:** 139 soal target layak (≥ 30 respons train, daya beda ≥ 0,1). Fixed form 52 soal (5 per konsep, *Gradient* hanya 2).
Parallel form A/B masing-masing 45 soal dengan selisih difficulty per konsep ≤ 0,025; 6 dari 11 konsep tidak punya cukup soal
untuk dua form penuh (< 10 soal). Replay adaptif pada 300 siswa test: kesepakatan label dengan diagnosis dari semua respons
0,862 (adaptif, 10 soal) vs 0,824 (acak); 0,911 vs 0,871 pada 20 soal.

**Diagnosis siswa test (label model, bukan mastery terbukti):** dari 5.412 sel siswa × konsep target: 727 *mastered*,
1.394 *not_mastered* (gap), 3.291 *insufficient_evidence*; 322 dari 492 siswa punya ≥ 1 gap. Di antara siswa yang dapat
didiagnosis, proporsi gap per konsep 55–80%. *Gradient Between Two Co-ordinates* tidak dapat didiagnosis untuk satu pun siswa.

**Demo pre/post (SIMULASI, 40 siswa sintetis):** korelasi perubahan P(mastery) terestimasi vs perubahan mastery sebenarnya
(diketahui karena disimulasikan) = 0,744.

## Status fitur

| Kategori | Fitur |
|---|---|
| **Berjalan dengan data EEDI** | audit artefak & leakage; analisis Q-matrix; peta konsep target; training P-DINA + 3 baseline; seleksi hyperparameter & kebijakan di validation; evaluasi test (AUC, accuracy, precision, recall, F1, log-loss, Brier, ECE, kalibrasi, per konsep, bootstrap CI); konsistensi split-half; profil mastery, learning gap & agregasi kelas siswa test; fixed/parallel forms; replay adaptif; inferensi siswa baru/parsial; aplikasi Streamlit |
| **Demonstrasi berbasis data sintetis** | analisis pre/post (`artifacts/prepost/SYNTHETIC_*`, halaman Pre/Post mode SIMULASI); simulasi recovery mastery & parameter soal (berlabel *SIMULATION*) |
| **Placeholder yang wajib diisi/divalidasi** | `configs/intervention_map.csv` (sumber materi, aktivitas, durasi); `prerequisite_concept_ids` & `curriculum_order` di peta konsep (draft); `configs/target_curriculum.csv` (draft topik SMA) |
| **Baru dapat dipakai setelah data kelas nyata tersedia** | hasil pre/post penelitian (`compare_pre_post` dengan CSV pre & post, ID siswa sama, opsional `group`); validasi model pada siswa SMA; diagnosis topik SMA yang tidak ada di EEDI (butuh soal baru yang dipetakan ke Q-matrix dan dikalibrasi); evaluasi efektivitas intervensi (butuh desain dengan kelompok pembanding) |

## Keterbatasan

* Tidak ada ground-truth mastery di EEDI: kualitas diagnosis dilaporkan sebagai konsistensi split-half dan simulasi, bukan akurasi.
* Item EEDI berdaya beda sedang (rata-rata g≈0,33, s≈0,24 untuk soal pilihan ganda 4 opsi), sehingga diagnosis per konsep butuh
  ±5 respons; form yang lebih pendek menghasilkan banyak konsep `insufficient_evidence`.
* Q-logistic kompensatori sedikit lebih unggul AUC-nya dibanding P-DINA (lihat hasil); P-DINA dipertahankan karena kalibrasinya
  lebih baik dan parameternya langsung bermakna (guess/slip, probabilitas penguasaan) — trade-off ini dilaporkan, tidak disembunyikan.
* Pre/post tanpa kelompok kontrol = *perubahan yang teramati*, bukan efek kausal.
