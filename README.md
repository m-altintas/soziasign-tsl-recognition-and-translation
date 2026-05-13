# SoziaSign — TSL Recognition and Translation

SoziaSign is a privacy-preserving Turkish Sign Language recognition and translation pipeline. It extracts 507-dimensional skeletal landmark vectors on-device, classifies them with a 5-layer GRU, and translates recognized glosses into Turkish sentences using a LoRA fine-tuned LLM. No raw video leaves the device.

The recognition component achieves **88.13% top-1 accuracy** on AUTSL (226 classes, signer-independent) and **72.86%** on BosphorusSign22k (744 classes). The best translation configuration (Gemma-2-9B-it, P3_EN prompt, LoRA) reaches an **87.19% success rate**, beating Gemini Pro on the binary pass/fail criterion.

## About this repository

Camera-ready code release for:

> **SoziaSign: A Privacy-Preserving Turkish Sign Language Recognition and Translation Pipeline**
> Mehmet Altıntaş, Aylin Barutçu, Mehmet Karatekin, İlbey Efe Taşabatlı
> MLMI 2026

Two independently installable pipelines:

| Directory | Pipeline |
|---|---|
| `tsl-recognition/` | MediaPipe landmark extraction + GRU-based sign classification |
| `tsl-translation/` | LoRA fine-tuning and evaluation of LLMs for gloss-to-text translation |

## Related projects

- **[sozia-research](https://github.com/Last-Branch/sozia-research)** — the actively maintained research repo this code derives from. Includes additional experiments (lip-reading, updated configs) and ongoing work.
- **[Sozia](https://github.com/Last-Branch/sozia-server)** — the end-to-end system that integrates this pipeline for real-time inference on Android.

## Repository structure

```
soziasign-tsl-recognition-and-translation/
├── tsl-recognition/
│   ├── pyproject.toml             # installable package: pip install -e .
│   ├── configs/
│   │   ├── environment.yml        # conda environment definition
│   │   └── recognition_train.yml  # training hyperparameters
│   ├── tsl_recognition/           # recognition pipeline package
│   │   ├── cli.py                 # subcommand entry point
│   │   ├── config.py              # TrainConfig dataclass
│   │   ├── dataset/               # AUTSL and BosphorusSign22k adapters
│   │   ├── extraction/            # MediaPipe landmark extraction
│   │   ├── evaluation/            # training, evaluation, inference, validation
│   │   └── models/                # GRU architecture
│   └── tests/
│       └── test_recognition_smoke.py
└── tsl-translation/
    ├── pyproject.toml             # installable package: pip install -e .
    ├── requirements.txt           # flat dep list for pip install -r
    ├── configs/
    │   └── translation_bench.yml  # benchmark / fine-tuning settings
    ├── gloss_to_text/             # translation pipeline package
    │   ├── prompts/strategies.py  # P1–P3 × EN/TR prompt templates
    │   ├── fine_tuning/           # LoRA fine-tuning + evaluation
    │   └── evaluation/            # baseline, RAG, and Gemini judge
    ├── scripts/
    │   └── prepare_data.py        # raw JSONL → train/valid split
    ├── data/
    │   ├── raw/                   # slr_gloss_tr_cleaned.jsonl
    │   └── processed/             # train.jsonl, valid.jsonl
    ├── experiments/               # reported experiment outputs (REPORT.json + per-model runs)
    └── tests/
        └── test_translation_smoke.py
```

## Reproducing the paper results

Pre-trained weights are not included. To reproduce the numbers, run the pipeline in order:

1. Get the datasets (see [Data preparation](#data-preparation) below)
2. Extract MediaPipe landmarks: `python -m tsl_recognition extract`
3. Train the GRU classifier: `python -m tsl_recognition train`
4. Evaluate on the held-out test set: `python -m tsl_recognition evaluate`

Both datasets require an access request to their authors — links are in [References](#references).

## Installation

### Recognition pipeline

```bash
cd tsl-recognition
conda env create -f configs/environment.yml
conda activate tsl-recognition
pip install -e .
```

The conda environment installs a CPU-only PyTorch wheel by default. If you have a GPU, follow the CUDA-specific install at [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -e .`.

To run the smoke tests:

```bash
pip install -e ".[dev]"
pytest
```

### Translation pipeline

```bash
cd tsl-translation
pip install -e .
# or, using the flat requirements file:
pip install -r requirements.txt
```

## Data preparation

### Recognition data

Prepare BosphorusSign22k and AUTSL according to their official distributions, then place them under `tsl-recognition/data/`:

```
data/BosphorusSign22k/
├── BosphorusSign22k_classes.csv
├── BosphorusSign22k.csv
└── raw/
    ├── 0001/
    │   ├── User_2_001.mp4
    │   └── ...
    └── ...

data/AUTSL/
├── SignList_ClassId_TR_EN.csv
├── train_labels.csv
├── validation_labels.csv
├── test_labels.csv
├── train/
├── val/
└── test/
```

Extraction writes output to `data/{Dataset}/processed/{ClassName_tr}/{sample}.npy`.

### Translation data

Place the raw data at:

```
tsl-translation/data/raw/slr_gloss_tr_cleaned.jsonl
```

Then generate the processed train/validation split:

```bash
cd tsl-translation
python scripts/prepare_data.py          # skips if files already exist
python scripts/prepare_data.py --force  # regenerate from scratch
```

Processed files land at `data/processed/train.jsonl` and `data/processed/valid.jsonl`.

## Usage

### Recognition pipeline

Run all commands from the `tsl-recognition/` directory:

```bash
# Compute dataset splits
python -m tsl_recognition split

# Extract MediaPipe landmarks from raw videos
python -m tsl_recognition extract

# Train the GRU classifier
python -m tsl_recognition train

# Evaluate on the test set
python -m tsl_recognition evaluate

# Run inference on a video file
python -m tsl_recognition infer --mode motion

# Validate landmark extraction quality
python -m tsl_recognition validate
```

### Translation pipeline

Run all commands from the `tsl-translation/` directory:

```bash
# Baseline evaluation (no fine-tuning)
python -m gloss_to_text.evaluation.base_model_bench --model_id google/gemma-2-9b-it

# LoRA fine-tuning + evaluation (EN prompt — best configuration)
python -m gloss_to_text.fine_tuning.unified_bench \
    --model_id google/gemma-2-9b-it --strategy P3_EN

# LoRA fine-tuning + evaluation (TR prompt)
python -m gloss_to_text.fine_tuning.unified_bench \
    --model_id google/gemma-2-9b-it --strategy P3_TR \
    --use_autocast --no_optim --no_grad_checkpointing

# RAG baseline
python -m gloss_to_text.evaluation.rag_bench

# Score predictions with the Gemini judge
python -m gloss_to_text.evaluation.gemini_judge
```

## Environment variables

| Variable | Required by |
|---|---|
| `HF_TOKEN` | Any Hugging Face model download |
| `GEMINI_API_KEY` | `gemini_judge.py` |

Place them in a `.env` file at the respective pipeline root or export them in your shell.

## Citation

```bibtex
@inproceedings{altintas2026soziasign,
  title     = {SoziaSign: A Privacy-Preserving Turkish Sign Language Recognition and Translation Pipeline},
  author    = {Altıntaş, Mehmet and Barutçu, Aylin and Karatekin, Mehmet and Taşabatlı, İlbey Efe},
  booktitle = {Proceedings of MLMI 2026},
  year      = {2026},
}
```

## References

```
[11] Ogulcan Ozdemir, Ahmet Alp Kindiroglu, Necati Cihan Camgoz, and Lale Akarun.
     2020. BosphorusSign22k sign language recognition dataset. In Proceedings of the
     9th Workshop on the Representation and Processing of Sign Languages (LREC 2020).
     European Language Resources Association (ELRA), Marseille, France, 181-188.

[14] Ozge Mercanoglu Sincan and Hacer Yalim Keles. 2020. AUTSL: A large scale
     multi-modal Turkish sign language dataset and baseline methods. IEEE Access 8
     (2020), 181340-181355. https://doi.org/10.1109/ACCESS.2020.3028072
```
