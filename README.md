# Anonymous Code Repository

Anonymous code repository for the paper SoziaSign: A Privacy-Preserving Turkish Sign Language Recognition and Translation Pipeline

## Table of Contents

- Repository Structure
- Requirements
- Data Preparation
- Usage
- Optional Batch Scripts
- References

## Repository Structure

```text
.
|-- tsl-recognition/
|   |-- configs/
|   `-- src/
`-- tsl-translation/
    |-- config/
    |-- data/
    |   |-- raw/
    |   `-- processed/
    `-- src/
```

## Requirements

### Recognition Module

Create the environment with:

```bash
conda env create -f tsl-recognition/configs/environment.yml
conda activate tsl-recognition
```

### Translation Module

Install dependencies with:

```bash
cd tsl-translation
pip install -r requirements.txt
```

Some translation experiments expect external credentials through environment variables, such as `HF_TOKEN` or `GEMINI_API_KEY`.

## Data Preparation

### Translation Data

Place the raw translation data at:

```text
tsl-translation/data/raw/slr_gloss_tr_cleaned.jsonl
```

Processed files are expected at:

```text
tsl-translation/data/processed/train.jsonl
tsl-translation/data/processed/valid.jsonl
```

To regenerate the processed train/validation split:

```bash
cd tsl-translation
python src/prepare_data.py --force
```

### Recognition Data

Prepare the recognition datasets according to their official distributions, then configure local dataset paths before running extraction, training, or evaluation.

## Usage

### Recognition Module

Run the CLI from the recognition project root:

```bash
cd tsl-recognition
python -m src --help
```

Common commands:

```bash
python -m src split
python -m src extract
python -m src train
python -m src evaluate
python -m src infer --mode motion
python -m src validate
```

### Translation Module

Example commands:

```bash
cd tsl-translation
python src/prepare_data.py
python src/base_model_bench.py --model_id <model_id>
python src/unified_bench.py --model_id <model_id> --strategy <strategy>
python src/unified_bench_tr.py --model_id <model_id> --strategy <strategy>
python src/rag_bench.py
```

## Optional Batch Scripts

The files under `tsl-translation/src/*.sh` are optional helper launchers for Unix-like or SLURM-based environments. They are included for reproducibility of experiment orchestration, but the canonical implementation lives in the Python entry points in `tsl-translation/src/`.

## References

If you want to use the recognition component, you can access their datasets here:

```text
[11] Ogulcan Ozdemir, Ahmet Alp Kindiroglu, Necati Cihan Camgoz, and Lale Akarun.
2020. BosphorusSign22k sign language recognition dataset. In Proceedings of the
9th Workshop on the Representation and Processing of Sign Languages (LREC 2020).
European Language Resources Association (ELRA), Marseille, France, 181-188.

[14] Ozge Mercanoglu Sincan and Hacer Yalim Keles. 2020. AUTSL: A large scale
multi-modal Turkish sign language dataset and baseline methods. IEEE Access 8
(2020), 181340-181355. https://doi.org/10.1109/ACCESS.2020.3028072
```
