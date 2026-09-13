# Deepfake Identity Detection Baseline

A temporal artifact-classification baseline for research on identity consistency across
video frames. It consumes precomputed frame embeddings, which keeps the repository
focused on modeling and evaluation rather than shipping face data.

## What it demonstrates

- Temporal 1D convolutions over frame-level embeddings
- Mask-aware pooling for variable-length clips
- Binary logits for real/fake classification
- A privacy-aware synthetic demo with no media files

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

This is an educational baseline, not a forensic service. Use consented data, document
dataset provenance, and evaluate identity leakage and demographic performance before
making claims.

## Structure

- src/model.py — temporal classifier
- src/demo.py — synthetic frame-embedding example

## License

Apache-2.0
