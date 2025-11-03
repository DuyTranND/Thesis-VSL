# Vietnamese Sign Language Recognition with EfficientGCN

Skeleton-based Vietnamese Sign Language Recognition (VSLR) using EfficientGCN. The end-to-end pipeline includes K-Means keyframe selection, 61-landmark extraction with MediaPipe, graph-centric cleaning and augmentation, training EfficientGCN family models (B0, B2, B4), and a simple web demo built with Streamlit + FastAPI.

<p align="center">
  <img src="docs/pipeline.png" alt="Pipeline" width="85%"><br>
  <em>Overview: Video -> 24 frames -> 61 landmarks -> 3 input branches -> EfficientGCN -> Prediction</em>
</p>

---

## Features

- Three input branches: **Joints**, **Velocity**, **Bones**
- **24 frame** selection via K-Means for better motion coverage
- **61 landmarks** per frame from MediaPipe (pose 19 + two hands 21x2)
- Cleaning: left-right hand swap fix, missing-hand completion, coordinate normalization
- Graph-aware augmentation: symmetric flip with joint remapping, region-wise scaling, mild temporal jitter, BFS-guided reconstruction
- EfficientGCN variants **B0**, **B2**, **B4** with **ST-JointAtt**
- Web demo: **Streamlit** frontend, **FastAPI** backend

---

## Repository Structure

```text
.
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ b0.yaml
│  ├─ b2.yaml
│  └─ b4.yaml
├─ data/
│  ├─ raw/                # source videos
│  ├─ interim/            # per-frame JSON/NPZ skeletons
│  └─ processed/          # tensors for EfficientGCN (3 branches)
├─ src/
│  ├─ datasets/           # dataset wrappers, collate, graph utils
│  ├─ models/             # EfficientGCN + ST-JointAtt
│  ├─ preprocess/         # kmeans sampling, mediapipe, clean, augment
│  ├─ train.py            # training loop
│  ├─ eval.py             # evaluation
│  └─ infer.py            # single-video inference
├─ scripts/
│  ├─ prepare_data.py     # raw video -> processed tensors (end-to-end)
│  └─ visualize_skel.py   # quick skeleton visualization
└─ web/
   ├─ api/                # FastAPI (uvicorn)
   │  └─ main.py
   └─ app.py              # Streamlit app
```

> If your repo layout differs, keep the step logic and adjust paths in `configs/*.yaml`.

---

## Quickstart

### 1) Setup

```bash
# Python 3.10+ is recommended
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### 2) Data Preparation

Put your videos into `data/raw/`, then run:

```bash
# Steps inside the script:
# 1) K-Means to pick 24 frames per video
# 2) MediaPipe to extract 61 landmarks per frame
# 3) Cleaning (swap-fix, missing-hand completion), normalization
# 4) Graph-aware augmentation
# 5) Produce 3-branch tensors in data/processed/
python scripts/prepare_data.py \
  --raw_dir data/raw \
  --interim_dir data/interim \
  --out_dir data/processed \
  --num_frames 24
```

Visual sanity-check for extracted skeletons:

```bash
python scripts/visualize_skel.py --interim_dir data/interim --sample 3
```

### 3) Train

Pick a config from `configs/`:

```bash
# Train B0
python src/train.py --config configs/b0.yaml \
  --data_dir data/processed \
  --work_dir workdir/b0
```

Resume from the latest checkpoint:

```bash
python src/train.py --config configs/b0.yaml --work_dir workdir/b0 --resume
```

### 4) Evaluate

```bash
python src/eval.py --config configs/b0.yaml \
  --data_dir data/processed \
  --checkpoint workdir/b0/checkpoint.pth
```

### 5) Inference on a Single Video

```bash
python src/infer.py \
  --video path/to/video.mp4 \
  --checkpoint workdir/b0/checkpoint.pth \
  --out_json result.json
```

### 6) Web Demo

```bash
# Backend
uvicorn web.api.main:app --host 0.0.0.0 --port 8000

# Frontend
streamlit run web/app.py
```

---

## Sample Configuration

`configs/b0.yaml`:

```yaml
model:
  name: EfficientGCN
  variant: B0                # B0, B2, or B4
  st_joint_att: true
  num_classes:  # fill with your number of signs

data:
  dir: data/processed
  num_frames: 24
  in_branches: [joints, velocity, bones]
  train_split: train.json
  val_split: val.json

train:
  epochs: 30
  batch_size: 64
  lr: 1e-3
  wd: 1e-4
  optimizer: adamw
  scheduler: cosine

augment:
  flip: true
  time_jitter: 0.1
  scale_region: {body: 0.02, hand: 0.03}
  bfs_reconstruct: true

misc:
  seed: 42
  num_workers: 4
  amp: true
```

---

## EfficientGCN Input Specification

- **Tensor shape**: `[N, C, T, V, M]`
  - `N`: batch size
  - `C`: channels (2D or 3D coords, optional confidence)
  - `T`: temporal length (24)
  - `V`: number of joints
  - `M`: number of people (usually 1)
- **Joints**: normalized coordinates anchored to a stable reference (nose or wrist).
- **Velocity**: temporal differences of joints, same normalization.
- **Bones**: vectors along graph edges (parent -> child).
- **Graph**: vertex set `V`, edge set `E`, adjacency matrices for pose and both hands. Horizontal flip requires symmetric joint remapping.

> If you use 2D MediaPipe coordinates, keep normalization and confidence masking consistent to reduce noise.

---

## Results (to be updated)

- With the 24-frame, 61-landmark pipeline, EfficientGCN shows fast convergence and strong validation accuracy.
- With B0 plus graph-aware augmentation, validation typically stabilizes around epoch 6-7.
- Exact metrics depend on the number of classes, train-val-test split, and augmentation strategy.

> Add your official logs and plots to `docs/results.md` and embed a loss/accuracy figure here once available.

---

## Practical Tips

- Enable `--amp` for mixed precision speedups.
- Set seeds and `cudnn.deterministic` when you need reproducibility.
- Carefully verify left-right mapping when applying horizontal flips.
- For very short or long videos, K-Means keyframe selection often outperforms uniform sampling.

---

## Roadmap

- [ ] Add face landmarks to capture lip shapes and expressions
- [ ] Few-shot augmentation for rare classes
- [ ] Export TorchScript or ONNX and add a realtime demo

---

## Citation

If you use EfficientGCN or this pipeline in your research, please cite:

```bibtex
@article{song2022constructing,
  author    = {Song, Yi-Fan and Zhang, Zhang and Shan, Caifeng and Wang, Liang},
  title     = {Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2022},
  doi       = {10.1109/TPAMI.2022.3157033}
}
```

---

## License

MIT License. See `LICENSE` for details.

## Acknowledgements

- EfficientGCN and the broader skeleton-based action recognition community
- MediaPipe for pose and hand landmark extraction
- Open-source contributors who made this work possible
```
