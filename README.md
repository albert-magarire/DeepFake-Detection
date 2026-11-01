# Installation

Use Python 3.10-3.12.

```bash
git clone https://github.com/albert-magarire/DeepFake-Detection.git
cd DeepFake-Detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install the matching `torch==2.8.0` and `torchvision==0.23.0` CUDA builds using the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) before installing the requirements. The scripts select CUDA when available; use `--device cpu` to run on CPU.

## Train

Prepare the image folders below, then run:

```bash
python train.py --data-dir data --epochs 50 --batch-size 16
```

Training downloads ImageNet weights for EfficientNet-B4 on first use. Use `--no-pretrained` to initialize all weights locally. Reduce `--batch-size` if GPU memory is limited (minimum 2).

`checkpoints/best.pt` is selected by validation macro F1. `checkpoints/last.pt` contains the last completed epoch. `checkpoints/config.json` records the settings. Choose a new `--output-dir` for each run; existing runs are not overwritten. These checkpoints are for evaluation, not optimizer-state resumption.

## Test

```bash
python test.py --data-dir data/test --checkpoint checkpoints/best.pt
```

The command prints accuracy, AUC, macro F1, per-class precision/recall/F1 and a confusion matrix with real/fake row and column order. AUC is `null` when only one class is present. To save per-image predictions:

```bash
python test.py --data-dir data/test --checkpoint checkpoints/best.pt --predictions outputs/predictions.csv
```

To classify one image:

```bash
python test.py --image path/to/image.jpg --checkpoint checkpoints/best.pt
```

Use a checkpoint created by `train.py`, or the bundled `backend/weights.pth`. The loader selects the matching B4 or B0 architecture and preprocessing. A fake probability greater than or equal to `0.5` maps to fake; otherwise it maps to real. Change the decision threshold with `--threshold`.

## Run the web app

Start the API from the repository root:

```bash
python main.py
```

In a second terminal, install and start the frontend with Node.js 22.12+:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend uploads images to `http://127.0.0.1:8000/predict`. API documentation is at `http://127.0.0.1:8000/docs`.

The included `backend/weights.pth` is copied unchanged from the [original application](https://github.com/joshkgrant2003/DeepfakeDetector). It requires **EfficientNet-B0 + FFT**. Training in this repository defaults to **EfficientNet-B4 + FFT**; B0 weights cannot initialize B4. The default app uses the bundled weights and their original PIL resize/normalization. To serve a newly trained B4 checkpoint instead, set `MODEL_CHECKPOINT` before starting the API:

```powershell
# Windows PowerShell, from the repository root
$env:MODEL_CHECKPOINT = "checkpoints/best.pt"
python main.py
```

```bash
# macOS / Linux, from the repository root
MODEL_CHECKPOINT=checkpoints/best.pt python main.py
```

Alternatively, run `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` from the repository root. Set `CORS_ORIGINS` to a comma-separated list of frontend origins when changing ports or hosting. Set `VITE_HOST` (or `VITE_LOCAL_HOST`) in the frontend environment to change the API URL; restart Vite or rebuild after changing it. Local origins on port 5173 are allowed by default. Images must be no larger than 10 MB.

The API retains the original response fields: `label` is `Real` or `Fake`, and `confidence` is the probability of fake. The frontend converts that value into confidence for the displayed label. Run `npm run build` to build the frontend, or `npm run lint` to check it.

## Data

Supply RGB images or extracted frames in this layout. Nested directories are supported.

```text
data/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

Supported formats: JPG, JPEG, PNG, BMP and WebP. Labels are `real=0` and `fake=1`. Training and validation require both classes; evaluation requires both folders but permits one to be empty. Images are converted to RGB; no face cropping is performed.

Assign the splits before training. Keep each source video, its manipulated versions and shared identities in one split. Do not sample test images from the training or validation pools. The loader checks repeated resolved file paths across local splits; it does not infer identities, match related videos or detect renamed copies.

To evaluate another image dataset, arrange its held-out images under `real/` and `fake/` and pass that directory to `test.py --data-dir`.

## Settings

`model.py` combines EfficientNet-B4 spatial features with a three-layer CNN over the centered log-magnitude FFT of the normalized RGB tensor. The spatial and frequency vectors are scaled by `0.8` and `0.2`, concatenated, and passed through a `512 → 128 → 1` classifier.

Default training uses 380-pixel inputs, AdamW, weight decay `1e-4`, spatial learning rate `5e-6`, and frequency/classifier learning rate `1e-4`. Spatial parameters are frozen for the first three epochs; batch-normalization running statistics still update. The optimizer is rebuilt when spatial parameters are unfrozen. Training uses class-balanced sampling with replacement, binary cross-entropy with logits, and drops the final incomplete training batch.

Training augmentation applies random resized crops, blur, JPEG compression and color jitter, followed by ImageNet normalization. Validation and testing use resizing and the same normalization. Seed 42 is used by default; exact numerical reproducibility across hardware is not guaranteed.

Run `python train.py --help` or `python test.py --help` for available options.
