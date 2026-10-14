# Maize crop-identity classifier

This adds an optional MobileNetV3-Small gate before the existing CLIP maize validator. It does not retrain or replace Swin, LSTM, or LightGBM. The gate is deliberately disabled until a trained checkpoint is available and held-out evaluation has been reviewed.

## Dataset layout

Create this local layout (images are intentionally not committed to Git):

```text
data/maize_crop_classifier/
  train/
    maize/
    non_maize/
  val/
    maize/
    non_maize/
  test/
    maize/
    non_maize/
```

Use verified maize leaves across healthy/damaged conditions, lighting, backgrounds, and growth stages. Include diverse non-maize leaves and unrelated non-leaf images. Reserve whole plant species and near-duplicate-free images for the test split so the evaluation tests generalization rather than memorization. Review image provenance and licences.

## Train

Run from the repository root using the project's active virtual environment:

```powershell
python training/train_maize_crop_classifier.py --epochs 8 --batch-size 32
```

The first run downloads the official torchvision MobileNetV3-Small pretrained weights if they are not cached. Training freezes the feature extractor and trains only the final classification layer. The script selects a maize-probability threshold on the validation split to target a maximum non-maize false-accept rate (default 5%) while retaining as many validation maize images as possible. This target is not a guarantee for unseen species.

Checkpoint output: `models/maize_crop_classifier.pth` (ignored by Git; do not commit model binaries without an explicit repository decision).

## Evaluate on held-out images

```powershell
python training/evaluate_maize_crop_classifier.py
```

Review the reported maize acceptance rate and non-maize false-accept rate, especially for plant species absent from train/validation. If false acceptance is too high or genuine maize rejection is too high, improve data quality/diversity and retrain; do not tune the threshold against the test set.

## Enable the API gate

Only after reviewing held-out results, set the environment variable before starting/restarting the backend:

```powershell
$env:AGRISHIELD_MAIZE_CLASSIFIER_ENABLED = "1"
python -m uvicorn app.main:app --reload
```

To disable the new gate in the current PowerShell session:

```powershell
Remove-Item Env:AGRISHIELD_MAIZE_CLASSIFIER_ENABLED -ErrorAction SilentlyContinue
```

When enabled, both `/api/predict/hybrid` and `/api/predict/image` check this model before CLIP and reject uncertain/non-maize predictions with HTTP 422. If the checkpoint is missing or cannot load, the gate fails closed with HTTP 503. When the environment variable is unset, the current CLIP validator and existing prediction pipeline operate as before.

## Important limitations

- Softmax scores are model scores, not calibrated probabilities of biological identity.
- A finite dataset cannot guarantee rejection of every unknown plant species.
- Test with unseen species and real field images before making accuracy claims.
- This gate answers crop identity only; it does not change the existing healthy/infected classifier or FAW-risk models.
