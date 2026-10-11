
from pathlib import Path

import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "validator_dataset"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
OLD_PROMPT = (
    "an infected maize leaf with visible pest or disease damage"
)

NEW_PROMPTS = [
    "a maize leaf damaged by fall armyworm feeding, with irregular holes and chewed edges",
    "a corn leaf showing fall armyworm damage with ragged holes and windowpane feeding",
    "a close-up photograph of a maize leaf with visible insect feeding damage",
]

MAIZE_PROMPTS = ["a healthy maize leaf", OLD_PROMPT] + NEW_PROMPTS

OTHER_PROMPTS = [
    "a leaf from an unrelated plant species",
    "a person",
    "a building or an urban structure",
    "a landscape or outdoor scenery",
    "a pet or other animal",
    "an unrelated everyday object",
]

def main():
    folders = [
        DATASET / "maize_test" / "maize_healthy",
        DATASET / "maize_test" / "maize_infected",
        DATASET / "non_maize",
    ]

    images = sorted(
        p for folder in folders if folder.exists()
        for p in folder.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise SystemExit("No test images found in validator_dataset.")

    print(f"Loading {MODEL_NAME} ({PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    all_prompts = MAIZE_PROMPTS + OTHER_PROMPTS
    tokens = tokenizer(all_prompts)

    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    totals = {
        "baseline": {"maize_ok": 0, "maize_bad": 0, "other_ok": 0, "other_bad": 0},
    }
    for prompt in NEW_PROMPTS:
        totals[prompt] = {
            "maize_ok": 0, "maize_bad": 0, "other_ok": 0, "other_bad": 0
        }

    print(f"Evaluating {len(images)} images...")

    for index, path in enumerate(images, start=1):
        try:
            with Image.open(path) as im:
                image = preprocess(im.convert("RGB")).unsqueeze(0)
        except Exception as exc:
            print(f"SKIPPED {path}: {exc}")
            continue

        with torch.inference_mode():
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            scores = (features @ text_features.T).squeeze(0).tolist()

        maize_expected = "maize_test" in str(path).lower()
        other_scores = scores[len(MAIZE_PROMPTS):]
        unrelated_best = max(other_scores)

        # Baseline: healthy maize or original infected-maize prompt.
        baseline_maize = max(scores[0], scores[1])
        baseline_accept = baseline_maize > unrelated_best

        if maize_expected:
            key = "maize_ok" if baseline_accept else "maize_bad"
        else:
            key = "other_bad" if baseline_accept else "other_ok"
        totals["baseline"][key] += 1

        # Each candidate replaces the original infected-maize prompt.
        for i, prompt in enumerate(NEW_PROMPTS, start=2):
            candidate_maize = max(scores[0], scores[i])
            accepted = candidate_maize > unrelated_best

            if maize_expected:
                key = "maize_ok" if accepted else "maize_bad"
            else:
                key = "other_bad" if accepted else "other_ok"
            totals[prompt][key] += 1

        if index % 10 == 0:
            print(f"Processed {index}/{len(images)}")

    print("\nRESULTS — MAIZE ACCEPTANCE / NON-MAIZE REJECTION")
    print("Decision rule: best maize score must beat every non-maize score.")
    print("Scores are cosine similarities, not probabilities.\n")

    for name, result in totals.items():
        maize_total = result["maize_ok"] + result["maize_bad"]
        other_total = result["other_ok"] + result["other_bad"]
        print(f"Prompt: {name}")
        print(f"  Maize accepted: {result['maize_ok']}/{maize_total}")
        print(f"  Maize rejected: {result['maize_bad']}/{maize_total}")
        print(f"  Non-maize rejected: {result['other_ok']}/{other_total}")
        print(f"  Non-maize accepted incorrectly: {result['other_bad']}/{other_total}")
        print()

if __name__ == "__main__":
    main()
