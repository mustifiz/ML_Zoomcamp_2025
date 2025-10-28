import os
import pickle
import urllib.request


PIPELINE_PATH = "pipeline_v1.bin"
PIPELINE_URL = (
    "https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin"
)


def ensure_pipeline(path: str, url: str):
    if os.path.exists(path):
        return
    print(f"Pipeline not found at {path}, downloading from {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved pipeline to {path}")


def load_pipeline(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def score_record(pipeline, record: dict) -> float:
    # pipeline expects a list of dicts for transform/predict_proba
    proba = pipeline.predict_proba([record])
    # return probability of the positive class
    return float(proba[0, 1])


def main():
    ensure_pipeline(PIPELINE_PATH, PIPELINE_URL)

    pipeline = load_pipeline(PIPELINE_PATH)

    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
    }

    p = score_record(pipeline, record)
    print(f"Conversion probability: {p:.3f}")

    # Round to 3 decimals and match provided choices
    choices = [0.333, 0.533, 0.733, 0.933]
    diffs = [abs(p - c) for c in choices]
    best = choices[diffs.index(min(diffs))]
    print(f"Closest provided choice: {best:.3f}")


if __name__ == "__main__":
    main()
