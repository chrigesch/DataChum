import joblib


def save_complete_pipeline(pipeline, filename: str):
    return joblib.dump(pipeline, filename)


def load_complete_pipeline(filename: str):
    joblib.load(filename)
