# Import the required Libraries
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.mcd import MCD


AVAILABLE_MODELS_ANOMALY_DETECTION = ("ECOD", "DeepSVDD", "Iforest", "LODA", "MCD")


def anomaly_detection_models_to_evaluate(models: list):
    # Assert input values
    for model in models:
        assert (
            model in AVAILABLE_MODELS_ANOMALY_DETECTION
        ), "Unrecognized value, model should be one of the following: " + str(
            AVAILABLE_MODELS_ANOMALY_DETECTION
        )
    # Initiate empty list to collect the selected models
    models_to_evaluate = []
    # Loop through imput and add selected models
    for model in models:
        if model == "ECOD":
            models_to_evaluate.append(("ECOD", ECOD(n_jobs=-1)))
        if model == "Iforest":
            models_to_evaluate.append(("Iforest", IForest(n_jobs=-1, random_state=123)))
        if model == "LODA":
            models_to_evaluate.append(("LODA", LODA()))
        if model == "MCD":
            models_to_evaluate.append(("MCD", MCD(random_state=123)))
    return models_to_evaluate
