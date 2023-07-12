# Import the required Libraries
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.pca import PCA


AVAILABLE_MODELS_ANOMALY_DETECTION = (
    "CBLOF",
    "COPOD",
    "ECOD",
    "DeepSVDD",
    "Iforest",
    "HBOS",
    "KNN",
    "LODA",
    "LOF",
    "MCD",
    "PCA",
)


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
        if model == "CBLOF":
            models_to_evaluate.append(("CBLOF", CBLOF(n_jobs=-1, random_state=123)))
        if model == "COPOD":
            models_to_evaluate.append(("COPOD", COPOD(n_jobs=-1)))
        if model == "ECOD":
            models_to_evaluate.append(("ECOD", ECOD(n_jobs=-1)))
        if model == "HBOS":
            models_to_evaluate.append(("HBOS", HBOS()))
        if model == "Iforest":
            models_to_evaluate.append(("Iforest", IForest(n_jobs=-1, random_state=123)))
        if model == "KNN":
            models_to_evaluate.append(("KNN", KNN(n_jobs=-1)))
        if model == "LOF":
            models_to_evaluate.append(("LOF", LOF(n_jobs=-1)))
        if model == "LODA":
            models_to_evaluate.append(("LODA", LODA()))
        if model == "MCD":
            models_to_evaluate.append(("MCD", MCD(random_state=123)))
        if model == "PCA":
            models_to_evaluate.append(("PCA", PCA(random_state=123)))
    return models_to_evaluate
