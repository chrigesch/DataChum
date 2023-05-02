# Import moduls from local directories
from assets.strings import in_cluster as string

# Import the required libraries
from sklearn.cluster import Birch, AgglomerativeClustering, DBSCAN
from sklearn.mixture import BayesianGaussianMixture

# Import libraries for debugging


AVAILABLE_MODELS_CLUSTER = [
    "AgglomerativeClustering_average",
    "AgglomerativeClustering_complete",
    "AgglomerativeClustering_single",
    "AgglomerativeClustering_ward",
    "BayesianGaussianMixture_diag",
    "BayesianGaussianMixture_full",
    "BayesianGaussianMixture_spherical",
    "BayesianGaussianMixture_tied",
    "BIRCH",
    "DBSCAN",
    "GaussianMixtureModels",
    "KMeans",
    "SpectralClustering",
]

MODELS_WITH_N_CLUSTER = [
    "AgglomerativeClustering_average",
    "AgglomerativeClustering_complete",
    "AgglomerativeClustering_single",
    "AgglomerativeClustering_ward",
    "BIRCH",
    "KMeans",
    "SpectralClustering",
]
MODELS_WITH_N_COMPONENTS = [
    "BayesianGaussianMixture_diag",
    "BayesianGaussianMixture_full",
    "BayesianGaussianMixture_spherical",
    "BayesianGaussianMixture_tied",
    "GaussianMixtureModels",
]


def cluster_models_to_evaluate(models: list):
    # Assert input values
    for model in models:
        assert (
            model in AVAILABLE_MODELS_CLUSTER
        ), string.models.assert_model_message + str(AVAILABLE_MODELS_CLUSTER)
    # Initiate empty list to collect the selected models
    models_to_evaluate = []
    # Loop through imput and add selected models
    for model in models:
        if model == "AgglomerativeClustering_average":
            models_to_evaluate.append(
                (
                    "AgglomerativeClustering_average",
                    AgglomerativeClustering(linkage="average"),
                )
            )
        if model == "AgglomerativeClustering_complete":
            models_to_evaluate.append(
                (
                    "AgglomerativeClustering_complete",
                    AgglomerativeClustering(linkage="complete"),
                )
            )
        if model == "AgglomerativeClustering_single":
            models_to_evaluate.append(
                (
                    "AgglomerativeClustering_single",
                    AgglomerativeClustering(linkage="single"),
                )
            )
        if model == "AgglomerativeClustering_ward":
            models_to_evaluate.append(
                (
                    "AgglomerativeClustering_ward",
                    AgglomerativeClustering(linkage="ward"),
                )
            )
        if model == "BayesianGaussianMixture_diag":
            models_to_evaluate.append(
                (
                    "BayesianGaussianMixture_diag",
                    BayesianGaussianMixture(
                        covariance_type="diag",
                        random_state=123,
                    ),
                )
            )
        if model == "BayesianGaussianMixture_full":
            models_to_evaluate.append(
                (
                    "BayesianGaussianMixture_full",
                    BayesianGaussianMixture(
                        covariance_type="full",
                        random_state=123,
                    ),
                )
            )
        if model == "BayesianGaussianMixture_spherical":
            models_to_evaluate.append(
                (
                    "BayesianGaussianMixture_spherical",
                    BayesianGaussianMixture(
                        covariance_type="spherical",
                        random_state=123,
                    ),
                )
            )
        if model == "BayesianGaussianMixture_tied":
            models_to_evaluate.append(
                (
                    "BayesianGaussianMixture_tied",
                    BayesianGaussianMixture(
                        covariance_type="tied",
                        random_state=123,
                    ),
                )
            )
        if model == "BIRCH":
            models_to_evaluate.append(("BIRCH", Birch()))
        if model == "DBSCAN":
            models_to_evaluate.append(("DBSCAN", DBSCAN()))
    return models_to_evaluate
