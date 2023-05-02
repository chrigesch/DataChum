# ----------------------------------------------------------------------------------------------------
# (a) Models:  n(1) 'BayesianGaussianMixture'-BayesianGaussianMixture(n_components=2,covariance_type='full',random_state=123) ['full','tied','diag','spherical'] # noqa: E501
#              n(1) 'BIRCH'-Birch(n_clusters=3,threshold=0.5,branching_factor=50)
#               (2) 'DBSCAN'-Density-Based Spatial Clustering, DBSCAN(eps=.5,min_samples=15)
#              n(3) 'GaussianMixtureModels', GaussianMixture(n_components=2,covariance_type='full',random_state=123) # noqa: E501
#              n(4) 'AgglomerativeClustering', AgglomerativeClustering(n_clusters=2,linkage='ward) | {'ward','complete','average','single'} # noqa: E501
#              n(5) 'KMeans'-K-Means Clustering, KMeans(n_clusters=8,random_state=123,n_init='auto')
#              n(6) 'SpectralClustering'SpectralClustering(n_clusters=8,affinity='rbf'',random_state=123,n_jobs=-1,verbose=False) # noqa: E501
# (b)   Evaluation metrics:
#       Silhouette, Calinski-Harabasz, Davies-Bouldin
#       BIC
#       https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
#       Bootstrapping method to evaluate evaluation metrics
#       https://stats.stackexchange.com/questions/11691/how-to-tell-if-data-is-clustered-enough-for-clustering-algorithms-to-produce-m/11702#11702
#       https://stats.stackexchange.com/questions/71184/cross-validation-or-bootstrapping-to-evaluate-classification-performance # noqa: E501
#       How many bootrapping resamples?
#       Check every 50 replicates (or 20?) coefficient of variation with stopping criterion (add a constant?): less than 1% # noqa: E501
#       https://uol.de/f/5/inst/biologie/ag/systematik/download/Publications/bootstopping2.pdf
#
#       CLEST - (combine clustering with classification)
#       https://link.springer.com/article/10.1186/gb-2002-3-7-research0036
# (c) Plots: 'cluster'-Cluster PCA Plot (2d),'tsne'-Cluster t-SNE (3d),'elbow'-Elbow Plot,'silhouette'-Silhouette Plot
#             'distance'-Distance Plot,'distribution'-Distribution Plot
#             'Mean Based Line Plot'
# ----------------------------------------------------------------------------------------------------
# (1) Brute force vs. Create a 'tuneable' sklearnmodel | Tune hyperparameters & preprocessing
# ----------------------------------------------------------------------------------------------------
# (1) https://towardsdatascience.com/are-you-still-using-the-elbow-method-5d271b3063bd
# (2) https://machinelearningmastery.com/probabilistic-model-selection-measures/
# (3) https://medium.com/analytics-vidhya/probabilistic-model-selection-with-aic-bic-in-python-f8471d6add32
# (4) https://towardsdatascience.com/how-to-improve-clustering-accuracy-with-bayesian-gaussian-mixture-models-2ef8bb2d603f # noqa: E501
