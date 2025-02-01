import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from unnormalized_spec import unnormalized_spectral_clustering
from normalized_spec_ngjordanweiss import normalized_spectral_clustering_ng_jordan_weiss
from normalized_spec_shimalik import normalized_spectral_clustering

# Load the data once
df = pd.read_csv(r'Cleaned_Data_new.csv', encoding='iso-8859-1')
features = df.iloc[:, 1:7]  # Assuming all features are from column 1 to 7

# Initialize results container
results = []

# Function to calculate clustering metrics and append to results
def evaluate_clustering(labels, graph_type, param, n_clusters, algo):
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    return (graph_type, param, n_clusters, silhouette, davies_bouldin, calinski_harabasz, algo)

# 1. ε-neighborhood graph with precomputed distances
def epsilon_neighborhood_graph(precomputed_distances, epsilon):
    adjacency_matrix = np.where(precomputed_distances < epsilon, 1, 0)
    return adjacency_matrix

# 2. k-nearest neighbor graph
def knn_graph(features, n_neighbors, metric='minkowski'):
    knn_graph = kneighbors_graph(features, n_neighbors=n_neighbors, mode='connectivity', include_self=False, metric=metric)
    return knn_graph.toarray()

# 3. Fully connected graph with Gaussian similarity
def fully_connected_graph(features, sigma, kernel='rbf', gamma=None):
    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / (2 * sigma ** 2)
        adjacency_matrix = rbf_kernel(features, gamma=gamma)
    elif kernel == 'cosine':
        adjacency_matrix = cosine_similarity(features)
    return adjacency_matrix

# Function to run a single experiment with different algorithms
def run_experiment(graph_type, param, adjacency_matrix, n_clusters=4, algo='unnormalized'):
    if algo == 'unnormalized':
        labels, optimal_clusters = unnormalized_spectral_clustering(adjacency_matrix, n_clusters)
    elif algo == 'normalized_ngjordanweiss':
        labels, optimal_clusters = normalized_spectral_clustering_ng_jordan_weiss(adjacency_matrix, n_clusters)
    elif algo == 'normalized_shimalik':
        labels, optimal_clusters = normalized_spectral_clustering(adjacency_matrix, n_clusters)
    else:
        raise ValueError(f"Unknown algorithm type: {algo}")
    
    # Add the algorithm name to the result tuple
    result = evaluate_clustering(labels, graph_type, param, n_clusters, algo)
    return result  # Return only the evaluation metrics

# Precompute pairwise distances for ε-neighborhood graphs
distance_metrics = ['euclidean', 'cityblock', 'cosine']
precomputed_distances = {metric: squareform(pdist(features, metric=metric)) for metric in distance_metrics}

# Prepare experiments for ε-neighborhood, k-nearest neighbor, and fully connected graphs
experiments = []
algorithms = ['unnormalized', 'normalized_ngjordanweiss', 'normalized_shimalik']

# 1. ε-neighborhood graph experiments
for algo in algorithms:
    for epsilon in np.arange(0.1, 3.5, 0.1):
        for metric in distance_metrics:
            adjacency_matrix = epsilon_neighborhood_graph(precomputed_distances[metric], epsilon)
            experiments.append((f'ε-neighborhood ({metric})', epsilon, adjacency_matrix, 4, algo))

# 2. k-nearest neighbor graph experiments
for algo in algorithms:
    for k in range(5, 31, 1):
        for metric in ['minkowski', 'euclidean', 'manhattan']:
            adjacency_matrix = knn_graph(features, k, metric=metric)
            experiments.append((f'k-nearest neighbors ({metric})', k, adjacency_matrix, 4, algo))

# 3. Fully connected graph experiments
for algo in algorithms:
    for sigma in np.arange(0.2, 4.0, 0.2):
        for kernel in ['rbf', 'cosine']:
            for gamma in [None, 0.05, 0.1, 0.5, 1.0]:
                adjacency_matrix = fully_connected_graph(features, sigma, kernel=kernel, gamma=gamma)
                experiments.append((f'Fully connected ({kernel})', sigma, adjacency_matrix, 4, algo))

# Run experiments sequentially
results = []
for experiment in experiments:
    graph_type, param, adjacency_matrix, n_clusters, algo = experiment
    result = run_experiment(graph_type, param, adjacency_matrix, 4, algo)  # Fixed 4 clusters
    results.append(result)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=['Graph Type', 'Parameter', 'Number of Clusters', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'Algorithm'])

# Loop through each algorithm and find the best result for each
for algo in algorithms:
    algo_results_df = results_df[results_df['Algorithm'] == algo]
    best_algo_result = algo_results_df.loc[algo_results_df['Silhouette Score'].idxmax()]
    
    print(f"\nBest result for {algo}:\n{best_algo_result}")
