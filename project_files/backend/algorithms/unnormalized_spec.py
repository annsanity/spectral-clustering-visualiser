import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph 
import geopandas as gpd
from project_files.backend.algorithms.visualise import visualize_clusters_on_map, visualize_graph2, create_adjacency_matrix


def clustering(num_clusters, similairty_attribute, graph, region):
     
    fig = None
    
    df, features, adjacency_matrix = create_adjacency_matrix(similairty_attribute, region, epsilon=0.6)

    # Perform spectral clustering
    k = num_clusters  # Adjust the number of clusters as needed
    labels, optimal_clusters = unnormalized_spectral_clustering(adjacency_matrix, k)
    df['Cluster'] = labels

    # Read the shapefile
    shapefile_path = r"project_files\backend\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)
    world = world[['NAME', 'geometry']]
    world = world.sort_values(by='NAME')

     # Preprocess country names to match
    df['Country'] = df['Country'].str.strip()
    world['NAME'] = world['NAME'].str.strip()


    # Merge clustering results with geographical data
    merged = world.merge(df, left_on='NAME', right_on='Country',how='left')

    # To include the countries whose data is available in the dataset
    merged_2 = world.merge(df, left_on='NAME', right_on='Country',how='right')


    # Calculate Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index
    metrics = []
    silhouette_score_value = silhouette_score(features, df['Cluster'])
    davies_bouldin_score_value = davies_bouldin_score(features, df['Cluster'])
    calinski_harabasz_score_value = calinski_harabasz_score(features, df['Cluster'])
    metrics.append(float(silhouette_score_value))
    metrics.append(float(davies_bouldin_score_value))
    metrics.append(float(calinski_harabasz_score_value))
    metrics.append(int(optimal_clusters))


    if(graph == "world_map") :
        graph = visualize_clusters_on_map(merged, r"project_files\frontend\static\clustering_map.html")
    elif(graph == "graph_2") :
        fig = visualize_graph2(merged_2, adjacency_matrix, labels)


    return metrics, fig

# Unnormalized spectral clustering function
def unnormalized_spectral_clustering(adjacency_matrix, k):

    W = adjacency_matrix.copy()
    L = laplacian(W, normed=False)

    # Compute optimal number of clusters
    eigenvalues, _ = eigh(L)
    eigenvalues = np.sort(eigenvalues)
    
    gaps = np.diff(eigenvalues)
    relative_gaps = gaps / (eigenvalues[:-1] + 1e-10)  # Avoid division by zero

    index = np.argmax(relative_gaps)
    optimal_clusters = index + 1
    
    # Calculating the first k eigenvectors of L
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, k-1])
    
    # Form the matrix Y where each row corresponds to a row of U
    Y = eigenvectors
    
    # Cluster the points Y in R^k with the k-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(Y)
    labels = kmeans.labels_
    
    return labels, optimal_clusters
