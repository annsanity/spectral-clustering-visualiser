import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph 
import geopandas as gpd
from algorithms.visualise import visualize_clusters_on_map, visualize_graph2, create_adjacency_matrix

def clustering(num_clusters, similairty_attribute, graph, region):

    fig = None

    df, features, adjacency_matrix = create_adjacency_matrix(similairty_attribute, region)

    # Perform spectral clustering
    k = num_clusters  # Adjust the number of clusters as needed
    labels, optimal_clusters = normalized_spectral_clustering(adjacency_matrix, k)
    df['Cluster'] = df['Cluster'] = labels

    # Read the shapefile
    shapefile_path = r"project_files\backend\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)
    world = world[['NAME', 'geometry']]
    world = world.sort_values(by='NAME')

     # Preprocess country names to match
    df['Country'] = df['Country'].str.strip()
    world['NAME'] = world['NAME'].str.strip()

    #Remove Antarctica
    world = world[world['NAME'] != 'Antarctica']

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

# Normalized spectral clustering according to Shi and Malik 2000
def normalized_spectral_clustering(adjacency_matrix, k):

    W = adjacency_matrix.copy()
    D = np.diag(np.sum(W, axis=1))

    # Calulate the Laplacian matrix L_rw
    L = laplacian(W, normed=False)
    D_inv = np.linalg.inv(D)
    L_rw = np.eye(W.shape[0]) - np.dot(D_inv, W)
    
    # Compute optimal number of clusters
    eigenvalues, _ = eigh(L_rw)
    eigenvalues = np.sort(eigenvalues)
    
    gaps = np.diff(eigenvalues)
    relative_gaps = gaps / (eigenvalues[:-1] + 1e-10)  # Avoid division by zero

    index = np.argmax(relative_gaps)
    optimal_clusters = index + 1

    # Step 3: Solve the generalized eigenvalue problem Lu = λDu
    # We need to solve for the first k eigenvectors
    eigenvalues, eigenvectors = eigh(L_rw, D, subset_by_index=[0, k-1])
    
    # Step 4: Form the matrix U containing the first k generalized eigenvectors as columns
    U = eigenvectors[:, :k]
    
    Y = U 
    
    # Step 5: Cluster the points Y in R^k with the k-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(Y)
    labels = kmeans.labels_

    return labels, optimal_clusters

