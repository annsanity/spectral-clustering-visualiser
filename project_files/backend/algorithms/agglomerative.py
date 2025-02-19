import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import geopandas as gpd
from sklearn.neighbors import kneighbors_graph
import numpy as np
from project_files.backend.algorithms.visualise import visualize_clusters_on_map, visualize_traditional_clustering, create_adjacency_matrix

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
shp_path = os.path.join(base_dir, "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")
print("Shapefile path:", shp_path)

def clustering(num_clusters, similairty_attribute, graph, region):

    fig = None
    df, features, adjacency_matrix = create_adjacency_matrix(similairty_attribute, region)

    # Apply Agglomerative clustering algorithm
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
    df['Cluster'] = agglomerative.fit_predict(features)

    # Read the shapefile
    world = gpd.read_file(shp_path)
    world = world[['NAME', 'geometry']]
    world = world.sort_values(by='NAME')

    # Preprocess country names to match
    df['Country'] = df['Country'].str.strip()
    world['NAME'] = world['NAME'].str.strip()


    # Merge clustering results with geographical data
    merged = world.merge(df, left_on='NAME', right_on='Country',how='left')

    # Calculate Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index
    metrics = []
    silhouette_score_value = silhouette_score(features, df['Cluster'])
    davies_bouldin_score_value = davies_bouldin_score(features, df['Cluster'])
    calinski_harabasz_score_value = calinski_harabasz_score(features, df['Cluster'])
    metrics.append(float(silhouette_score_value))
    metrics.append(float(davies_bouldin_score_value))
    metrics.append(float(calinski_harabasz_score_value))

    labels = df['Cluster'].values

    # To include the countries whose data is available in the dataset
    merged_2 = world.merge(df, left_on='NAME', right_on='Country',how='right')

    if(graph == "world_map") :
        graph = visualize_clusters_on_map(merged,r"project_files\frontend\static\clustering_map.html")
    elif(graph == "graph_2") :
        fig = visualize_traditional_clustering(merged_2,adjacency_matrix, labels)
    
    return metrics, fig




