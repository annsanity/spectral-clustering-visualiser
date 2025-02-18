import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
import numpy as np
import geopandas as gpd
from project_files.backend.algorithms.visualise import visualize_clusters_on_map, visualize_traditional_clustering, create_adjacency_matrix
from sklearn.neighbors import kneighbors_graph

def clustering(num_clusters, similairty_attribute, graph, region):

    fig = None

    df, features, adjacency_matrix = create_adjacency_matrix(similairty_attribute, region)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['Cluster'] = kmeans.fit_predict(features)

    # Read the shapefile
    shapefile_path = r"project_files\backend\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)
    world = world[['NAME', 'geometry']]
    world = world.sort_values(by='NAME')

     # Preprocess country names to match
    df['Country'] = df['Country'].str.strip()
    world['NAME'] = world['NAME'].str.strip()

    
    # Merge clustering results with geographical data
    merged = world.merge(df, left_on='NAME', right_on='Country', how='left')

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
