import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
import geopandas as gpd
import numpy as np
from project_files.backend.algorithms.visualise  import visualize_clusters_on_map, visualize_traditional_clustering, filter_countries_by_region
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

def clustering(num_clusters,similairty_attribute, graph, region):
    
    fig = None
    
    df = pd.read_csv(r'Cleaned_Data_new.csv', encoding='iso-8859-1')

    # Filter the data based on the region
    df = filter_countries_by_region(df, region)

    # Assign nodes as index values of the countries
    nodes = df.index.values
    n = nodes.shape[0]

    # Initialize eps and min_samples
    eps, min_samples = None, None

    # Set features and DBSCAN parameters based on region and similarity_attribute

    if similairty_attribute == 'all':
        features = df.iloc[:, 1:7]
        params = {
            'Asia': (0.4605263157894737, 2),
            'Africa': None,  # No valid params
            'North America': (0.15263157894736842, 2),
            'South America': None,  # No valid params
            'Europe': (0.15263157894736842, 2),
            'Oceania': (0.4605263157894737, 2),
            'World': (0.15263157894736842, 3)
        }
    elif similairty_attribute == 'hdi_rank':
        features = df.iloc[:, -7].to_frame()
        params = {
            'Asia': (0.05, 6),
            'Africa': (0.05, 2),
            'North America': (0.05, 2),
            'South America': (0.05, 2),
            'Europe': (0.05, 13),
            'Oceania': (0.15263157894736842, 2),
            'World': (0.25526315789473686, 97)
        }
    elif similairty_attribute == 'export_import':
        features = df.iloc[:, -6].to_frame()
        params = {
            'Asia': (0.15263157894736842, 2),
            'Africa': (0.05, 9),
            'North America': (0.05, 2),
            'South America': None,  # No valid params
            'Europe': (0.05, 2),
            'Oceania': (0.05, 2),
            'World': (0.05, 2)
        }
    elif similairty_attribute == 'foreign_inflow':
        features = df.iloc[:, -5].to_frame()
        params = {
            'Asia': None,  # No valid params
            'Africa': None,  # No valid params
            'North America': None,  # No valid params
            'South America': None,  # No valid params
            'Europe': None,  # No valid params
            'Oceania': None,  # No valid params
            'World': None  # No valid params
        }
    elif similairty_attribute == 'dev_assistance':
        features = df.iloc[:, -4].to_frame()
        params = {
            'Asia': None,  # No valid params
            'Africa': (0.15263157894736842, 2),
            'North America': None,  # No valid params
            'South America': None,  # No valid params
            'Europe': None,  # No valid params
            'Oceania': (0.05, 2),
            'World': (0.05, 4)
        }
    elif similairty_attribute == 'priv_capital_flow':
        features = df.iloc[:, -3].to_frame()
        params = {
            'Asia': None,  # No valid params
            'Africa': (0.05, 2),
            'North America': None,  # No valid params
            'South America': None,  # No valid params
            'Europe': None,  # No valid params
            'Oceania': None,  # No valid params
            'World': None  # No valid params
        }
    elif similairty_attribute == 'remittance_inflow':
        features = df.iloc[:, -2].to_frame()
        params = {
            'Asia': (0.15263157894736842, 2),
            'Africa': (0.05, 2),
            'North America': (0.05, 2),
            'South America': None,  # No valid params
            'Europe': (0.05, 2),
            'Oceania': (0.05, 2),
            'World': (0.15263157894736842, 2)
        }

    
    if params[region] == None:
        return None
    else:
        print(params[region])
        eps, min_samples = params[region]
    
    # Calculate the similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarities = []
                for k in range(features.shape[1]):
                    similarities.append(np.exp(-np.linalg.norm(features.iloc[i, k] - features.iloc[j, k])**2 / (2 * 0.05**2)))
                similarity_matrix[i, j] = np.mean(similarities)

    n_neighbors = max(5, int(np.sqrt(len(df))))
    knn_graph = kneighbors_graph(similarity_matrix, n_neighbors=n_neighbors, mode='distance', include_self=False)
    adjacency_matrix = knn_graph.toarray()
    
    # # Calculate the Euclidean distance matrix
    # distance_matrix = pairwise_distances(features, metric='euclidean')

    # # Convert the distance matrix to an adjacency matrix using a threshold (e.g., ε-neighborhood)
    # adjacency_matrix = np.where(distance_matrix <= 0.5, 1, 0)


     # Apply DBSCAN clustering algorithm
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(features)

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
    merged_2 = world.merge(df, left_on='NAME', right_on='Country', how='right')

    # Visualize the clusters
    if graph == "world_map":
        graph = visualize_clusters_on_map(merged, r"project_files\frontend\static\clustering_map.html")
    elif graph == "graph_2":
        fig = visualize_traditional_clustering(merged_2, adjacency_matrix, labels)

    return metrics, fig
