import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from visualise import filter_countries_by_region

# Define max clusters by region
maxClustersByRegion = {
    "Asia": 43,
    "Africa": 46,
    "North America": 15,
    "South America": 11,
    "Europe": 40,
    "Oceania": 5,
    "World": 100
}

def find_best_dbscan_params_with_multiple_criteria(features, eps_range, min_samples_range):
    best_params_silhouette = None
    best_params_dbi = None
    best_params_chi = None
    
    best_score_silhouette = -1
    best_score_dbi = float('inf')  # For DBI, lower is better
    best_score_chi = -1

    best_labels_silhouette = None
    best_labels_dbi = None
    best_labels_chi = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)
            unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if unique_clusters > 1:  # Valid number of clusters
                # Calculate silhouette score
                silhouette = silhouette_score(features, labels)
                if silhouette > best_score_silhouette:
                    best_score_silhouette = silhouette
                    best_params_silhouette = (eps, min_samples)
                    best_labels_silhouette = labels
                
                # Calculate Davies-Bouldin Index
                dbi = davies_bouldin_score(features, labels)
                if dbi < best_score_dbi:
                    best_score_dbi = dbi
                    best_params_dbi = (eps, min_samples)
                    best_labels_dbi = labels
                
                # Calculate Calinski-Harabasz Index
                chi = calinski_harabasz_score(features, labels)
                if chi > best_score_chi:
                    best_score_chi = chi
                    best_params_chi = (eps, min_samples)
                    best_labels_chi = labels

    return {
        "silhouette": {"params": best_params_silhouette, "score": best_score_silhouette, "labels": best_labels_silhouette},
        "davies_bouldin": {"params": best_params_dbi, "score": best_score_dbi, "labels": best_labels_dbi},
        "calinski_harabasz": {"params": best_params_chi, "score": best_score_chi, "labels": best_labels_chi}
    }

def get_min_samples_range(region):
    max_clusters = maxClustersByRegion.get(region, 20)  # Default to 20 if region not found
    # Set min_samples range based on the region's max clusters
    return range(2, max_clusters + 1)

def select_features(df, similarity_attribute):
    # Extract relevant features for clustering based on similarity attribute
    if similarity_attribute == 'all':
        features = df.iloc[:, 1:7]
    elif similarity_attribute == 'hdi_rank':
        features = df.iloc[:, -7].to_frame()
    elif similarity_attribute == 'export_import':
        features = df.iloc[:, -6].to_frame()
    elif similarity_attribute == 'foreign_inflow':
        features = df.iloc[:, -5].to_frame()
    elif similarity_attribute == 'dev_assistance':
        features = df.iloc[:, -4].to_frame()
    elif similarity_attribute == 'priv_capital_flow':
        features = df.iloc[:, -3].to_frame()
    elif similarity_attribute == 'remittance_inflow':
        features = df.iloc[:, -2].to_frame()
    else:
        raise ValueError("Unknown similarity attribute provided.")
    
    return features


# Load the cleaned data
df_original = pd.read_csv('ClusteringVisualiser\Cleaned_Data_new.csv')

# Define parameter ranges for DBSCAN
eps_range = np.linspace(0.05, 2.0, 20)

# List of regions to process
regions = ['Asia', 'Africa', 'North America', 'South America', 'Europe', 'Oceania', 'World']

# Define similarity attributes to be tested
similarity_attributes = ['all', 'hdi_rank', 'export_import', 'foreign_inflow', 'dev_assistance', 'priv_capital_flow', 'remittance_inflow']

for region in regions:
    print(f"Processing region: {region}")
    
    # Filter data for the current region
    df = filter_countries_by_region(df_original, region)
    
    # Get the appropriate min_samples_range for the region
    min_samples_range = get_min_samples_range(region)
    
    for similarity_attribute in similarity_attributes:
        # Select features based on the current similarity attribute
        features = select_features(df, similarity_attribute)
        
        
        # Find the best DBSCAN parameters for the selected features using multiple criteria
        best_scores = find_best_dbscan_params_with_multiple_criteria(features, eps_range, min_samples_range)
        
        print(f"Best params for '{region}' - '{similarity_attribute}':")
        print(f"  Silhouette: {best_scores['silhouette']['params']}, Score: {best_scores['silhouette']['score']}")
        print(f"  Davies-Bouldin: {best_scores['davies_bouldin']['params']}, Score: {best_scores['davies_bouldin']['score']}")
        print(f"  Calinski-Harabasz: {best_scores['calinski_harabasz']['params']}, Score: {best_scores['calinski_harabasz']['score']}")
