import folium
import pandas as pd
from folium import LayerControl
import networkx as nx
import os
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


def create_adjacency_matrix(similarity_attribute, region, epsilon = 0.5):

    # Load the cleaned data
    df = pd.read_csv(r'Cleaned_Data_new.csv', encoding='iso-8859-1')
    
    # Filter the data based on the region (Assuming filter_countries_by_region is correctly defined elsewhere)
    df = filter_countries_by_region(df, region)

    # Assign nodes as index values of the countries
    nodes = df.index.values
    n = len(nodes)

    # Extract relevant features for clustering based on the selected similarity attribute
      # Extract relevant features for clustering based on the selected similarity attribute
    if similarity_attribute == 'all':
        features = df.iloc[:, 1:7]  # Assuming all relevant features are between columns 1 and 7
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
        raise ValueError("Invalid similarity attribute")


    # # Calculate the similarity matrix
    # similarity_matrix = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         if i != j:
    #             similarities = []
    #             for k in range(features.shape[1]):
    #                 similarities.append(np.exp(-np.linalg.norm(features.iloc[i, k] - features.iloc[j, k])**2 / (2 * 0.05**2)))
    #             similarity_matrix[i, j] = np.mean(similarities)

    # n_neighbors = max(5, int(np.sqrt(len(df))))
    # knn_graph = kneighbors_graph(similarity_matrix, n_neighbors=n_neighbors, mode='distance', include_self=False)
    # adjacency_matrix = knn_graph.toarray()

    # Calculate the Euclidean distance matrix
    distance_matrix = pairwise_distances(features, metric='euclidean')

    # Convert the distance matrix to an adjacency matrix using a threshold (e.g., ε-neighborhood)
    adjacency_matrix = np.where(distance_matrix <= epsilon, 1, 0)


    return df, features, adjacency_matrix

def filter_countries_by_region(df, region):

    if(region == 'Europe' or region == 'Asia'):
        df = df[(df['Region'] == region) | (df['Region'] == 'Asia/Europe')]

    elif region == 'World':
        pass
    else :
        df = df[df['Region'] == region]
    return df
    

def visualize_clusters_on_map(merged_gdf, output_html_path):
    # Define the initial center of the map
    world_map = folium.Map(location=[20, 0], zoom_start=2)

    # Get unique clusters
    unique_labels = merged_gdf['Cluster'].dropna().unique()

    # Assign colors using the assign_colors function
    cluster_colors = assign_colors(unique_labels)

    # Add all countries to the map
    for idx, row in merged_gdf.iterrows():
        # Safely access properties
        geometry = row['geometry'].__geo_interface__
        properties = {
            'name': row['NAME'],
            'Cluster': row['Cluster'] if not pd.isna(row['Cluster']) else 'No Cluster'
        }

        # Determine the color based on whether the country has a cluster or not
        if pd.isna(row['Cluster']):
            color = 'lightgrey'  # Neutral color for countries with no cluster
        else:
            color = cluster_colors.get(int(row['Cluster']))

        # Add country to the map
        folium.GeoJson(
            data={
                'type': 'Feature',
                'geometry': geometry,
                'properties': properties
            },
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7 if color != 'lightgrey' else 0.3
            },
            tooltip=folium.GeoJsonTooltip(fields=['name', 'Cluster'], aliases=['Name', 'Cluster'])
        ).add_to(world_map)

    # Add a layer control to the map
    if not any(isinstance(layer, LayerControl) for layer in world_map._children.values()):
        LayerControl().add_to(world_map)

    if os.path.exists(output_html_path):
        os.remove(output_html_path)

    # Save the map as an HTML file
    world_map.save(output_html_path)

    return world_map

def  visualize_graph2(merged, adjacency_matrix, labels):
    G = nx.Graph()

    # Add nodes with their cluster labels as attributes
    for i, country in enumerate(merged['Country']):
        G.add_node(country, cluster=labels[i])

    # Add edges based on the adjacency matrix
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix)):
            if adjacency_matrix[i][j] > 0:
                G.add_edge(merged['Country'][i], merged['Country'][j], weight=adjacency_matrix[i][j])

    # Compute positions using layout
    pos = nx.spring_layout(G, seed=42)

    # Define distinct colors for clusters (similar to your image)
    unique_labels = list(set(labels))
    cluster_colors = assign_colors(unique_labels)

    # Prepare data for Plotly graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []


    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(cluster_colors[labels[merged['Country'].tolist().index(node)]])
        node_size.append(15)  # Increase node size for visibility


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='black')),
        textfont=dict(
            size=10,
            color='black'
        )
    )
    

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    
    # Return the figure and node positions
    return fig


def visualize_traditional_clustering(merged, adjacency_matrix, labels):
    G = nx.Graph()

    # Add nodes with their cluster labels as attributes
    for i, country in enumerate(merged['Country']):
        G.add_node(country, cluster=labels[i])

    # Add edges based on the adjacency matrix
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix)):
            if adjacency_matrix[i][j] > 0:
                G.add_edge(merged['Country'][i], merged['Country'][j], weight=adjacency_matrix[i][j])

    pos = nx.spring_layout(G, seed=42)

    # Define distinct colors for clusters (similar to your image)
    unique_labels = list(set(labels))
    cluster_colors = assign_colors(unique_labels)

    # Prepare data for Plotly graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(0,0,0,0)'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []


    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(cluster_colors[labels[merged['Country'].tolist().index(node)]])
        node_size.append(15)  # Increase node size for visibility


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='black')),
        textfont=dict(
            size=10,
            color='black'
        )
    )
    

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    
    # Return the figure and node positions
    return fig


def assign_colors(labels):
    # Fixed sequence of 40 colors
    fixed_colors = [
        'green', 'blue', 'cyan', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'red',
        'magenta', 'yellow', 'teal', 'coral', 'lime', 'navy', 'lavender', 'maroon', 'gold', 'turquoise',
        'indigo', 'salmon', 'khaki', 'orchid', 'plum', 'sienna', 'peru', 'peachpuff', 'slateblue', 'seagreen',
        'crimson', 'darkorange', 'darkgreen', 'darkblue', 'darkred', 'darkviolet', 'darkturquoise', 'darkkhaki', 'darksalmon', 'darkcyan'
    ]

    # Create a dictionary to map each label to a color
    cluster_colors = {}
    for i, label in enumerate(labels):
        if label not in cluster_colors:
            cluster_colors[label] = fixed_colors[i % len(fixed_colors)]

    return cluster_colors