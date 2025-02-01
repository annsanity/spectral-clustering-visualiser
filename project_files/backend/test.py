from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import pandas as pd

# Initialize the WebDriver (assuming you have ChromeDriver installed)
driver = webdriver.Chrome()

# Open your Flask app's URL
driver.get("http://127.0.0.1:3000/frontend/index.html")  # Replace with your actual URL

# Wait for the page to load
time.sleep(3)

# Interact with the controls
def set_options(algorithm, attribute, region, visual_type, num_clusters):
    # Select the clustering algorithm
    Select(driver.find_element(By.ID, "algorithm")).select_by_value(algorithm)
    
    # Select the attribute
    Select(driver.find_element(By.ID, "attribute")).select_by_value(attribute)
    
    # Select the region
    Select(driver.find_element(By.ID, "region")).select_by_value(region)
    
    # Select the visual type
    Select(driver.find_element(By.ID, "visual_type")).select_by_value(visual_type)
    
    # Set the number of clusters
    num_clusters_slider = driver.find_element(By.ID, "num_clusters")
    driver.execute_script(f"arguments[0].value = {num_clusters};", num_clusters_slider)
    
    # Trigger the oninput event to update the display
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'))", num_clusters_slider)

# Function to retrieve the metrics values
def get_metrics():
    silhouette_score = driver.find_element(By.ID, "silhouette-score").text
    davies_bouldin = driver.find_element(By.ID, "davies-bouldin").text
    calinski_harabasz = driver.find_element(By.ID, "calinski-harabasz").text
    optimal_clusters = driver.find_element(By.ID, "optimal-clusters").text
    
    return silhouette_score, davies_bouldin, calinski_harabasz, optimal_clusters

# Function to run the clustering
def run_clustering():
    run_button = driver.find_element(By.XPATH, "//button[text()='Run Clustering']")
    run_button.click()
    # Wait for the clustering to complete (adjust the time if needed)
    time.sleep(20)

# Example usage
algorithms = ['dbscan','unnormalized_spec', 'normalized_spec_shimalik', 'normalized_spec_ngjordanweiss', 'kmeans', 'agglomerative']
attributes = ['all', 'remittance_inflow', 'priv_capital_flow', 'dev_assistance', 'foreign_inflow', 'export_import', 'hdi_rank']

regions = ["World","Asia","Africa","Europe","North America","South America","Oceania"]
visual_types = ["graph_2"]
num_clusters_list = [4]

# Initialize an empty list to store results
results = []

for algorithm in algorithms:
    for attribute in attributes:
        for region in regions:
            for visual_type in visual_types:
                if(region == "Oceania"):
                    num_clusters_list = [4]
                else : 
                    num_clusters_list = [4]
                for num_clusters in num_clusters_list:
                    set_options(algorithm, attribute, region, visual_type, num_clusters)
                    run_clustering()
                    metrics = get_metrics()
                    # Append the result
                    results.append([algorithm, attribute, region, visual_type, num_clusters] + list(metrics))

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(results, columns=["Algorithm", "Attribute", "Region", "Visual Type", "Num Clusters", 
                                    "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Optimal Clusters"])
df.to_csv("clustering_metrics_pre_tuning_final.csv", index=False)

# Close the browser
driver.quit()
