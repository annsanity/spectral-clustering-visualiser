document.addEventListener('DOMContentLoaded', function() {
    const algorithm = document.getElementById('algorithm');
    const num_clusters = document.getElementById('num_clusters');
    const attribute = document.getElementById('attribute');
    const visualizer = document.getElementById('visualizer');
    const silhouette = document.getElementById('silhouette-score');
    const davies_bouldin = document.getElementById('davies-bouldin');
    const calinski_harabasz = document.getElementById('calinski-harabasz');
    const numClustersDisplay = document.getElementById('num-clusters-display');
    const visualType = document.getElementById('visual_type');
    const optimal_clusters = document.getElementById('optimal-clusters');
    const region = document.getElementById('region');

    function updateClusterCountDisplay() {
        numClustersDisplay.textContent = 'Number of Clusters: ' + num_clusters.value;
    }

    function loading(){
        visualizer.innerHTML = "Performing Clustering...";
    }

    function runClustering() {
        const algorithmValue = algorithm.value;
        const numClustersValue = num_clusters.value;
        const attributeValue = attribute.value;
        const visualTypeValue = visualType.value;

        console.log('Number of clusters and algorithm', numClustersValue , algorithmValue);

        fetch('/api/clustering', {
            method: 'POST',
            body: JSON.stringify({
                algorithm: algorithmValue,
                num_clusters: numClustersValue,
                attribute: attributeValue,
                visual_type : visualTypeValue,
                region : region.value
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {

            if (data.error) {
                visualizer.innerHTML = data.error;
                optimal_clusters.textContent = "N/A";
                silhouette.textContent = "N/A";
                davies_bouldin.textContent = "N/A";
                calinski_harabasz.textContent = "N/A";
                return; // Exit since there's an error
            }
            
            const metrics = data.metrics;
            const graph_json = JSON.parse(data.graph_json);
            
            if (algorithmValue === 'kmeans' || algorithmValue === 'dbscan' || algorithmValue === 'agglomerative') {
                optimal_clusters.textContent = "N/A";
            } else {
                optimal_clusters.textContent = metrics ? metrics[3] : "N/A";
            }
            
            if (visualTypeValue === 'graph_2') {
                visualizer.innerHTML = '';
                Plotly.newPlot('visualizer', graph_json.data, graph_json.layout);
            } else {
                updateVisualizerForTraditionalAlgos();
            }

            silhouette.textContent = metrics ? metrics[0] : "N/A";
            davies_bouldin.textContent = metrics ? metrics[1] : "N/A";
            calinski_harabasz.textContent = metrics ? metrics[2] : "N/A";
        })
        .catch(error => {
            console.error('Error during clustering:', error);
        });
    }

    function updateVisualizerForTraditionalAlgos() {
        visualizer.innerHTML = '';
        fetch('static/clustering_map.html')
        .then(response => response.text())
        .then(html => {
            visualizer.innerHTML = html;
            const scripts = visualizer.querySelectorAll('script');
            scripts.forEach(script => {
                const newScript = document.createElement('script');
                Array.from(script.attributes).forEach(attr => newScript.setAttribute(attr.name, attr.value));
                newScript.textContent = `(function() { ${script.textContent} })();`;
                document.head.appendChild(newScript).parentNode.removeChild(newScript);
            }); 
        })
        .catch(error => {
            console.error('Error fetching the map:', error);
        });
    }

    if (!document.getElementById('tooltip')) {
        const tooltip = document.createElement('div');
        tooltip.id = 'tooltip';
        tooltip.style.position = 'absolute';
        tooltip.style.opacity = 0;
        tooltip.style.backgroundColor = 'white';
        tooltip.style.border = '1px solid black';
        tooltip.style.padding = '5px';
        document.body.appendChild(tooltip);
    }

    function updateMaxClusters(){
        const maxClustersByRegion = {
            "Asia": 43,
            "Africa": 46,
            "North America": 15,
            "South America": 11,
            "Europe": 40,
            "Oceania": 5,
            "World": 100
        };

        const regionValue = region.value;
        const maxClusters = maxClustersByRegion[regionValue];

        // Update the max attribute of the range input
        num_clusters.max = maxClusters;

        // Ensure the current value is within the new max
        if (num_clusters.value > maxClusters) {
            num_clusters.value = maxClusters;
        }
    
        updateClusterCountDisplay();
    }

    // Event listener to update max clusters when the region changes
    region.addEventListener('change', updateMaxClusters);

    // Initial setup
    updateMaxClusters();

    // Make these functions accessible globally if needed
    window.runClustering = runClustering;
    window.updateClusterCountDisplay = updateClusterCountDisplay;
    window.loading = loading;
    window.updateMaxClusters = updateMaxClusters;
});
