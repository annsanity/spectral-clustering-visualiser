# Importing necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS 
# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.algorithms import (
    agglomerative,
    dbscan,
    kmeans,
    normalized_spec_ngjordanweiss,
    normalized_spec_shimalik,
    unnormalized_spec
)
import plotly.io as pio

# Create a Flask instance
app = Flask(__name__)
CORS(app)

# Endpoint for the clustering API
@app.route('/api/clustering', methods=['POST'])
def clustering():
    data = request.get_json()
    algorithm = data['algorithm']
    similairty_attribute = data['attribute']
    num_clusters = int(data['num_clusters'])
    graph = data['visual_type']
    region = data['region']
    fig = None
    graph_json = None
    if(algorithm == 'agglomerative') :
        metrics, fig = agglomerative.clustering(num_clusters,similairty_attribute, graph, region)
    elif(algorithm == 'dbscan') :
        result = dbscan.clustering(num_clusters,similairty_attribute, graph, region)
        if result is None: 
            return jsonify({'error': 'DBSCAN cannot be performed for this configuration.'}), 200
        metrics, fig = result
    elif(algorithm == 'kmeans') :
        metrics, fig = kmeans.clustering(num_clusters,similairty_attribute, graph, region)
    elif(algorithm == 'normalized_spec_ngjordanweiss') :
        metrics, fig = normalized_spec_ngjordanweiss.clustering(num_clusters,similairty_attribute, graph, region)
    elif(algorithm == 'normalized_spec_shimalik') :
        metrics, fig = normalized_spec_shimalik.clustering(num_clusters,similairty_attribute, graph, region) 
    elif(algorithm == 'unnormalized_spec') :
        metrics, fig = unnormalized_spec.clustering(num_clusters,similairty_attribute, graph, region)
    else :
        return jsonify({'error': 'Invalid algorithm name'})
    
    if(fig is not None):
        graph_json = pio.to_json(fig)

    response = jsonify({'metrics': metrics, 'graph_json': graph_json})

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
