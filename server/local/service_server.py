from flask import Flask, request, jsonify
import os
from pathlib import Path

app = Flask(__name__)
model_registry = {}

@app.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    model = model_registry.get(model_id)
    if model:
        return jsonify({
            "path": str(model['path']),
            "shards": model['shards']
        })
    return jsonify({"error": "Model not found"}), 404

@app.route('/models/<model_id>', methods=['POST'])
def register_model(model_id):
    data = request.json
    model_registry[model_id] = {
        "path": data['path'],
        "shards": data.get('shards', [])
    }
    return jsonify({"status": "registered"}), 201

def run_registry_server(port=8000):
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    run_registry_server()