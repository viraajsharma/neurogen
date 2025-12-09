from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Get the parent directory (neurogen root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PARENT_DIR, 'frontend')
ASSETS_DIR = os.path.join(PARENT_DIR, 'assets')

@app.route('/')
def index():
    """Serve the main landing page"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    """Serve frontend static files"""
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve asset files (images, etc.)"""
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'ok', 'message': 'NeuroGen backend is running'}

if __name__ == '__main__':
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Assets directory: {ASSETS_DIR}")
    print("Starting NeuroGen backend server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
