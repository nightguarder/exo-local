from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
from pathlib import Path

class ModelFileHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.models_dir = Path(os.getenv("MODEL_CACHE", "models"))
        super().__init__(*args, directory=str(self.models_dir), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_file_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ModelFileHandler)
    print(f"File server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_file_server()