import http.server
import socketserver
import os
import sys

DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web'))
PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add required headers for SharedArrayBuffer
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

# Enable address reuse
class CustomServer(socketserver.TCPServer):
    allow_reuse_address = True

try:
    with CustomServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Serving directory '{DIRECTORY}' at http://localhost:{PORT}")
        print(f"Application context root: http://localhost:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            # Force complete exit
            sys.exit(0)
except OSError as e:
    print(f"Error: {e}")
    sys.exit(1)
