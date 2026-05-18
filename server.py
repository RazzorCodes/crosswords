import http.server
import json
import os

class DataCollectionHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            label = data.get('label')
            strokes = data.get('strokes')
            
            if label is not None and strokes is not None:
                with open('dataset.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data) + '\n')
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
                print(f"Saved sample for label: {label}")
            else:
                self.send_error(400, "Missing label or strokes")
        except Exception as e:
            self.send_error(500, str(e))

def run(server_class=http.server.HTTPServer, handler_class=DataCollectionHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting data collection server on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
