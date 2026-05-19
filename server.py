import http.server
import json
import os
import pickle
import numpy as np
import torch
from trainer_1 import predict as svm_predict
from cnn_builder import render_stroke_entry
from cnn_model import get_cnn_model

class DataCollectionHandler(http.server.BaseHTTPRequestHandler):
    _clf = None
    _cnn = None

    @classmethod
    def get_clf(cls):
        if cls._clf is None:
            if os.path.exists('letter_clf.pkl'):
                with open('letter_clf.pkl', 'rb') as f:
                    cls._clf = pickle.load(f)
        return cls._clf

    @classmethod
    def get_cnn(cls):
        if cls._cnn is None:
            if os.path.exists('cnn_model.pth'):
                cls._cnn = get_cnn_model()
                cls._cnn.load_state_dict(torch.load('cnn_model.pth'))
                cls._cnn.eval()
        return cls._cnn

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/stats':
            counts = {chr(i): 0 for i in range(65, 91)}
            total = 0
            if os.path.exists('dataset.jsonl'):
                with open('dataset.jsonl', 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            label = data.get('label', '').upper()
                            if label in counts:
                                counts[label] += 1
                                total += 1
                        except: pass
            
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'counts': counts, 'total': total}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/predict':
                strokes = data.get('strokes')
                if strokes:
                    # 1. SVM Prediction
                    clf = self.get_clf()
                    svm_label = '?'
                    svm_probs = {chr(i): 1.0/26 for i in range(65, 91)}
                    if clf:
                        svm_label, svm_probs_raw = svm_predict(clf, {'strokes': strokes})
                        svm_probs = {k: float(v) for k, v in svm_probs_raw.items()}
                    
                    # 2. CNN Prediction
                    cnn = self.get_cnn()
                    cnn_label = '?'
                    cnn_probs = {chr(i): 1.0/26 for i in range(65, 91)}
                    if cnn:
                        img = render_stroke_entry({'strokes': strokes})
                        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
                        with torch.no_grad():
                            outputs = cnn(img_tensor)
                            probs = torch.softmax(outputs, dim=1).numpy()[0]
                            cnn_probs = {chr(65+i): float(p) for i, p in enumerate(probs)}
                            cnn_label = chr(65 + np.argmax(probs))
                    
                    # Phase 2: Secondary Corum (Teacher)
                    # For now, combined confidence
                    teacher_label = svm_label
                    teacher_conf = svm_probs[svm_label]
                    
                    if cnn_label == svm_label:
                        teacher_conf = max(teacher_conf, cnn_probs[cnn_label])
                    
                    teacher_action = "none"
                    if teacher_conf > 0.85:
                        teacher_action = "accept"
                    elif teacher_conf > 0.60:
                        teacher_action = "prompt"
                    else:
                        teacher_action = "discard"

                    response = {
                        'svm': {'label': svm_label, 'probs': svm_probs},
                        'cnn': {'label': cnn_label, 'probs': cnn_probs},
                        'teacher': {
                            'label': teacher_label,
                            'confidence': teacher_conf,
                            'action': teacher_action
                        }
                    }
                    
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    return

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

                # Phase 4: Model Updates - SVM at 50+ new samples
                try:
                    with open('dataset.jsonl', 'r') as f:
                        count = sum(1 for _ in f)
                    if count > 0 and count % 50 == 0:
                        print(f"Triggering SVM retrain at {count} samples...")
                        os.system("python3 train_svm_server.py &")
                except: pass

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
