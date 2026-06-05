import os
import sys
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = "/content/drive/MyDrive/Software"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

app = Flask(__name__)
CORS(app, origins="*")

from backend.inference_engine import analyze

@app.route('/')
def home():
    return jsonify({"message": "Mental Health Backend Running"})

@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        data = request.get_json()
        if not data or "text" not in data or "audio" not in data or "visual" not in data:
            return jsonify({"status": "error", "message": "Missing text, audio or visual"}), 400

        text_feat   = np.array(data["text"],   dtype=np.float32)
        audio_feat  = np.array(data["audio"],  dtype=np.float32)
        visual_feat = np.array(data["visual"], dtype=np.float32)

        scores = analyze(text_feat, audio_feat, visual_feat)
        result = {k: float(v) for k, v in scores.items()}
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
