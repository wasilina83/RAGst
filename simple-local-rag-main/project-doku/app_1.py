from flask import Flask, request, jsonify
import os
import torch
from scripts.rag_pipeline_1 import run_rag_pipeline

app = Flask(__name__)

# Setup file upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the embedding model only if needed
# e.g., embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
torch.cuda.empty_cache()
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Run RAG pipeline
    query = request.form.get("query")
    response = run_rag_pipeline(file_path, query, embedding_model)

    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)
