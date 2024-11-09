from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.rag_pipeline_20 import process_pdf_and_generate_embeddings
from scripts.rag_body_102 import (
    embeddings_to_tensor,
    query_topk,
    output_result,
    get_gpu_memory,
    suggest_llm_based_on_gpu,
    check_and_load_llm_model
)

# Flask App Setup
app = Flask(__name__)
CORS(app)  # Aktiviert Cross-Origin-Anfragen

# File Upload Folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
# Load LLM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id, use_quantization_config = suggest_llm_based_on_gpu()
print(f" unsere id: {model_id}")
tokenizer, model = check_and_load_llm_model(model_id, use_quantization_config, device)
# Funktion zur Formatierung eines Prompts mit Kontextinformationen
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Erzeugt einen Prompt, der die Benutzeranfrage mit dem gegebenen Kontext kombiniert.
    """
    # Kontext in einen einzigen Absatz umwandeln
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    
    # Basis-Prompt für das Modell erstellen
    base_prompt = """
    Beantworte bitte die Frage anhand der folgenden Kontextinformationen. 
    Lass dir Zeit zum Nachdenken, indem du die relevanten Passagen aus dem Kontext nimmst, bevor du die Frage beantwortest. 
    Wiederhole nicht die Gedanken, sondern nur die Antwort. Achte darauf, dass deine Antwort so weit wie möglich die Frage erklärt, jedoch acuh seht kurz bitte. Fasse dich kurtz und in maximal 3 Sätzen!
    Verwende nun die folgenden Kontextinformationen, um die Benutzeranfrage zu beantworten:
    {context}
    Relevant Kontextelemente: <extract relevant passages from the context here>
    User query: {query}
    """

    # Prompt mit Kontext und Anfrage aktualisieren
    base_prompt = base_prompt.format(context=context, query=query)

    # Erstelle den Prompt im Dialogformat für das Modell
    dialogue_template = [
        {"role": "user", "content": base_prompt}
    ]

    # Wende das Template an
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)

    return prompt

# Clear GPU cache if CUDA is available
if torch.cuda.is_available():
    torch.cuda.empty_cache()



@app.route('/upload', methods=['POST'])
def upload_pdf():
    feedback = {}  # Initialisiere ein Dictionary, um Rückmeldungen zu sammeln

    if 'file' not in request.files:
        feedback["error"] = "No file provided"
        return jsonify(feedback), 400
    
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Hole die Benutzeranfrage ab
    query = request.form.get("query")
    if not query:
        feedback["error"] = "No query provided"
        return jsonify(feedback), 400

    try:
        # Schritt 1: PDF verarbeiten und Einbettungen erstellen
        feedback["embedding_start"] = "Start processing PDF and generating embeddings."
        csv_path, pages_and_chunks = process_pdf_and_generate_embeddings(pdf_path=file_path, chunk_size=10, min_token_length=30)
        feedback["embedding_done"] = f"Embeddings generated and saved to {csv_path}"

        # Schritt 2: Einbettungen laden
        feedback["embedding_loading"] = "Loading embeddings from CSV."
        embeddings = embeddings_to_tensor(csv_path, device)
        feedback["embedding_loaded"] = "Embeddings loaded successfully."

        # Schritt 3: Ähnlichkeitsvergleich durchführen
        feedback["similarity_check"] = "Performing similarity check with query."
        top_results_dot_product = query_topk(embeddings, query, embedding_model)
        feedback["similarity_done"] = "Similarity check completed."

        # Schritt 4: Ergebnis anhand der Einbettungen ermitteln
        feedback["result_output"] = "Extracting relevant context from embeddings."
        page_number = output_result(query, embeddings, pages_and_chunks, embedding_model)
        feedback["page_number"] = f"Best match found on page {page_number}."

        # Schritt 5: Antwort mit LLM erstellen
        feedback["llm_prompt"] = "Generating answer using LLM based on context."
        context_items = [pages_and_chunks[idx] for idx in top_results_dot_product[1]]
        prompt = prompt_formatter(query, context_items)
        
        # Tokenize und generiere Antwort
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"], max_length=600)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        feedback["llm_answer"] = "Answer generated successfully."

        # Rückgabe der Antwort und des Seitenzahl-Ergebnisses
        return jsonify({"feedback": feedback, "answer": answer, "page_number": page_number})
    except Exception as e:
        feedback["error"] = f"An error occurred: {str(e)}"
        return jsonify(feedback), 500

if __name__ == '__main__':
    app.run(debug=True)
