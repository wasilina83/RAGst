from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.rag_pipeline_20 import *
from scripts.rag_body_102 import *


# Flask App Setup
app = Flask(__name__)
CORS(app)  # Aktiviert Cross-Origin-Anfragen
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
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
    Bitte beantworte die folgende Frage basierend auf den bereitgestellten Kontextinformationen.
    Lies die relevanten Passagen sorgfältig, um die Frage präzise zu beantworten.
    Gib nur die Antwort, ohne die Gedanken zu wiederholen. Achte darauf, die Antwort möglichst klar und in maximal 30 Wörtern oder kürzer zu formulieren.
    Nutze die folgenden Kontextinformationen zur Beantwortung der Benutzerfrage:
    {context}
    Wichtige Kontextelemente: <extrahiere relevante Passagen aus dem Kontext hier>
    Benutzeranfrage: {query}
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
    feedback = {}
    feBenutzeredback = {}  # Initialisiere ein Dictionary, um Rückmeldungen zu sammeln

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
        print("PDF embedding - Step 1: Loading PDF and preparing sentence chunks...")
        feedback["embedding_start"] = "Start loading PDF and preparing sentence chunks."
        pages_and_texts = open_and_read_pdf(pdf_path=file_path)
        print("S2")
        feedback["embedding_S1.1.1"] = "Start sentencize text."
        pages_and_texts = sentencize_text(pages_and_texts)
        print("S2")
        feedback["embedding_S1.1.2"] = "Start sentencize text."
        pages_and_chunks = prepare_chunks(pages_and_texts, chunk_size=10)
        print("S2")
        if os.path.exists(check_and_embed_pdf(file_path)):
            print(f"Embeddings für {check_and_embed_pdf(file_path)} existieren bereits, Verwendung der bestehenden Datei.")
            csv_path = check_and_embed_pdf(file_path)
        else:                               
            print("Step 2: Filtering sentence chunks by token count...")
            feedback["embedding_S1.2"] = "Start filtering sentence chunks by token count"
            filtered_chunks = token_filter(pages_and_chunks, min_token_length=30)
    
            print("Step 3: Generating embeddings for filtered chunks...")
            feedback["embedding_S1.3"] = "Step 3: Generating embeddings for filtered chunks..."
            embeddings_tensor = embed_text(filtered_chunks)
    
            print("Step 4: Saving embeddings to CSV...")
            feedback["embedding_S1.4"] = "Step 4: Saving embeddings to CSV..."
            csv_path = embed_text_save(filtered_chunks, file_path)  # PDF-Pfad an die Funktion übergeben
            print(f"Embeddings saved to {csv_path}")
            print("Process complete.")
        
        print("S3")
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
        context_items = [pages_and_chunks[idx] for idx in top_results_dot_product[1][:3]]
        prompt = prompt_formatter(query, context_items)
        
        # Tokenize und generiere Antwort
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        lenge= len(inputs["input_ids"][0])
        # Maximale Eingabelänge basierend auf Modellgrenze berechnen
        max_input_length = model.config.max_position_embeddings - 7000  # Puffer für Antwort
        print(f"\n\n  len(inputs input_ids 0 ) {lenge}\n\n")
        print(f"\n\n  max_input_length {max_input_length}\n\n")
        if len(inputs["input_ids"][0]) > max_input_length:
            # Kontext kürzen, bis die Länge passt
            while len(inputs["input_ids"][0]) > max_input_length:
                print(f"context_items1 {context_items}")
                context_items.pop(-1)  # Entferne das am wenigsten relevante Element
                print(f"context_items2 {context_items}")
            prompt = prompt_formatter(query, context_items)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(inputs["input_ids"], max_length=1200)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback["llm_answer"] = "Answer generated successfully"

        

        # Rückgabe der Antwort und des Seitenzahl-Ergebnisses
        
        rtrn=jsonify({"feedback": feedback, "answer": answer, "page_number": page_number})
        print(rtrn)
        return jsonify({"answer": answer})
    except Exception as e:
        feedback["error"] = f"An error occurred: {str(e)}"
        return jsonify(feedback), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
