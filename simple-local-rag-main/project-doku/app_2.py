from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.rag_pipeline_20 import *
from scripts.rag_body_102 import *
import re


# Flask App Setup
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
embedding_model = SentenceTransformer("all-mpnet-base-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id, use_quantization_config = suggest_llm_based_on_gpu()
tokenizer, model = check_and_load_llm_model(model_id, use_quantization_config, device)

def remove_until_patterns_in_place(my_list, patterns):
    """
    Removes all elements from the list until one of the specified patterns is found, modifying the list in place.
    """
    if patterns in my_list:
        index = my_list.index(patterns)
        # Delete all items before the codeword
        index = index+2
        del my_list[:index]
    else:
        print(f"Codeword '{patterns}' not found in the list.")
    return my_list



def generate_subqueries(query: str) -> list:
    """
    Nutzt das LLM, um eine Liste von Subqueries zu erstellen, die zur besseren Beantwortung der Frage dienen.
    """
    prompt = f"""
Erstelle eine Liste spezifischer Stichpunkte oder Subqueries, die helfen, die Frage präziser zu beantworten.
Verwende nur kurze Stichworte oder Phrasen (1 bis 3 Wörter pro Eintrag).
Vermeide längere Erklärungen; die Liste soll eine kompakte Auswahl von 5 Schlüsselbegriffen sein, die in einer Datenbank zur Suche verwendet werden könnten. Ergänze auch ähnliche Begriffe, die thematisch passen.

Formatiere die Liste wie folgt:

1. Stichpunkt/Subquery 1
2. Stichpunkt/Subquery 2
3. Stichpunkt/Subquery 3
...
Die Frage lautet: '{query}'
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrahiere die Subqueries aus der Antwort und filtere sie
    subqueries = response.strip().split("\n")
    
    patterns ="Die Frage lautet: 'Was ist Geisbeinflussung?'"
    remove_until_patterns_in_place(subqueries, patterns)
    
    subqueries = [subquery.strip() for subquery in subqueries if subquery]  # Filtere leere Einträge
    return subqueries




def extract_subqueries(response: list) -> list:
    # Reguläre Ausdrücke für Subqueries und Unterfragen
    subquery_pattern = r"(\d+)\.\s*(.*?)\s*$"
    print(f"\nsubquery_matches{response}\n")
    # Strukturierte Liste der Subqueries erstellen
    subqueries = []
    for query in response:
        cleaned_text = re.sub(r'^[^a-zA-ZäöüÄÖÜ]*', '', query, flags=re.MULTILINE)
        subqueries.append(cleaned_text)
    return subqueries


def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Erzeugt einen Prompt zur Beantwortung der Frage basierend auf Kontextinformationen.
    """
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = f"""
    Bitte beantworte die folgende Frage basierend auf den bereitgestellten Kontextinformationen.
    Lies die relevanten Passagen sorgfältig, um die Frage präzise zu beantworten.
    Gib nur die Antwort, ohne die Gedanken zu wiederholen. Achte darauf, die Antwort möglichst klar und in maximal 30 Wörtern oder kürzer zu formulieren!
    Nutze die folgenden Kontextinformationen zur Beantwortung der Benutzerfrage:
    {context}
    Benutzeranfrage: {query}
    """
    return base_prompt


@app.route('/upload', methods=['POST'])
def upload_pdf():
    feedback = {}
    
    if 'file' not in request.files or 'query' not in request.form:
        feedback["error"] = "PDF und Anfrage erforderlich"
        return jsonify(feedback), 400
    
    file = request.files['file']
    query = request.form.get("query")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Schritt 1: PDF verarbeiten und Chunking durchführen
        pages_and_texts = open_and_read_pdf(file_path)
        pages_and_texts = sentencize_text(pages_and_texts)
        pages_and_chunks = prepare_chunks(pages_and_texts, chunk_size=10)
        
        # Multi-Query-Retriever: Generiere Subqueries
        response = generate_subqueries(query)
        print((f"response type: {type(response)}"))
        subqueries = extract_subqueries(response)
        feedback["subqueries"] = subqueries
        print(f"subqueries: {subqueries}")

        all_retrieval_results = []
        
        # Schritt 2: Einbettungen für das PDF laden oder erstellen
        if os.path.exists(check_and_embed_pdf(file_path)):
            csv_path = check_and_embed_pdf(file_path)
            print("file exists")
        else:
            print("\n file dont exists \n")
            filtered_chunks = token_filter(pages_and_chunks, min_token_length=30)
            embeddings_tensor = embed_text(filtered_chunks)
            csv_path = embed_text_save(filtered_chunks, file_path)
        embeddings = embeddings_to_tensor(csv_path, device)
        
        # Für jede Subquery Kontext extrahieren und die besten Ergebnisse finden
        all_retrieval_results = []
        
        # Für jede Subquery Kontext extrahieren und die besten Ergebnisse finden
        for subquery in subqueries:
            # Finde die besten Übereinstimmungen für die aktuelle Subquery
            top_results = query_topk(embeddings, subquery, embedding_model)
            
            # Extrahiere Werte und Indizes aus den `topk`-Ergebnissen
            top_values = top_results.values
            top_indices = top_results.indices
        
            print(f"\nErgebnisse für Subquery '{subquery}':")
            for value, index in zip(top_values, top_indices):
                # Überprüfe, ob der Wert über dem Schwellenwert liegt
                if value.item() > 0.4:
                    print(f"Wert: {value.item():.4f}, Index: {index.item()}")  # Ausgabe der Ergebnisse
        
                    # Füge nur eindeutige Elemente hinzu, um doppelte Ergebnisse zu vermeiden
                    if pages_and_chunks[index.item()] not in all_retrieval_results:
                        print(f"\n {pages_and_chunks[index.item()]} \n {pages_and_chunks[index.item()]['chunk_token_count']/pages_and_chunks[index.item()]['chunk_word_count']}\n")
                        if pages_and_chunks[index.item()]['chunk_token_count']/pages_and_chunks[index.item()]['chunk_word_count'] < 3:
                            all_retrieval_results.append(pages_and_chunks[index.item()])
                        
        
        print("\nKombinierte Kontexte für alle Subqueries (nur Werte über 0.4):")
        for i, context in enumerate(all_retrieval_results, 1):
            print(f"{i}. {context}")


        print(all_retrieval_results)
    
        # Kombinierten Prompt für das LLM erstellen
        prompt = prompt_formatter(query, all_retrieval_results)
        
        # Antwort generieren
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"], max_length=900)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(answer)
        
        feedback["llm_answer"] = answer
        return jsonify({"feedback": feedback, "answer": answer})
    
    except Exception as e:
        feedback["error"] = f"An error occurred: {str(e)}"
        return jsonify(feedback), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
