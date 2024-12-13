from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.rag_pipeline_20 import *
from scripts.rag_body_102 import *
import re
import json

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
        index = index+1
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
Vermeide längere Erklärungen; die Liste soll eine kompakte Auswahl von 5 umformulierten Fragen sein. Ergänze auch ähnliche Begriffe, die thematisch passen. Formuliere immer die 5 Sätze kurz.
Wischtig: Fange die liste immer mit der fettgeschriebenen Übeschrifft "Antwort:" an.
...
Die Frage lautet: '{query}'
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    # Extrahiere die Subqueries aus der Antwort und filtere sie
    subqueries = response.strip().split("\n")
    
    patterns ="**Antwort:**"
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
    Gib nur die Antwort, ohne die Gedanken zu wiederholen. Achte darauf, die Antwort möglichst klar kürzer zu formulieren!  
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
        print((f"response type: {type(response)} {response}"))
        subqueries = extract_subqueries(response)
        subqueries=[query.replace('**', '') for query in subqueries]
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
        retrieval_with_scores = []
        added_indices = set()
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
                if value.item() > 0.29:
                    print(f"Wert: {value.item():.2f}, Index: {index.item()}")  # Ausgabe der Ergebnisse
                    if index.item() not in added_indices:
                        # Füge den Index zum Set hinzu
                        added_indices.add(index.item())
        
                        # Füge nur eindeutige Elemente hinzu, um doppelte Ergebnisse zu vermeiden
                        retrieval_with_scores.append((value.item(), pages_and_chunks[index.item()]))
                        
        
        # Sortiere die Ergebnisse nach den `value`-Werten in absteigender Reihenfolge
        retrieval_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Extrahiere die sortierten Kontexte
        all_retrieval_results = [context for _, context in retrieval_with_scores]
        
        print("\nKombinierte Kontexte für alle Subqueries (sortiert nach Wert):")
        for i, context in enumerate(all_retrieval_results, 1):
            print(f"{i}. {context}")
        
        # Kombinierten Prompt für das LLM erstellen
        prompt = prompt_formatter(query, all_retrieval_results)
        
        # Antwort generieren
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        lenge = len(inputs["input_ids"][0])
        # Maximale Eingabelänge basierend auf Modellgrenze berechnen
        max_input_length = model.config.max_position_embeddings - 7000  # Puffer für Antwort
        
        if len(inputs["input_ids"][0]) > max_input_length:
            # Kontext kürzen, bis die Länge passt
            while lenge > max_input_length:
                lenge = len(inputs["input_ids"][0])
                all_retrieval_results.pop(-1)  # Entferne das am wenigsten relevante Element
                prompt = prompt_formatter(query, all_retrieval_results)
                
                # Tokenize und generiere Antwort
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        
                lenge = len(inputs["input_ids"][0])
        
        outputs = model.generate(inputs["input_ids"], max_length=1500)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(answer)
        
        feedback["llm_answer"] = answer
        filename = f"{file.filename}-a2.json"
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(output_file_path):
            with open(output_file_path , "w") as output_file:
                json.dump({}, output_file, indent=4)  # Leeres JSON-Objekt erstellen
            print(f"{output_file_path} wurde erstellt und mit einem leeren JSON-Objekt initialisiert.")
        else:
            print(f"{output_file_path} existiert bereits.")

                # Save rtrn to a file
        
        with open(output_file_path, "r") as output_file:
            existing_data = json.load(output_file)
            new_response = {"feedback": feedback, "answer": answer}
            existing_data[query] = new_response
        with open(output_file_path, "w") as output_file:
            json.dump(existing_data, output_file, indent=4)

        print(f"Response saved to {output_file_path}")
        
        return jsonify({"feedback": feedback, "answer": answer})
    
    except Exception as e:
        feedback["error"] = f"An error occurred: {str(e)}"
        return jsonify(feedback), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
