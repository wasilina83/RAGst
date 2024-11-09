import torch
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import pandas as pd

# Funktion zum Laden und Umwandeln von Embeddings in einen Tensor
def embeddings_to_tensor(embeddings_df_save_path: str, device: torch.device) -> torch.Tensor:
    """
    Lädt Embeddings aus einer CSV-Datei, konvertiert sie in einen Tensor und verschiebt sie auf das angegebene Gerät (GPU/CPU).
    
    Parameter:
        embeddings_df_save_path (str): Der Pfad zur gespeicherten CSV-Datei mit den Embeddings.
        device (torch.device): Das Gerät, auf das die Embeddings verschoben werden (z. B. 'cuda' für GPU oder 'cpu').
        
    Rückgabewert:
        torch.Tensor: Tensor, der alle Embeddings enthält und bereit für die Modellverwendung ist.
    """
    
    # CSV mit Textabschnitten und ihren Embeddings laden
    try:
        text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei unter dem Pfad '{embeddings_df_save_path}' wurde nicht gefunden.")
    
    # Sicherstellen, dass die 'embedding' Spalte existiert
    if 'embedding' not in text_chunks_and_embedding_df.columns:
        raise ValueError("Die CSV-Datei muss eine Spalte namens 'embedding' enthalten.")

    # String-Darstellungen der Embeddings in numpy Arrays umwandeln
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")  # Entfernt eckige Klammern und trennt durch Leerzeichen
    )

    # DataFrame in eine Liste von Dictionaries umwandeln
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    print("Erste 5 Zeilen der CSV-Datei mit Embeddings:")
    print(text_chunks_and_embedding_df.head())

    # Embeddings in einen Tensor umwandeln und auf das angegebene Gerät verschieben
    embeddings = torch.tensor(
        np.array(text_chunks_and_embedding_df["embedding"].tolist()),  # Umwandlung der Liste von Arrays in ein numpy Array
        dtype=torch.float32  # Datentyp auf float32 setzen, da dies häufig verwendet wird
    ).to(device)  # Verschieben auf das angegebene Gerät (z. B. GPU oder CPU)
    
    return embeddings



# Funktion zur Berechnung der Top-k Ergebnisse basierend auf der Ähnlichkeit zwischen der Anfrage und den Embeddings
def query_topk(embeddings, query, embedding_model, top_k=5):
    """
    Embeds the query and finds the top-k most similar results from the embeddings.

    Parameters:
        embeddings (torch.tensor): The embeddings of the text chunks.
        query (str): The query to be embedded.
        embedding_model (SentenceTransformer): The model used to embed the query.
        top_k (int): Number of top results to return.

    Returns:
        top_results_dot_product: Top-k scores and indices of the relevant text chunks.
    """
    # Embed the query using the same model used for text chunks
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Compute dot product scores between the query embedding and all text embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    # Get the top-k results based on the highest dot product scores
    top_results_dot_product = torch.topk(dot_scores, k=top_k)
    
    return top_results_dot_product

# Hilfsfunktion, um den Text sauber zu umbrechen (nützlich für die Ausgabe)
def print_wrapped(text, wrap_length=80):
    """
    Print wrapped text (for better readability).
    
    Parameters:
        text (str): The text to be wrapped.
        wrap_length (int): Maximum number of characters per line.
    """
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Funktion zum Ausgeben der relevanten Ergebnisse mit Scores und Text
def output_result(query, embeddings, pages_and_chunks, embedding_model):
    """
    Outputs the top-k results for a given query, including the relevant text chunk and page number.
    
    Parameters:
        query (str): The query to search for.
        embeddings (torch.tensor): The embeddings of the text chunks.
        pages_and_chunks (list[dict]): A list of dictionaries containing the text chunks and their metadata.
        embedding_model (SentenceTransformer): The model used to embed the query.
    """
    print(f"Query: '{query}'\n")
    print("Results:")
    
    # Get the top-k results based on the dot product similarity
    top_results_dot_product = query_topk(embeddings, query, embedding_model)
    
    # Loop through the top results and print their details
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
    
    # Return the page number of the most relevant result
    return pages_and_chunks[top_results_dot_product[1][0]]['page_number']

# Funktion zur Extraktion der relevanten Seite aus der PDF
def get_page(pdf_path, top_result):
    """
    Retrieves the image of the page from the PDF corresponding to the top result.
    
    Parameters:
        pdf_path (str): The path to the PDF file.
        top_result (int): The index of the top result.
    
    Returns:
        img_array (numpy.ndarray): The image of the page.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    # Load the page (index 11 + top_result to account for the page number offset)
    page = doc.load_page(11 + top_result)
    
    # Get the image of the page with a high DPI for better quality
    img = page.get_pixmap(dpi=300)
    
    # Optionally, save the image as PNG
    img.save("output_filename.png")
    
    # Convert the Pixmap to a numpy array for further processing
    img_array = np.frombuffer(img.samples_mv, dtype=np.uint8).reshape((img.h, img.w, img.n))
    
    doc.close()
    
    return img_array

# Funktion zur Anzeige des relevanten Ergebnisses als Bild
def plt_best_res(img_array, query):
    """
    Displays the image of the relevant page with the query title.
    
    Parameters:
        img_array (numpy.ndarray): The image array of the relevant page.
        query (str): The query to be displayed in the title.
    """
    plt.figure(figsize=(13, 10))
    plt.imshow(img_array)
    plt.title(f"Query: '{query}' | Relevante Seite:")
    plt.axis('off')  # Turn off the axis for a cleaner display
    plt.show()

# Funktion zum Abrufen der relevanten Ressourcen und Ausgeben der besten Ergebnisse
def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model: SentenceTransformer, n_resources_to_return: int=5):
    """
    Embeds a query with the given model and returns the top-k most relevant resources.
    
    Parameters:
        query (str): The query to be embedded.
        embeddings (torch.tensor): The embeddings of the text chunks.
        model (SentenceTransformer): The model used to embed the query.
        n_resources_to_return (int): The number of resources to return.
    
    Returns:
        scores (torch.Tensor): The similarity scores for the top-k results.
        indices (torch.Tensor): The indices of the top-k results.
    """
    # Embed the query using the same model as the text chunks
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute dot product similarity scores between the query embedding and the text embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    # Get the top-k results based on the highest similarity scores
    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    
    return scores, indices

# Funktion zur Anzeige der besten Ergebnisse und ihrer Scores
def print_top_results_and_scores(query: str, embeddings: torch.tensor, pages_and_chunks, n_resources_to_return=5):
    """
    Prints the top-k most relevant results for the given query and their corresponding details.
    
    Parameters:
        query (str): The query to be processed.
        embeddings (torch.tensor): The embeddings of the text chunks.
        pages_and_chunks (list[dict]): A list of dictionaries containing the text chunks and metadata.
        n_resources_to_return (int): The number of resources to display.
    """
    # Retrieve top-k results based on query embedding
    scores, indices = retrieve_relevant_resources(query, embeddings, embedding_model, n_resources_to_return)
    
    print(f"Query: '{query}'\n")
    print("Results:")
    
    # Print the results with their scores and relevant text chunks
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

# Funktion zum Abrufen des verfügbaren GPU-Speichers
def get_gpu_memory():
    """
    Gibt den gesamten verfügbaren GPU-Speicher in GB zurück.
    """
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30), 2)
    print(f"Available GPU memory: {gpu_memory_gb} GB")
    return gpu_memory_gb

# Funktion zur Bestimmung des geeigneten LLM-Modells basierend auf der GPU-Kapazität
def suggest_llm_based_on_gpu():
    """
    Bestimmt das geeignete LLM-Modell basierend auf dem verfügbaren GPU-Speicher.
    Gibt das Modell-ID und die Quantisierungskonfiguration zurück.
    """
    gpu_memory_gb = get_gpu_memory()

    if gpu_memory_gb < 5.1:
        print(f"Your available GPU memory is {gpu_memory_gb} GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        model_id = "google/gemma-2-2b-it"
        use_quantization_config = True
    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
        model_id = "google/gemma-2-2b-it"
        use_quantization_config = True
    elif gpu_memory_gb < 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        model_id = "google/gemma-2-2b-it"
        use_quantization_config = False
    else:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 7B in 4-bit or float16 precision.")
        model_id = "google/gemma-7b-it"
        use_quantization_config = False

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")
    return model_id, use_quantization_config

# Funktion zum Abrufen eines geeigneten LLM-Modells basierend auf GPU-Kapazität
def get_llm_model():
    """
    Wählt das Modell aus und lädt es basierend auf der GPU-Kapazität und den empfohlenen Einstellungen.
    """
    model_id, use_quantization_config = suggest_llm_based_on_gpu()

    # Erstellen einer Quantisierungs-Konfiguration für kleinere Modelle (optional)
    quantization_config = None
    if use_quantization_config:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    # Optional: Setup Flash Attention 2 für schnellere Inferenz, falls verfügbar
    if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    # Tokenizer und Modell instanziieren
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id, 
        torch_dtype=torch.float16, 
        quantization_config=quantization_config, 
        low_cpu_mem_usage=False, 
        attn_implementation=attn_implementation
    )

    # Falls keine Quantisierung verwendet wird, das Modell auf GPU verschieben
    if not use_quantization_config:
        llm_model.to("cuda")

    return tokenizer, llm_model

# Funktion zur Berechnung der Anzahl der Modellparameter
def get_model_num_params(model: torch.nn.Module):
    """
    Berechnet die Anzahl der Parameter eines Modells.
    """
    return sum([param.numel() for param in model.parameters()])

# Funktion zur Berechnung des Speicherbedarfs eines Modells in GB
def get_model_mem_size(model: torch.nn.Module):
    """
    Berechnet die Größe des Modells in GB.
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Gesamten Speicherbedarf berechnen
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)  # In MB
    model_mem_gb = model_mem_bytes / (1024**3)  # In GB

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}


    
def check_and_load_llm_model(model_id: str, use_quantization_config: bool = False, device: torch.device = None):
    """
    Überprüft, ob das LLM-Modell bereits geladen und installiert ist. Wenn nicht, wird es basierend auf der GPU-Kapazität heruntergeladen und geladen.
    
    Parameter:
        model_id (str): Das Modell-ID oder der Modellname.
        use_quantization_config (bool): Wenn das Modell in quantisierter Form geladen werden soll.
        device (torch.device): Das Gerät, auf dem das Modell geladen werden soll ('cuda' für GPU oder 'cpu').
    
    Rückgabewert:
        tokenizer: Der Tokenizer des Modells.
        model: Das geladene Modell.
    """
    # Überprüfen, ob das Modell bereits im lokalen Cache von HuggingFace vorhanden ist
    model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers", model_id)
    
    if os.path.exists(model_cache_dir):
        print(f"Das Modell '{model_id}' ist bereits im Cache vorhanden.")
    else:
        print(f"Das Modell '{model_id}' wird jetzt heruntergeladen und installiert...")
    
    try:
        # Tokenizer laden
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Optional: Quantisierungs-Konfiguration für Modelle mit hohem Speicherbedarf
        quantization_config = None
        if use_quantization_config:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        # Modell laden
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # oder torch.float32, je nach Modellanforderungen
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        
        # Falls auf GPU verfügbar und das Modell nicht quantisiert ist, auf die GPU verschieben
        if device and not use_quantization_config:
            model.to(device)
        
        print(f"Das Modell '{model_id}' wurde erfolgreich geladen.")
        return tokenizer, model

    except Exception as e:
        print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nFehler beim Laden des Modells: {str(e)}\n\n\n\n\n\n\n\n")
        return None, None


