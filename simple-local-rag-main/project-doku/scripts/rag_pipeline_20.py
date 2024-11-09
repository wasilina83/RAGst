# Imports and Setup
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import fitz  # pymupdf
import pandas as pd
import re
import spacy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap
from transformers.utils import is_flash_attn_2_available 

# Load SpaCy's German language model
nlp = spacy.load("de_core_news_sm")
# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
nlp.add_pipe("sentencizer")
# Initialize the embedding model and move to the correct device
embedding_model = SentenceTransformer("all-mpnet-base-v2")
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to clean up text
def text_formatter(text: str) -> str:
    """
    Cleans up the input text by removing unnecessary newline characters and extra spaces.
    
    Parameters:
        text (str): The raw text to be formatted.
        
    Returns:
        str: The cleaned and formatted text.
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text
    
# PDF reading function (only focuses on text, rather than images/figures etc)
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file and reads its text content page by page, returning statistics for each page.
    
    Parameters:
        pdf_path (str): The file path to the PDF document.
        
    Returns:
        list[dict]: A list of dictionaries with information about each page, including:
                    - adjusted page number
                    - character count
                    - word count
                    - estimated token count (based on character length)
                    - extracted text content
    """
    doc = fitz.open(pdf_path)  # Open the PDF document
    pages_and_texts = []
    # Iterate over the pages of the document
    for page_number, page in tqdm(enumerate(doc), desc="Reading PDF"):  
        text = page.get_text()  # Extract plain text from page
        text = text_formatter(text)  # Clean the text
        
        # Collect statistics and the cleaned text for the current page
        pages_and_texts.append({
            "page_number": page_number + 1,  # Adjust page number to start from 1
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),  # Estimate based on periods
            "page_token_count": len(text) / 4,  # Approximation: 1 token ≈ 4 characters
            "text": text
        })
    return pages_and_texts


# Sentence Extraction Function
def sentencize_text(pages_and_texts: list[dict]) -> list[dict]:
    """
    Processes each page's text by splitting it into sentences using SpaCy and counts sentences accurately.
    
    Parameters:
        pages_and_texts (list[dict]): List of page dictionaries, each containing page text and statistics.
        
    Returns:
        list[dict]: The updated list with additional sentence-level data for each page.
    """
    for item in tqdm(pages_and_texts, desc="Splitting text into sentences"):
        # Use SpaCy to divide text into sentences
        item["sentences"] = list(nlp(item["text"]).sents)
        
        # Convert sentences to strings and count them
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    
    return pages_and_texts

# Function to split a list into chunks of a specified size
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits input_list into sublists of maximum size slice_size.
    
    For example, a list of 17 sentences with slice_size=10 will result in two chunks:
    [[10 elements], [7 elements]]
    
    Parameters:
        input_list (list): List of elements to be split.
        slice_size (int): Maximum number of elements in each chunk.
        
    Returns:
        list[list[str]]: A list of lists, where each sublist has up to slice_size elements.
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Function to chunk sentences within each page
def sentence_chunker(pages_and_texts: list[dict], chunk_size: int = 10) -> list[dict]:
    """
    Divides sentences from each page into smaller chunks of a specified size.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries with sentence data for each page.
        chunk_size (int): Number of sentences per chunk.
        
    Returns:
        list[dict]: Updated list of page dictionaries with sentence chunks and chunk count.
    """
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

# Function to prepare chunks and collect statistics
def prepare_chunks(pages_and_texts: list[dict], chunk_size: int = 10) -> list[dict]:
    """
    Processes each page by creating sentence chunks, calculating text statistics, and storing them as separate items.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page text and sentence information.
        chunk_size (int): Number of sentences per chunk.
        
    Returns:
        list[dict]: List of dictionaries with chunked sentence data, each treated as an independent item.
    """
    # Ensure pages have sentences chunked
    pages_and_texts = sentence_chunker(pages_and_texts, chunk_size)
    pages_and_chunks = []

    # Iterate over each page and each chunk, preparing chunk data
    for item in tqdm(pages_and_texts, desc="Preparing sentence chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {
                "page_number": item["page_number"],
                # Join sentences in the chunk into a single string (paragraph-like format)
                "sentence_chunk": " ".join(sentence_chunk).replace("  ", " ").strip()
            }
            # Clean up spacing issues (ensure space after full stops and capital letters)
            chunk_dict["sentence_chunk"] = re.sub(r'\.([A-Z])', r'. \1', chunk_dict["sentence_chunk"])

            # Calculate statistics for the chunk
            chunk_dict["chunk_char_count"] = len(chunk_dict["sentence_chunk"])
            chunk_dict["chunk_word_count"] = len(chunk_dict["sentence_chunk"].split())
            chunk_dict["chunk_token_count"] = len(chunk_dict["sentence_chunk"]) / 4  # Approximation: 1 token ≈ 4 characters
            
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

# Main pipeline function to process the PDF and prepare sentence chunks
def initialize_pipeline(pdf_path: str) -> list[dict]:
    """
    Initializes the processing pipeline by loading and analyzing the PDF, creating sentence chunks, and preparing embeddings.
    
    Parameters:
        pdf_path (str): File path to the PDF document.
        
    Returns:
        list[dict]: List of dictionaries, each containing chunk data with statistics.
    """
    print("Loading PDF and processing text...")
    pages_and_texts = open_and_read_pdf(pdf_path)
    df = pd.DataFrame(pages_and_texts)
    
    # Display statistics of the page data
    print("Page data summary:")
    print(df.describe().round(2))
    print(df.head())

    print("Preparing sentence chunks and computing embeddings...")
    pages_and_chunks = prepare_chunks(pages_and_texts)
    df_chunks = pd.DataFrame(pages_and_chunks)
    
    # Display chunk statistics
    print("Chunk data summary:")
    print(df_chunks.describe().round(2))
    
    return pages_and_chunks

# Filter function to retain only chunks above a minimum token count
def token_filter(pages_and_chunks: list[dict], min_token_length: int = 30) -> list[dict]:
    """
    Filters chunks that meet a minimum token count requirement.
    
    Parameters:
        pages_and_chunks (list[dict]): List of chunk dictionaries with token count information.
        min_token_length (int): Minimum token count to include a chunk.
        
    Returns:
        list[dict]: List of filtered chunks meeting the token count threshold.
    """
    df = pd.DataFrame(pages_and_chunks)
    filtered_chunks = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return filtered_chunks


# Initialize the embedding model on the GPU (requires GPU setup, e.g., NVIDIA RTX 4090)
embedding_model.to("cuda")

# Function to embed text chunks using a GPU
def embed_text(pages_and_chunks: list[dict], batch_size: int = 32):
    """
    Creates embeddings for each text chunk using the GPU, both individually and in batches.
    
    Parameters:
        pages_and_chunks (list[dict]): List of chunk dictionaries containing text to embed.
        batch_size (int): Number of chunks processed in each batch, for speed and memory management.
        
    Returns:
        torch.Tensor: Tensor of all embeddings for the text chunks.
    """
    # Individual embeddings for each chunk (stored directly in the dictionary)
    for item in tqdm(pages_and_chunks, desc="Generating individual embeddings"):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    
    # Create a list of all text chunks
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]

    # Batch embeddings for all text chunks
    text_chunk_embeddings = embedding_model.encode(
        text_chunks,
        batch_size=batch_size,
        convert_to_tensor=True
    )

    return text_chunk_embeddings


# Function to save embeddings to a CSV file
# Funktion, um Embeddings lokal im Embeddings-Ordner zu speichern
def embed_text_save(pages_and_chunks: list[dict], pdf_name: str, embeddings_dir: str = "./docs/embeddings") -> str:
    """
    Speichert die Embeddings und Textschnipsel in einer CSV-Datei im angegebenen Ordner,
    wobei der Name der Datei mit dem Namen der PDF übereinstimmt.

    Parameter:
        pages_and_chunks (list[dict]): Liste der Dictionaries mit Textschnipseln und deren Embeddings.
        pdf_name (str): Der Name der PDF-Datei, aus dem der Name der Embedding-Datei abgeleitet wird.
        embeddings_dir (str): Der Ordner, in dem die Embedding-Datei gespeichert wird (Standard: "./embeddings").

    Returns:
        str: Der Pfad zur gespeicherten Embedding-Datei.
    """
    # Sicherstellen, dass der Zielordner existiert
    os.makedirs(embeddings_dir, exist_ok=True)

    # PDF-Dateinamen in ein Dateiformat umwandeln (ohne Erweiterung) und an den Ordnernamen anhängen
    pdf_base_name = os.path.splitext(os.path.basename(pdf_name))[0]
    save_path = os.path.join(embeddings_dir, f"{pdf_base_name}_embeddings.csv")

    # Embeddings und Textschnipsel in eine CSV-Datei speichern
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
    text_chunks_and_embeddings_df.to_csv(save_path, index=False)

    return save_path


# Funktion zur Überprüfung und Speicherung der Embeddings
def check_and_embed_pdf(file_path: str) -> str:
    """
    Überprüft, ob Embeddings bereits existieren, und erstellt sie bei Bedarf.
    
    Parameter:
        file_path (str): Der Pfad der hochgeladenen PDF-Datei.
        pages_and_chunks (list[dict]): Liste der Textschnipsel und deren Seiteninformationen.
        
    Returns:
        str: Der Pfad zur Embedding-Datei.
    """
    embeddings_dir = "./docs/embeddings"
    pdf_base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(embeddings_dir, f"{pdf_base_name}_embeddings.csv")
    return save_path


# Mainfunktion zum Prozessieren der PDF und Generierung der Embeddings






