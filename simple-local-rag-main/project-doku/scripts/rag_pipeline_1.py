# Imports and setup
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

torch.cuda.empty_cache()
# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
llm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to(device)

# Clear cache after loading the model
torch.cuda.empty_cache()


# Initialize Spacy NLP Model and SentenceTransformer Model
nlp = spacy.load("de_core_news_sm")

# Initialize the embedding model and move to the correct device
embedding_model = SentenceTransformer("all-mpnet-base-v2")
embedding_model.to(device)
torch.cuda.empty_cache()

# Helper function to clean up text
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

# PDF reading function
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "page_number": page_number,
            "text": text,
            "sentences": [str(sentence) for sentence in nlp(text).sents]
        })
    return pages_and_texts

# Splitting function for sentence chunks
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Chunk preparation with embeddings
def prepare_chunks(pages_and_texts: list[dict], chunk_size=10):
    pages_and_chunks = []
    for item in pages_and_texts:
        sentence_chunks = split_list(item["sentences"], chunk_size)
        for chunk in sentence_chunks:
            joined_chunk = " ".join(chunk).strip()
            pages_and_chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": joined_chunk,
                "embedding": embedding_model.encode(joined_chunk, convert_to_tensor=True).to(device)
            })
    return pages_and_chunks

# Main pipeline initialization
def initialize_pipeline(pdf_path: str):
    print("Loading PDF and processing text...")
    pages_and_texts = open_and_read_pdf(pdf_path)
    print("Preparing sentence chunks and embeddings...")
    pages_and_chunks = prepare_chunks(pages_and_texts)
    return pages_and_chunks

# Load and prepare embeddings on the correct device
#pages_and_chunks = initialize_pipeline(pdf_path="Sicherung-des-Schienenverkehrs.pdf")
#embeddings = torch.stack([item["embedding"] for item in pages_and_chunks])

# Retrieval function based on cosine similarity
def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
    scores, indices = torch.topk(util.dot_score(query_embedding, embeddings)[0], k=n_resources_to_return)
    context_items = [pages_and_chunks[i] for i in indices]
    return context_items

# Prompt formatting for text generation
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    return f"Based on the context, answer the query.\nContext items:\n{context}\nQuery: {query}\nAnswer:"

# Main function to ask questions
def ask(query, temperature=0.7, max_new_tokens=512):
    context_items = retrieve_relevant_resources(query=query)
    prompt = prompt_formatter(query=query, context_items=context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Example usage
#query = "was ist LST genau, wieso ist es wichtig?"
#print("Answer:\n", ask(query=query))

def run_rag_pipeline(file_path, query, tokenizer, llm_model, embedding_model):
    pages_and_chunks = initialize_pipeline(file_path)
    embeddings = torch.stack([item["embedding"] for item in pages_and_chunks])
    answer = ask(query, temperature=0.7, max_new_tokens=512)
    return answer
    