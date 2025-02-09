import nltk
import glob
import csv
import math
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from itertools import combinations

# Ensure sentence tokenizer is available
#nltk.download('punkt')

def chunk_text(text: str, max_chunk_size: int):
    # Ensure the NLTK punkt tokenizer is available.
    sentences = nltk.sent_tokenize(text)
    
    # Initialize the model and precompute sentence embeddings.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    
    # Initial chunking: Group contiguous sentences such that the chunk's character length is < max_chunk_size.
    chunks_indices = []  # each element is a list of sentence indices belonging to that chunk.
    current_chunk = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        # If adding the next sentence would exceed the max_chunk_size, start a new chunk.
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks_indices.append(current_chunk)
            current_chunk = [i]
            current_length = sentence_length
        else:
            current_chunk.append(i)
            current_length += sentence_length
    if current_chunk:
        chunks_indices.append(current_chunk)
    
    # Calculate the target number of chunks.
    target_chunk_count = int(np.ceil(len(text) / max_chunk_size) * 2)
    
    # If we already have enough chunks, reconstruct and return them.
    if len(chunks_indices) >= target_chunk_count:
        return [" ".join(sentences[i] for i in chunk) for chunk in chunks_indices]
    
    # Iteratively split chunks until the target chunk count is reached.
    # Instead of re-encoding chunk texts, we use the precomputed sentence_embeddings.
    while len(chunks_indices) < target_chunk_count:
        best_split_value = np.inf
        best_chunk_idx = None
        best_split_position = None
        
        # For each chunk that has at least two sentences, look for the adjacent pair with the lowest similarity.
        for chunk_idx, indices in enumerate(chunks_indices):
            if len(indices) < 2:
                continue
            # Check adjacent sentence pairs.
            for j in range(len(indices) - 1):
                sim = np.dot(sentence_embeddings[indices[j]], sentence_embeddings[indices[j+1]])
                if sim < best_split_value:
                    best_split_value = sim
                    best_chunk_idx = chunk_idx
                    best_split_position = j + 1  # position to split (i.e. after index j)
        
        # If no split was found (all chunks have one sentence), break out.
        if best_chunk_idx is None:
            break
        
        # Split the chosen chunk into two parts.
        indices = chunks_indices.pop(best_chunk_idx)
        left_indices = indices[:best_split_position]
        right_indices = indices[best_split_position:]
        # Insert the two new chunks back into our list.
        chunks_indices.insert(best_chunk_idx, left_indices)
        chunks_indices.insert(best_chunk_idx + 1, right_indices)
    
    # Reconstruct the text chunks from the sentence indices.
    chunks = [" ".join(sentences[i] for i in indices) for indices in chunks_indices]
    return chunks

def load_text(file_path):
    """Loads text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

    return [(i + 1, chunk) for i, chunk in enumerate(chunks)]

def save_chunks_to_csv(chunks, output_file):
    """Saves chunks as a CSV file with chunk numbers."""
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Index", "Chunk Content"])
        writer.writerows(chunks)

# Process all .txt files in the directory
if __name__ == "__main__":
    #max_chunk_size = 2000  # Set chunk size (2000, 3000, or 4000)

    for input_file in glob.glob("*.txt"):
        output_file_prefix = input_file.replace(".txt", "")
        # Load text
        text = load_text(input_file)

        for max_chunk_size in [2000]:
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv([(i + 1, chunk) for i, chunk in enumerate(chunked_data)], output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")