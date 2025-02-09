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
    
    # Initialize model and compute sentence embeddings only once.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    # Normalize embeddings so that dot products equal cosine similarity.
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)
    
    # -------------------------
    # Initial segmentation based solely on character count.
    # Group contiguous sentences until reaching max_chunk_size.
    chunks_indices = []  # Each element is a list of sentence indices for that chunk.
    current_chunk = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sent_len = len(sentence)
        # If adding this sentence exceeds max_chunk_size and there is already content, start a new chunk.
        if current_chunk and (current_length + sent_len > max_chunk_size):
            chunks_indices.append(current_chunk)
            current_chunk = [i]
            current_length = sent_len
        else:
            current_chunk.append(i)
            current_length += sent_len
    if current_chunk:
        chunks_indices.append(current_chunk)
    
    # Calculate the target number of chunks.
    target_chunk_count = int(np.ceil(len(text) / max_chunk_size) * 2)
    if len(chunks_indices) >= target_chunk_count:
        return [" ".join(sentences[i] for i in indices) for indices in chunks_indices]
    
    # -------------------------
    # Helper functions:
    def compute_cumsum(indices):
        # Compute cumulative sum of embeddings for the given sentence indices.
        return np.cumsum(sentence_embeddings[indices], axis=0)
    
    def chunk_text_length(indices):
        # Approximate text length by summing sentence lengths and adding spaces between sentences.
        return sum(len(sentences[i]) for i in indices) + max(0, len(indices) - 1)
    
    # -------------------------
    # Iteratively split chunks until we reach the target chunk count.
    # For each splittable chunk (with at least 2 sentences), we try every valid boundary.
    # We use cumulative sums to quickly compute average embeddings of the left and right parts.
    while len(chunks_indices) < target_chunk_count:
        best_candidate_value = np.inf
        best_candidate = None  # Tuple: (chunk_index, split_position)
        
        # Consider every chunk.
        for chunk_idx, indices in enumerate(chunks_indices):
            if len(indices) < 2:
                continue  # Cannot split a chunk with only one sentence.
            cumsum = compute_cumsum(indices)
            N = len(indices)
            # Try every boundary within the chunk.
            for j in range(1, N):
                left_indices = indices[:j]
                right_indices = indices[j:]
                # Check that both new chunks obey the max_chunk_size constraint.
                if chunk_text_length(left_indices) > max_chunk_size or chunk_text_length(right_indices) > max_chunk_size:
                    continue
                # Compute average embeddings quickly using the cumulative sums.
                left_avg = cumsum[j-1] / j
                right_avg = (cumsum[-1] - cumsum[j-1]) / (N - j)
                candidate_value = np.dot(left_avg, right_avg)  # With normalized embeddings, this is cosine similarity.
                # We want the split that minimizes similarity between the two halves.
                if candidate_value < best_candidate_value:
                    best_candidate_value = candidate_value
                    best_candidate = (chunk_idx, j)
        
        # If no valid candidate split was found, exit the loop.
        if best_candidate is None:
            break
        
        chunk_idx, split_pos = best_candidate
        indices = chunks_indices.pop(chunk_idx)
        left_indices = indices[:split_pos]
        right_indices = indices[split_pos:]
        # Insert the two new chunks at the position of the original chunk.
        chunks_indices.insert(chunk_idx, left_indices)
        chunks_indices.insert(chunk_idx + 1, right_indices)
    
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

        for max_chunk_size in [2000,3000,4000]:
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv([(i + 1, chunk) for i, chunk in enumerate(chunked_data)], output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")