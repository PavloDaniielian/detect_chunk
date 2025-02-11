import nltk
import glob
import csv
import math
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.cluster import SpectralClustering

# Ensure sentence tokenizer is available
#nltk.download('punkt')

def chunk_text(text: str, max_chunk_size: int):
    # --- Step 1: Tokenization & Sentence Embeddings ---
    sentences = nltk.sent_tokenize(text)
    N = len(sentences)
    if N == 0:
        return []

    lengths = [len(s) for s in sentences]

    def chunk_size(indices):
        """Computes chunk size, including spaces between sentences."""
        return sum(lengths[i] for i in indices) + (len(indices) - 1)

    # --- Step 2: Compute Sentence Embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)

    # --- Step 3: Build Initial Chunk Boundaries ---
    target_chunks = int(math.ceil(len(text) / max_chunk_size) * 2)
    K = min(target_chunks, N)

    boundaries = [0]
    for i in range(1, K):
        boundaries.append(int(round(i * N / K)))
    boundaries.append(N)

    # --- Step 4: Deterministic Local Search with Adaptive Refinement ---
    def candidate_reward(boundaries):
        """Compute the score using check-chunker logic."""
        segments = []
        for i in range(len(boundaries) - 1):
            seg = list(range(boundaries[i], boundaries[i + 1]))
            if chunk_size(seg) > max_chunk_size:
                return -1000  # Heavy penalty for invalid chunk
            segments.append(seg)

        check_chunks = []
        for seg_index, seg in enumerate(segments):
            for j in range(0, len(seg), 3):
                group = seg[j:j+3]
                if len(group) == 0:
                    continue
                avg_emb = np.mean(sentence_embeddings[group], axis=0)
                check_chunks.append((seg_index, avg_emb))

        if not check_chunks:
            return -1000

        intra = []
        inter = []
        for i in range(len(check_chunks) - 1):
            for j in range(i + 1, len(check_chunks)):
                sim_val = np.dot(check_chunks[i][1], check_chunks[j][1])
                if check_chunks[i][0] == check_chunks[j][0]:
                    intra.append(sim_val)
                else:
                    inter.append(sim_val)

        avg_intra = np.mean(intra) if intra else 0
        avg_inter = np.mean(inter) if inter else 0
        return avg_intra - avg_inter

    current_reward = candidate_reward(boundaries)
    improved = True
    max_window = 10  # Increased search range
    while improved:
        improved = False
        for i in range(1, len(boundaries) - 1):
            best_boundary = boundaries[i]
            best_r = candidate_reward(boundaries)
            lower_limit = boundaries[i - 1] + 1
            upper_limit = boundaries[i + 1] - 1

            # Try shifting the boundary within a larger window
            for candidate in range(max(lower_limit, best_boundary - max_window), 
                                   min(upper_limit, best_boundary + max_window) + 1):
                new_boundaries = boundaries.copy()
                new_boundaries[i] = candidate
                r = candidate_reward(new_boundaries)
                if r > best_r:
                    best_r = r
                    best_boundary = candidate

            if best_boundary != boundaries[i]:
                boundaries[i] = best_boundary
                current_reward = best_r
                improved = True  # Keep searching if improvement found

    # --- Step 5: Reconstruct Optimized Chunks ---
    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            chunk = " ".join(sentences[j] for j in range(start, end))
            chunks.append(chunk)

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
            print(f"Chunking {input_file} max_chunk_size={max_chunk_size} ...")
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv([(i + 1, chunk) for i, chunk in enumerate(chunked_data)], output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")