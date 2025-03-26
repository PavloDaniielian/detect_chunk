import nltk
import glob
import csv
import math
import random
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from itertools import combinations

# Ensure sentence tokenizer is available
#nltk.download('punkt')
#nltk.download('punkt_tab')

def chunk_text(text: str, max_chunk_size: int):
    # Ensure the NLTK punkt tokenizer is available.
    sentences = nltk.sent_tokenize(text)
    N = len(sentences)
    if N == 0:
        return []
    lengths = [len(s) for s in sentences]
    
    # Helper: Compute chunk size (sum of sentence lengths plus one space between each pair).
    def chunk_size(indices):
        return sum(lengths[i] for i in indices) + (len(indices) - 1)
    
    # --- Step 2: Determine target number of chunks.
    # We require exactly: target_chunks = ceil(len(text)/max_chunk_size) * 2, capped by N.
    target_chunks = int(math.ceil(len(text) / max_chunk_size) * 2)
    K = min(target_chunks, N)  # number of chunks
    if K < 1:
        K = 1

    # --- Step 3: Compute sentence embeddings (normalized).
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)

    # --- Step 4: Define the candidate reward function.
    # This function mimics the check_chunker:
    #   - For each segment (as defined by boundaries), we group its sentences in consecutive groups of up to 3.
    #   - Each group’s embedding is computed as the average of its sentence embeddings.
    #   - Then, reward = (mean intra-chunk similarity) - (mean inter-chunk similarity).
    # If any segment violates the size constraint, we return a heavy penalty.
    def candidate_reward(boundaries):
        segments = []
        for i in range(len(boundaries) - 1):
            seg = list(range(boundaries[i], boundaries[i+1]))
            if chunk_size(seg) > max_chunk_size:
                return -1000  # penalty for violating size constraint
            segments.append(seg)
        check_chunks = []
        # Split each segment into groups of up to 3 sentences.
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
            for j in range(i+1, len(check_chunks)):
                sim_val = np.dot(check_chunks[i][1], check_chunks[j][1])
                if check_chunks[i][0] == check_chunks[j][0]:
                    intra.append(sim_val)
                else:
                    inter.append(sim_val)
        avg_intra = np.mean(intra) if intra else 0
        avg_inter = np.mean(inter) if inter else 0
        return avg_intra - avg_inter

    # --- Step 5: Build an initial segmentation.
    # We set boundaries as indices into the sentence list.
    # For K chunks, we want K+1 boundaries: [0, b1, b2, ..., N]
    boundaries = [0]
    for i in range(1, K):
        boundaries.append(int(round(i * N / K)))
    boundaries.append(N)
    
    # --- Step 6: Deterministic Local Search to Refine Boundaries.
    current_reward = candidate_reward(boundaries)
    improved = True
    window = 10  # search ±5 sentences around each boundary
    while improved:
        improved = False
        # Iterate over each movable boundary (skip the first and last).
        for i in range(1, len(boundaries) - 1):
            best_boundary = boundaries[i]
            best_r = candidate_reward(boundaries)
            lower_limit = boundaries[i-1] + 1
            upper_limit = boundaries[i+1] - 1
            # Restrict the search to a window around the current boundary.
            start_candidate = max(best_boundary - window, lower_limit)
            end_candidate = min(best_boundary + window, upper_limit)
            for candidate in range(start_candidate, end_candidate + 1):
                new_boundaries = boundaries.copy()
                new_boundaries[i] = candidate
                r = candidate_reward(new_boundaries)
                if r > best_r:
                    best_r = r
                    best_boundary = candidate
            if best_boundary != boundaries[i]:
                boundaries[i] = best_boundary
                current_reward = best_r
                improved = True
        # (Repeat until a full pass produces no improvement.)
    
    # --- Step 7: Reconstruct text chunks from the refined boundaries.
    chunks = []
    for i in range(len(boundaries) - 1):
        seg = " ".join(sentences[j] for j in range(boundaries[i], boundaries[i+1]))
        chunks.append( (i+1, seg) )
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
        print(f"⌚ Loading {input_file} ...")
        text = load_text(input_file)
        print(f"✅ Loaded {input_file}.")

        for max_chunk_size in [2000,3000,4000]:
            print(f"⌛ Chunking {input_file} max_chunk_size={max_chunk_size} ...")
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv(chunked_data, output_file)
            print(f"🎯 Chunking completed! Chunks saved to {output_file}.")