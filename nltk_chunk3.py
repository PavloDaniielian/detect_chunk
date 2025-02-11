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
import networkx as nx

# Ensure sentence tokenizer is available
#nltk.download('punkt')

def chunk_text(text: str, max_chunk_size: int):
    # --- Step 1: Tokenization & Sentence Embeddings ---
    #nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    N = len(sentences)
    if N == 0:
        return []

    lengths = [len(s) for s in sentences]

    def chunk_size(indices):
        return sum(lengths[i] for i in indices) + (len(indices) - 1)

    # --- Step 2: Compute Sentence Embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)

    # --- Step 3: Identify Initial Chunk Boundaries (Using Semantic Drop-Off) ---
    target_chunks = int(math.ceil(len(text) / max_chunk_size) * 2)
    K = min(target_chunks, N)

    # Compute cosine similarity drop-offs to detect topic shifts
    similarity_drops = [np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]) for i in range(N - 1)]
    
    # Find K-1 weakest links (lowest similarity) as breakpoints
    breakpoints = sorted(range(len(similarity_drops)), key=lambda x: similarity_drops[x])[:K-1]
    boundaries = sorted([bp + 1 for bp in breakpoints])

    # Ensure boundaries are valid and fit max_chunk_size
    boundaries = [0] + boundaries + [N]

    # --- Step 4: Dynamic Programming Optimization ---
    INF = float('-inf')
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    backtrack = [[-1] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0  # Base case

    for k in range(1, K + 1):
        for i in range(1, N + 1):
            for j in range(k - 1, i):
                if chunk_size(range(j, i)) > max_chunk_size:
                    continue
                intra_chunk_score = np.mean(np.dot(sentence_embeddings[j:i], sentence_embeddings[j:i].T))
                boundary_penalty = -np.dot(sentence_embeddings[j - 1], sentence_embeddings[j]) if j > 0 else 0
                candidate_score = dp[k - 1][j] + intra_chunk_score + boundary_penalty
                if candidate_score > dp[k][i]:
                    dp[k][i] = candidate_score
                    backtrack[k][i] = j

    # --- Step 5: Extract Best Segmentation ---
    best_boundaries = [0] * (K + 1)
    best_boundaries[K] = N
    i, k = N, K
    while k > 0:
        j = backtrack[k][i]
        if j == -1:
            # Fallback to greedy segmentation if DP failed
            step = max(1, N // K)
            best_boundaries = [min(N, i * step) for i in range(K)]
            best_boundaries.append(N)
            break
        best_boundaries[k - 1] = j
        i, k = j, k - 1

    # --- Step 6: Compute Final Reward ---
    def compute_final_reward(boundaries, embeddings, max_size):
        segments = []
        for i in range(len(boundaries) - 1):
            seg = list(range(boundaries[i], boundaries[i + 1]))
            if chunk_size(seg) > max_size:
                return -1000  # Heavy penalty
            segments.append(seg)

        check_chunks = []
        for seg_index, seg in enumerate(segments):
            for j in range(0, len(seg), 3):
                group = seg[j:j+3]
                if len(group) == 0:
                    continue
                avg_emb = np.mean(embeddings[group], axis=0)
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

    final_reward = compute_final_reward(best_boundaries, sentence_embeddings, max_chunk_size)

    # --- Step 7: Construct and Return Final Segmented Text Chunks ---
    chunks = []
    for i in range(len(best_boundaries) - 1):
        start, end = best_boundaries[i], best_boundaries[i + 1]
        if start < end:  # Prevent empty chunks
            chunk = " ".join(sentences[j] for j in range(start, end))
            chunks.append(chunk)

    return chunks, final_reward

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
            chunked_data, rewards = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv([(i + 1, chunk) for i, chunk in enumerate(chunked_data)], output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}. Score is {rewards}")