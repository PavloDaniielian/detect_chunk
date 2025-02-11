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
    
    # Initialize model and compute sentence embeddings once.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    # Normalize embeddings so that dot products equal cosine similarity.
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)
    
    def chunk_text_length(indices):
        """Estimate the length (in characters) of a chunk built from the given sentence indices."""
        return sum(len(sentences[i]) for i in indices) + max(0, len(indices) - 1)
    
    # -------------------------------------------------------
    # Step 1: Initial segmentation based solely on character count.
    chunks_indices = []  # Each element is a list of sentence indices for that chunk.
    current_chunk = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sent_len = len(sentence)
        # If adding this sentence exceeds max_chunk_size (and current_chunk is non-empty), finish the chunk.
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
    
    # -------------------------------------------------------
    # Step 2: Iterative splitting using cumulative sums of precomputed embeddings.
    def compute_cumsum(indices):
        return np.cumsum(sentence_embeddings[indices], axis=0)
    
    while len(chunks_indices) < target_chunk_count:
        best_candidate_value = np.inf
        best_candidate = None  # Tuple: (chunk_index, split_position)
        
        for chunk_idx, indices in enumerate(chunks_indices):
            if len(indices) < 2:
                continue  # Cannot split a chunk with only one sentence.
            cumsum = compute_cumsum(indices)
            N = len(indices)
            # Try every possible boundary within the chunk.
            for j in range(1, N):
                left_indices = indices[:j]
                right_indices = indices[j:]
                # Check that both new chunks obey the size constraint.
                if chunk_text_length(left_indices) > max_chunk_size or chunk_text_length(right_indices) > max_chunk_size:
                    continue
                left_avg = cumsum[j-1] / j
                right_avg = (cumsum[-1] - cumsum[j-1]) / (N - j)
                candidate_value = np.dot(left_avg, right_avg)  # Lower value => lower similarity.
                if candidate_value < best_candidate_value:
                    best_candidate_value = candidate_value
                    best_candidate = (chunk_idx, j)
        
        if best_candidate is None:
            break  # No valid split found.
        
        chunk_idx, split_pos = best_candidate
        indices = chunks_indices.pop(chunk_idx)
        left_indices = indices[:split_pos]
        right_indices = indices[split_pos:]
        chunks_indices.insert(chunk_idx, left_indices)
        chunks_indices.insert(chunk_idx + 1, right_indices)
    
    # -------------------------------------------------------
    # Step 3: First boundary refinement stage.
    # For each adjacent pair, try moving a sentence from one side to the other if it lowers the dot product
    # between the average embeddings.
    improved = True
    while improved:
        improved = False
        for i in range(len(chunks_indices) - 1):
            left = chunks_indices[i]
            right = chunks_indices[i+1]
            
            left_avg = np.mean(sentence_embeddings[left], axis=0)
            right_avg = np.mean(sentence_embeddings[right], axis=0)
            current_dot = np.dot(left_avg, right_avg)
            best_dot = current_dot
            best_move = None
            
            # Option A: Move first sentence from right to left.
            if len(right) > 1:
                candidate_left = left + [right[0]]
                candidate_right = right[1:]
                if (chunk_text_length(candidate_left) <= max_chunk_size and 
                    chunk_text_length(candidate_right) <= max_chunk_size):
                    new_left_avg = np.mean(sentence_embeddings[candidate_left], axis=0)
                    new_right_avg = np.mean(sentence_embeddings[candidate_right], axis=0)
                    new_dot = np.dot(new_left_avg, new_right_avg)
                    if new_dot < best_dot:
                        best_dot = new_dot
                        best_move = 'move_from_right'
            
            # Option B: Move last sentence from left to right.
            if len(left) > 1:
                candidate_left = left[:-1]
                candidate_right = [left[-1]] + right
                if (chunk_text_length(candidate_left) <= max_chunk_size and 
                    chunk_text_length(candidate_right) <= max_chunk_size):
                    new_left_avg = np.mean(sentence_embeddings[candidate_left], axis=0)
                    new_right_avg = np.mean(sentence_embeddings[candidate_right], axis=0)
                    new_dot = np.dot(new_left_avg, new_right_avg)
                    if new_dot < best_dot:
                        best_dot = new_dot
                        best_move = 'move_from_left'
            
            if best_move == 'move_from_right':
                moved_sentence = right.pop(0)
                left.append(moved_sentence)
                chunks_indices[i] = left
                chunks_indices[i+1] = right
                improved = True
            elif best_move == 'move_from_left':
                moved_sentence = left.pop(-1)
                right.insert(0, moved_sentence)
                chunks_indices[i] = left
                chunks_indices[i+1] = right
                improved = True
    
    # -------------------------------------------------------
    # Step 4: Second (aggressive) boundary optimization via local objective.
    # Here we define a local objective for a boundary between two chunks.
    def local_objective(chunk):
        # Use average cosine similarity between adjacent sentences as a proxy for intra-chunk coherence.
        if len(chunk) < 2:
            return 0
        sims = [np.dot(sentence_embeddings[chunk[j]], sentence_embeddings[chunk[j+1]]) for j in range(len(chunk)-1)]
        return np.mean(sims)
    
    def boundary_local_score(left, right):
        # Local score is the sum of intra-chunk scores minus the inter-chunk similarity
        # (here using the dot product between the last sentence of left and first sentence of right).
        left_obj = local_objective(left)
        right_obj = local_objective(right)
        inter_sim = np.dot(sentence_embeddings[left[-1]], sentence_embeddings[right[0]])
        return left_obj + right_obj - inter_sim
    
    improvement = True
    while improvement:
        improvement = False
        for i in range(len(chunks_indices) - 1):
            left = chunks_indices[i]
            right = chunks_indices[i+1]
            current_score = boundary_local_score(left, right)
            best_score = current_score
            best_move = None
            # Option A: Try moving the first sentence of right to left.
            if len(right) > 1:
                candidate_left = left + [right[0]]
                candidate_right = right[1:]
                if (chunk_text_length(candidate_left) <= max_chunk_size and 
                    chunk_text_length(candidate_right) <= max_chunk_size):
                    candidate_score = boundary_local_score(candidate_left, candidate_right)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_move = 'move_from_right'
            # Option B: Try moving the last sentence of left to right.
            if len(left) > 1:
                candidate_left = left[:-1]
                candidate_right = [left[-1]] + right
                if (chunk_text_length(candidate_left) <= max_chunk_size and 
                    chunk_text_length(candidate_right) <= max_chunk_size):
                    candidate_score = boundary_local_score(candidate_left, candidate_right)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_move = 'move_from_left'
            if best_move == 'move_from_right':
                moved_sentence = right.pop(0)
                left.append(moved_sentence)
                chunks_indices[i] = left
                chunks_indices[i+1] = right
                improvement = True
            elif best_move == 'move_from_left':
                moved_sentence = left.pop(-1)
                right.insert(0, moved_sentence)
                chunks_indices[i] = left
                chunks_indices[i+1] = right
                improvement = True
    
    # -------------------------------------------------------
    # Finally, reconstruct the text chunks from sentence indices.
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
            print(f"Chunking {input_file} max_chunk_size={max_chunk_size} ...")
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv([(i + 1, chunk) for i, chunk in enumerate(chunked_data)], output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")