import nltk
import numpy as np
import math
import glob
import csv
from sentence_transformers import SentenceTransformer, util

# Ensure sentence tokenizer is available
#nltk.download('punkt')

# Load a sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text: str, max_chunk_size=2000):
    """Fast and optimized chunking algorithm."""
    sentences = nltk.tokenize.sent_tokenize(text)
    N = len(sentences)
    if N == 0:
        return []
    
    total_chunk_quantity = int(np.ceil(len(text) / max_chunk_size) * 2)

    index_array = np.zeros(total_chunk_quantity+1, dtype=int)
    
    # shift to down by keeping chunk size
    cur_ind = total_chunk_quantity
    index_array[cur_ind] = N
    index = N - 1
    while index > 0:
        chunk = sentences[index]
        while True:
            index = index - 1
            chunk = sentences[index] + " " + chunk
            if index == 0 or chunk.__len__() > max_chunk_size:
                cur_ind = cur_ind - 1
                index_array[cur_ind] = index + 1
                break

    embeddings = model.encode(sentences, convert_to_numpy=True)
    embeddings3 = model.encode([sentences[i:i+3] for i in range(N-2)], convert_to_numpy=True)

    def getSimilarity(cci, cur_index):
        start_index = index_array[cci]
        total_similarity = 0

        # check in own chunk
        ae = 0
        for j in range(start_index, cur_index):
            ae += np.dot( embeddings[j], embeddings[cur_index] )
        total_similarity += 1 * ( ae / (cur_index - start_index) )

        # check neighborhood chunk
        # embedding_me = []
        # while cci > 0:
        #     cci -= 1
        #     if index_array[cci+1] - index_array[cci] > 6:
        #         embedding_me.append( (-1, embeddings3[ index_array[cci] ]) )
        #         embedding_me.append( (-1, embeddings3[ index_array[cci]+3 ]) )
        #         break
        # if len(embedding_me) <= 0:
        #         embedding_me.append( (-1, embeddings3[ N-3 ]) )
        #         embedding_me.append( (-1, embeddings3[ N-6 ]) )
        # if cur_index < N-6:
        #     embedding_me.append( (1, embeddings3[cur_index+1]) )
        #     embedding_me.append( (1, embeddings3[cur_index+4]) )
        # else:
        #     embedding_me.append( (1, embeddings3[0]) )
        #     embedding_me.append( (1, embeddings3[3]) )
        # for i in range(start_index, cur_index+1, 3):
        #     if i < N-2:
        #         embedding_me.append( (0, embeddings3[i]) )
        # intra = []
        # inter = []
        # for i in range(len(embedding_me) - 1):
        #     for j in range(i+1, len(embedding_me)):
        #         similarity = np.dot(embedding_me[i][1], embedding_me[j][1])
        #         if embedding_me[i][0] == embedding_me[j][0]:
        #             intra.append(similarity)
        #         else:
        #             inter.append(similarity)
        # total_similarity += 0.3 * ( np.mean(intra) - np.mean(inter) )
        return total_similarity

    # split entire
    cci = 0
    start_index = index_array[cci]
    similarity_min = 99999999
    start_index_next = -1
    chunk = sentences[start_index]
    i = start_index + 1
    while True:
        chunk = chunk + " " + sentences[i]
        if chunk.__len__() > max_chunk_size:
            cci = cci + 1
            start_index = start_index_next
            if cci >= total_chunk_quantity or start_index < index_array[cci]:
                break
            index_array[cci] = start_index
            similarity_min = 99999999
            chunk = sentences[start_index]
            i = start_index
        else:
            similarity = getSimilarity(cci, i)
            if similarity <= similarity_min:
                similarity_min = similarity
                start_index_next = i
        i += 1
        if i >= N:
            ccn = cci + 1
            dota = []
            for i in range(ccn):
                start_index = index_array[i]
                for j in range( 1, (index_array[i+1] if i<total_chunk_quantity-1 else N) - start_index ):
                    dota.append( ( i, j, getSimilarity(i, start_index+j) ) )
            dota = sorted( dota, key=lambda x: x[2] ) [ : total_chunk_quantity - ccn ]
            dota = sorted( dota, key=lambda x: (x[0], x[1]), reverse=True )
            for element in dota:
                cci = element[0]
                for j in range(ccn, cci+1, -1):
                    index_array[j] = index_array[j-1]
                index_array[cci+1] = index_array[cci] + element[1]
                ccn = ccn + 1
            break

    #Deterministic local search to refine index_array.
    lengths = [len(s) for s in sentences]
    def chunk_size(indices):
        return sum(lengths[i] for i in indices) + (len(indices) - 1)
    sentence_embeddings = embeddings.copy()
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)
    def candidate_reward(index_array):
        segments = []
        for i in range(len(index_array) - 1):
            seg = list(range(index_array[i], index_array[i+1]))
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
    
    current_reward = candidate_reward(index_array)
    improved = True
    window = 10  # search ±5 sentences around each boundary
    while improved:
        improved = False
        # Iterate over each movable boundary (skip the first and last).
        for i in range(1, len(index_array) - 1):
            best_boundary = index_array[i]
            best_r = candidate_reward(index_array)
            lower_limit = index_array[i-1] + 1
            upper_limit = index_array[i+1] - 1
            # Restrict the search to a window around the current boundary.
            start_candidate = max(best_boundary - window, lower_limit)
            end_candidate = min(best_boundary + window, upper_limit)
            for candidate in range(start_candidate, end_candidate + 1):
                new_index_array = index_array.copy()
                new_index_array[i] = candidate
                r = candidate_reward(new_index_array)
                if r > best_r:
                    best_r = r
                    best_boundary = candidate
            if best_boundary != index_array[i]:
                index_array[i] = best_boundary
                current_reward = best_r
                improved = True

    chunks = []
    for i in range(len(index_array)-1):
        chunks.append( ( i+1, " ".join(sentences[ index_array[i] : index_array[i+1] ]) ) )
    return chunks

def load_text(file_path):
    """Loads text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

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