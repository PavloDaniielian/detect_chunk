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

def split_sentences(text):
    """Uses NLTK to split text into sentences."""
    return nltk.tokenize.sent_tokenize(text)

def compute_embeddings(sentences):
    """Computes embeddings for each sentence."""
    return model.encode(sentences, convert_to_numpy=True)

def chunk_text(text: str, max_chunk_size=2000):
    """Fast and optimized chunking algorithm."""
    sentences = split_sentences(text)
    N = len(sentences)
    if N == 0:
        return []

    embeddings = compute_embeddings(sentences)
    total_chunk_num = int(np.ceil(len(text) / max_chunk_size) * 2)

    index_array = np.zeros(total_chunk_num, dtype=int)

    cur_ind = total_chunk_num
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

    cci = 0
    start_index = index_array[cci]
    dota = []
    chunk = sentences[start_index]
    i = start_index + 1
    while i < N:
        chunk = chunk + " " + sentences[i]
        if chunk.__len__() > max_chunk_size:
            k, _ = min( enumerate(dota), key=lambda x: x[1] )
            cci = cci + 1
            start_index = start_index + 1 + k
            if cci >= total_chunk_num or start_index < index_array[cci]:
                break
            index_array[cci] = start_index
            dota = []
            chunk = sentences[start_index]
            i = start_index + 1
        else:
            ae = 0
            for j in range(start_index+1, i+1):
                ae += np.dot( embeddings[j], embeddings[start_index] )
            dota.append( ae / (i - start_index) )
        i += 1

    chunks = []
    for i in range(0, len(index_array)-1):
        chunks.append( ( i+1, " ".join(sentences[ index_array[i] : index_array[i+1] ]) ) )
    i = len(index_array)-1
    chunks.append( (i+1, " ".join(sentences[ index_array[i] : N ]) ))
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
        text = load_text(input_file)

        for max_chunk_size in [2000,3000,4000]:
            print(f"Chunking {input_file} max_chunk_size={max_chunk_size} ...")
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv(chunked_data, output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")