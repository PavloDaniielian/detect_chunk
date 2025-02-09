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

class Chunk_No_Array:
    def __init__(self, ind: int, no: int, dotval: str):
        self.ind = ind
        self.no = no
        self.dotval = dotval

def chunk_text(text : str, max_chunk_size=2000, threshold=0.4):
    """Splits text into meaningful chunks while ensuring chunk quantity matches formula."""
    sentences = split_sentences(text)
    embeddings = compute_embeddings(sentences)

    total_chunk_num = np.ceil( text.__len__() / max_chunk_size ) * 2

    chunks_no_a = []
    dota = [0]
    cci = 0
    i = 1
    while i < sentences.__len__():
        ae = 0
        for j in range(cci, i):
            ae += np.dot( embeddings[j], embeddings[i] )
        ae /= i-cci
        if dota.__len__() <= i:
            dota.append(ae)
        else:
            dota[i] = ae
        if " ".join(sentences[cci:i]).__len__() > max_chunk_size:
            k, _ = min( enumerate(dota[cci+1:i]), key=lambda x: x[1] )
            i = cci + k + 1
            ae = 0
        if ae < threshold:
            chunks_no_a.append(Chunk_No_Array(chunks_no_a.__len__(), i, dota[i]))
            cci = i
            i += 1
            if dota.__len__() <= i:
                dota.append(ae)
            else:
                dota[i] = ae
        i += 1
    
    if chunks_no_a.__len__() + 1 > total_chunk_num :
        removingN = chunks_no_a.__len__() + 1 - total_chunk_num
        chunks_no_a_sorted = sorted(chunks_no_a, key=lambda x: x["dotval"], reverse=True)
        ii = 0
        while removingN > 0 or ii<chunks_no_a.__len__():
            noNow = chunks_no_a_sorted[ii].no
            indNow = chunks_no_a_sorted[ii].ind
            noLeft = chunks_no_a[indNow-1].no if indNow>0 else 0
            noRight = chunks_no_a[indNow+1].no if indNow<chunks_no_a.__len__()-1 else sentences.__len__()
        

    chunks = []
    iPrevNo = 0
    for chunk_no in chunks_no_a:
        chunks.append( " ".join(sentences[iPrevNo:chunk_no.ind]) )
        iPrevNo = chunk_no.ind
    chunks.append( " ".join(sentences[iPrevNo:]) )

    return [(i + 1, chunk) for i, chunk in enumerate(chunks)]

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

        for max_chunk_size in [2000]:
            # Generate optimized chunks
            chunked_data = chunk_text(text, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv(chunked_data, output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")