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

def load_text(file_path):
    """Loads text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def split_sentences(text):
    """Uses NLTK to split text into sentences."""
    return nltk.tokenize.sent_tokenize(text)

def compute_embeddings(sentences):
    """Computes embeddings for each sentence."""
    return model.encode(sentences, convert_to_numpy=True)

def cluster_sentences(sentences, embeddings, threshold=0.7):
    """
    Groups sentences into meaningful paragraphs based on semantic similarity.
    """
    paragraphs = []
    current_paragraph = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()
        #similarity = np.dot(embeddings[i - 1], embeddings[i])

        if similarity > threshold:
            current_paragraph.append(sentences[i])
        else:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = [sentences[i]]

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

def chunk_text(text, max_chunk_size=2000, offset=0.3):
    """Splits text into meaningful chunks while ensuring chunk quantity matches formula."""
    sentences = split_sentences(text)
    embeddings = compute_embeddings(sentences)
    paragraphs = cluster_sentences(sentences, embeddings)

    total_chars = sum(len(p) for p in paragraphs)

    # Apply chunk count formula: ceil(total_characters / max_chunk_size) * 2
    estimated_chunks = math.ceil(total_chars / max_chunk_size) * 2

    # Adjust chunk sizes with offset
    adjusted_max_size = int(max_chunk_size * (1 + offset))

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if current_length + paragraph_length > adjusted_max_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(paragraph)
        current_length += paragraph_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Ensure the number of chunks matches the formula
    while len(chunks) < estimated_chunks:
        chunks.append(chunks[-1])  # Duplicate last chunk if needed

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
        text_data = load_text(input_file)

        for max_chunk_size in [2000,3000,4000]:
            # Generate optimized chunks
            chunked_data = chunk_text(text_data, max_chunk_size)
            # Save to CSV
            output_file = f"{output_file_prefix}__{max_chunk_size}.csv"
            save_chunks_to_csv(chunked_data, output_file)
            print(f"✅ Chunking completed! Chunks saved to {output_file}.")