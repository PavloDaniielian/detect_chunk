import nltk
import glob
import csv
import math
from sentence_transformers import SentenceTransformer, util
import numpy as np

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

def cluster_sentences(sentences, embeddings, threshold=0.3):
    """
    Clusters sentences into meaningful paragraphs based on semantic similarity.
    """
    paragraphs = []
    current_paragraph = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()

        if similarity > threshold:
            # If similar, add to the current paragraph
            current_paragraph.append(sentences[i])
        else:
            # Otherwise, start a new paragraph
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = [sentences[i]]

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

def chunk_text(text, max_chunk_size=2000, threshold=0.3):
    """Splits text into meaningful paragraphs using semantic similarity."""
    sentences = split_sentences(text)
    embeddings = compute_embeddings(sentences)

    sentence1 = " ".join(sentences[0: 3])
    sentence2 = " ".join(sentences[3: 5])
    sentence3 = " ".join(sentences[5: 6])
    e1 = model.encode(sentence1, convert_to_numpy=True)
    e2 = model.encode(sentence2, convert_to_numpy=True)
    e3 = model.encode(sentence3, convert_to_numpy=True)
    
    a = np.dot(e1, e2)
    a = np.dot(e1, e3)
    a = np.dot(e2, e3)

    a = np.dot(embeddings[1], embeddings[0])
    a = np.dot(embeddings[2], embeddings[0])
    a = np.dot(embeddings[2], embeddings[1])
    a = np.dot(embeddings[3], embeddings[0])
    a = np.dot(embeddings[3], embeddings[1])
    a = np.dot(embeddings[3], embeddings[2])
    a = np.dot(embeddings[4], embeddings[0])
    a = np.dot(embeddings[4], embeddings[1])
    a = np.dot(embeddings[4], embeddings[2])
    a = np.dot(embeddings[4], embeddings[3])
    a = np.dot(embeddings[5], embeddings[0])
    a = np.dot(embeddings[5], embeddings[1])
    a = np.dot(embeddings[5], embeddings[2])
    a = np.dot(embeddings[5], embeddings[3])
    a = np.dot(embeddings[5], embeddings[4])
    a = np.dot(embeddings[6], embeddings[0])
    a = np.dot(embeddings[6], embeddings[1])
    a = np.dot(embeddings[6], embeddings[2])
    a = np.dot(embeddings[6], embeddings[3])
    a = np.dot(embeddings[6], embeddings[4])
    a = np.dot(embeddings[6], embeddings[5])

    return [(i + 1, sentence, np.dot(embeddings[i], embeddings[i+1]) if i < len(sentences) - 1 else 0, util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() if i < len(sentences) - 1 else 0) 
        for i, sentence in enumerate(sentences)]

def save_chunks_to_csv(chunks, output_file):
    """Saves chunks as a CSV file with chunk numbers."""
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Index", "Chunk Content", "dot", "pytorch_cos_sim"])
        writer.writerows(chunks)

# Process all .txt files in the directory
if __name__ == "__main__":
    max_chunk_size = 2000  # Set chunk size for optimal processing
    threshold = 0.7

    for input_file in glob.glob("*.txt"):
        output_file_prefix = input_file.replace(".txt", "")
        # Load text
        text_data = load_text(input_file)

        # Generate optimized chunks
        chunked_data = chunk_text(text_data, max_chunk_size, threshold)
        # Save to CSV
        output_file = f"{output_file_prefix}__dot.csv"
        save_chunks_to_csv(chunked_data, output_file)
        print(f"✅ Chunking completed! Chunks saved to {output_file}.")