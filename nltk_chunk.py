import nltk
import csv
import math
import glob


def load_text(file_path):
    """Loads text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_text(text, max_chunk_size=2000):
    """
    Splits text into chunks while preserving sentence boundaries and optimizing for semantic similarity.

    :param text: The input text as a string.
    :param max_chunk_size: Maximum size of each chunk (2000, 3000, or 4000).
    :return: List of (chunk_number, chunk_text).
    """
    sentences = nltk.tokenize.sent_tokenize(text)  # Tokenize text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for i in range(0, len(sentences), 3):  # Group sentences in 3s for better intra-chunk similarity
        sentence_group = " ".join(sentences[i:i+3])
        sentence_length = len(sentence_group)

        # If adding the sentence group exceeds max chunk size, finalize current chunk
        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add sentence group to current chunk
        current_chunk.append(sentence_group)
        current_length += sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [(i + 1, chunk) for i, chunk in enumerate(chunks)]

def save_chunks_to_csv(chunks, output_file):
    """Saves chunks as a CSV file with chunk numbers."""
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Index", "Chunk Content"])  # Header
        writer.writerows(chunks)

# Example Usage
if __name__ == "__main__":
    max_chunk_size = 2000  # Set chunk size for optimal processing

    for input_file in glob.glob("*.txt"):
        output_file = input_file.replace(".txt", ".csv")
        # Load text
        text_data = load_text(input_file)

        # Generate optimized chunks
        chunked_data = chunk_text(text_data, max_chunk_size)

        # Save to CSV
        save_chunks_to_csv(chunked_data, output_file)

        print(f"✅ Chunking completed! Chunks saved to {output_file}.")