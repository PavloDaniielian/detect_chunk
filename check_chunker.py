import numpy as np
import nltk
import asyncio
from sentence_transformers import SentenceTransformer
import glob

# Ensure NLTK tokenizer is available
#nltk.download('punkt')

class CheckChunk:
    def __init__(self, source_chunk: str, text: str):
        self.source_chunk = source_chunk
        self.text = text


async def chunk_checker(chunks):
    check_chunks = []
    intrachunk_similarities = []
    interchunk_similarities = []

    for i, chunk in enumerate(chunks):
        sentences = nltk.sent_tokenize(chunk)
        for j in range(0, len(sentences), 3):
            text = " ".join(sentences[j: j + 3])
            check_chunks.append(CheckChunk(i, text))

    # ✅ Replace OpenAI with Hugging Face Sentence Transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Open-source model
    embeddings = model.encode([chunk.text for chunk in check_chunks], convert_to_numpy=True)

    # Calculate intra-chunk and inter-chunk similarities
    for i in range(len(check_chunks) - 1):
        j = i + 1
        while j < len(check_chunks):
            similarity = np.dot(embeddings[i], embeddings[j])
            if check_chunks[i].source_chunk == check_chunks[j].source_chunk:
                intrachunk_similarities.append(similarity)
            else:
                interchunk_similarities.append(similarity)
            j += 1

    # Calculate the embedding reward
    reward = (
        np.mean(intrachunk_similarities) if intrachunk_similarities else 0
    ) - (np.mean(interchunk_similarities) if interchunk_similarities else 0)

    return reward

if __name__ == "__main__":
    import csv

    # Set event loop policy for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    for csv_file in glob.glob("*.csv"):
        chunks = []

        # Read chunks from CSV
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                chunks.append(row['Chunk Content'])

        # Run the async function properly
        reward = asyncio.run(chunk_checker(chunks))
        print(f"Embedding Reward {csv_file}: {reward}")