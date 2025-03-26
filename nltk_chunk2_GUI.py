import nltk
import csv
import math
import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer
import numpy as np

# Ensure sentence tokenizer is available
# nltk.download('punkt')

def chunk_text(text: str, max_chunk_size: int):
    sentences = nltk.sent_tokenize(text)
    N = len(sentences)
    if N == 0:
        return []
    lengths = [len(s) for s in sentences]
    
    def chunk_size(indices):
        return sum(lengths[i] for i in indices) + (len(indices) - 1)
    
    target_chunks = int(math.ceil(len(text) / max_chunk_size) * 2)
    K = min(target_chunks, N)
    if K < 1:
        K = 1

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings / np.maximum(norms, 1e-10)

    def candidate_reward(boundaries):
        segments = []
        for i in range(len(boundaries) - 1):
            seg = list(range(boundaries[i], boundaries[i+1]))
            if chunk_size(seg) > max_chunk_size:
                return -1000  # penalty for violating size constraint
            segments.append(seg)
        check_chunks = []
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

    boundaries = [0]
    for i in range(1, K):
        boundaries.append(int(round(i * N / K)))
    boundaries.append(N)
    
    current_reward = candidate_reward(boundaries)
    improved = True
    window = 10
    while improved:
        improved = False
        for i in range(1, len(boundaries) - 1):
            best_boundary = boundaries[i]
            best_r = candidate_reward(boundaries)
            lower_limit = boundaries[i-1] + 1
            upper_limit = boundaries[i+1] - 1
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

    chunks = []
    for i in range(len(boundaries) - 1):
        seg = " ".join(sentences[j] for j in range(boundaries[i], boundaries[i+1]))
        chunks.append((i+1, seg))
    return chunks

# Define the GUI part
class ChunkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Chunking Tool")
        self.root.geometry("1200x800")  # Set window size (Width x Height)

        # Configure grid layout for responsiveness
        self.root.columnconfigure(0, weight=3)  # Input column
        self.root.columnconfigure(1, weight=1)  # Settings column
        self.root.columnconfigure(2, weight=3)  # Output column
        self.root.rowconfigure(0, weight=1)  # Expanding row

        # --- Left Column: Input Text ---
        self.frame_left = tk.Frame(root, padx=10, pady=10)
        self.frame_left.grid(row=0, column=0, sticky="nsew")
        self.frame_left.rowconfigure(1, weight=1)  # Allow text box expansion

        self.input_label = tk.Label(self.frame_left, text="Enter or Paste the Text Below:")
        self.input_label.pack(anchor="w", pady=5)

        self.input_text = tk.Text(self.frame_left, wrap=tk.WORD)
        self.input_text.pack(expand=True, fill="both")

        # --- Middle Column: Settings & Button ---
        self.frame_middle = tk.Frame(root, padx=10, pady=10)
        self.frame_middle.grid(row=0, column=1, sticky="nsew")

        self.chunk_size_label = tk.Label(self.frame_middle, text="Enter max chunk size (e.g., 2000):")
        self.chunk_size_label.pack(pady=5)

        self.chunk_size_entry = tk.Entry(self.frame_middle)
        self.chunk_size_entry.pack(pady=5)

        self.chunk_button = tk.Button(self.frame_middle, text="Start Chunking", command=self.start_chunking)
        self.chunk_button.pack(pady=20, fill="x")

        # --- Right Column: Output Text ---
        self.frame_right = tk.Frame(root, padx=10, pady=10)
        self.frame_right.grid(row=0, column=2, sticky="nsew")
        self.frame_right.rowconfigure(1, weight=1)  # Allow text box expansion

        self.output_label = tk.Label(self.frame_right, text="Chunking Output:")
        self.output_label.pack(anchor="w", pady=5)

        self.output_text = tk.Text(self.frame_right, wrap=tk.WORD)
        self.output_text.pack(expand=True, fill="both")

    def start_chunking(self):
        """Start the chunking process."""
        try:
            max_chunk_size = int(self.chunk_size_entry.get())
            input_text = self.input_text.get("1.0", tk.END).strip()
            if not input_text:
                raise ValueError("Please provide input text.")

            # Perform chunking
            self.output_text.insert(tk.END, "Chunking in progress...\n")
            self.root.update_idletasks()
            chunks = chunk_text(input_text, max_chunk_size)

            # Display output chunks in the output text box
            output_text = ""
            for chunk_num, chunk_content in chunks:
                output_text += f"Chunk {chunk_num}:\n{chunk_content}\n\n"
            
            self.output_text.delete(1.0, tk.END)  # Clear previous output
            self.output_text.insert(tk.END, output_text)

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChunkingApp(root)
    root.mainloop()
