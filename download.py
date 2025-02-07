import nltk
import multiprocessing

def download_punkt():
    nltk.download('punkt')

if __name__ == "__main__":
    process = multiprocessing.Process(target=download_punkt)
    process.start()
    process.join()
