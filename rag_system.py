import os
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pickle

class DataLoader:
    """Class to handle loading and preprocessing of data from PDFs and URLs."""
    
    def __init__(self, pdf_folder, url_file):
        """
        Initialize the DataLoader with paths to the PDF folder and URL file.
        
        Args:
            pdf_folder (str): Path to the folder containing PDF files.
            url_file (str): Path to the text file containing URLs.
        """
        self.pdf_folder = pdf_folder
        self.url_file = url_file
    
    def load_pdfs(self):
        """Load and extract text from all PDFs in the specified folder."""
        pdf_texts = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder, filename)
                pdf_texts.append(self._extract_text_from_pdf(pdf_path))
        return pdf_texts
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def load_urls(self):
        """Load and scrape text from all URLs in the specified file."""
        with open(self.url_file, 'r') as file:
            urls = file.readlines()
        url_texts = [self._scrape_data_from_url(url.strip()) for url in urls]
        return url_texts
    
    def _scrape_data_from_url(self, url):
        """Scrape text data from a single URL."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()

class TextProcessor:
    """Class to handle text processing tasks such as chunking and embedding."""
    
    def __init__(self, chunk_size=1000, overlap=50):
        """
        Initialize the TextProcessor with chunk size and overlap.
        
        Args:
            chunk_size (int): Size of each text chunk.
            overlap (int): Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def split_text_into_chunks(self, text):
        """Split text into chunks of specified size with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
    
    def embed_text(self, text):
        """Embed text using a pre-trained model."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

class VectorStore:
    """Class to handle storage and retrieval of embedded text chunks."""
    
    def __init__(self, dimension):
        """
        Initialize the VectorStore with the dimension of embeddings.
        
        Args:
            dimension (int): Dimension of the embeddings.
        """
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(self, embeddings):
        """Add embeddings to the vector store."""
        self.index.add(embeddings)
    
    def search(self, query_embedding, top_k=5):
        """Search for the top_k most similar embeddings to the query."""
        _, indices = self.index.search(query_embedding, top_k)
        return indices

def save_data(index, chunks, index_file='faiss_index.bin', chunks_file='all_chunks.pkl'):
    """Save the FAISS index and text chunks to disk."""
    faiss.write_index(index, index_file)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)

def load_data(index_file='faiss_index.bin', chunks_file='all_chunks.pkl'):
    """Load the FAISS index and text chunks from disk."""
    index = faiss.read_index(index_file)
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

class Retriever:
    """Class to handle retrieval of relevant documents based on user query."""
    
    def __init__(self, index, all_chunks):
        """
        Initialize the Retriever with the FAISS index and text chunks.
        
        Args:
            index (faiss.Index): The FAISS index.
            all_chunks (list): The list of text chunks.
        """
        self.index = index
        self.all_chunks = all_chunks
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def embed_text(self, text):
        """Embed text using a pre-trained model."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    def retrieve_relevant_documents(self, query, top_k=5):
        """Retrieve the top_k most relevant documents for the given query."""
        query_embedding = self.embed_text(query)
        indices = self.index.search(query_embedding, top_k)
        return [self.all_chunks[i] for i in indices[0].tolist()]

def generate_response_from_gemini(query, context, api_key):
    """
    Generate a response from the GEMINI model using the provided query and context.

    Args:
        query (str): The user query.
        context (str): The context from the relevant documents.
        api_key (str): The GEMINI API key.

    Returns:
        dict: The response from the GEMINI model.
    """
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": f"Context: {context}\n\nQuery: {query}"}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"Error from GEMINI API: {response.text}")
        return {"error": "Error from GEMINI API"}
    
    return response.json()

def main():
    """Main execution function to load data, process it, and save the FAISS index and chunks."""
    pdf_folder = 'data'
    url_file = 'urls.txt'

    # Initialize data loader and load data
    data_loader = DataLoader(pdf_folder, url_file)
    pdf_texts = data_loader.load_pdfs()
    url_texts = data_loader.load_urls()

    # Initialize text processor and process data
    text_processor = TextProcessor()
    all_chunks = []
    for text in pdf_texts + url_texts:
        chunks = text_processor.split_text_into_chunks(text)
        all_chunks.extend(chunks)

    embedded_chunks = np.vstack([text_processor.embed_text(chunk) for chunk in all_chunks])

    # Initialize vector store and add embeddings
    vector_store = VectorStore(dimension=embedded_chunks.shape[1])
    vector_store.add_embeddings(embedded_chunks)

    # Save the index and chunks to disk
    save_data(vector_store.index, all_chunks)

if __name__ == '__main__':
    main()