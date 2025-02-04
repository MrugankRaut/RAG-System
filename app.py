from flask import Flask, request, jsonify
import pickle
import faiss
import json
import torch
from transformers import AutoTokenizer, AutoModel
import requests
from flask import render_template

app = Flask(__name__)

# Load the FAISS index
index = faiss.read_index('faiss_index.bin')

# Load the chunks
with open('all_chunks.pkl', 'rb') as f:
    all_chunks = pickle.load(f)


GEMINI_API_KEY = "YOUR API KEY"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def retrieve_relevant_documents(query, index, all_chunks, top_k=5):
    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding, top_k)
    return [all_chunks[i] for i in indices[0]]

def generate_response_from_gemini(query, context, api_key):
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
    
    return response.json()

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    relevant_documents = retrieve_relevant_documents(user_query, index, all_chunks)
    context = " ".join(relevant_documents)
    response = generate_response_from_gemini(user_query, context, GEMINI_API_KEY)
    text_content = response["candidates"][0]["content"]["parts"][0]["text"]
    return jsonify({"response": text_content})

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)