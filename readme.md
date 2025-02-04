# Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system using Flask, FAISS, and the GEMINI API. The system extracts data from PDFs and URLs, processes the text into chunks, embeds the chunks, and stores them in a vector store. It then retrieves relevant documents based on user queries and generates responses using the GEMINI API.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MrugankRaut/RAG-System.git
    cd rag-system
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your data:
    - Place your PDF files in the [data](http://_vscodecontentref_/0) folder.
    - Create a [urls.txt](http://_vscodecontentref_/1) file with URLs to scrape text from.

2. Run the [rag_system.py](http://_vscodecontentref_/2) script to process the data and create the FAISS index:
    ```bash
    python rag_system.py
    ```

3. Start the Flask application:
    ```bash
    python app.py
    ```

4. Open your web browser and navigate to `http://localhost:5000/` to access the homepage and submit queries.

## API Endpoints

### POST /query

**Description**: Processes user queries, retrieves relevant documents, and generates a response using the GEMINI API.


- **Method**: POST
- **Content-Type**: application/json
- **Body**: JSON object containing the user query.
  ```json
  {
    "query": "Your query here"
  }
