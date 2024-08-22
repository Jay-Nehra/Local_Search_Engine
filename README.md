# Search Engine Application

## Overview
This application is a simple search engine built with Streamlit, utilizing TF-IDF and cosine similarity for searching through a collection of documents. The app allows users to upload JSON documents, configure a search index, and perform searches with optional keyword filtering and boosting factors for specific fields.

## Features
- **Upload Documents**: Users can upload new JSON documents to the knowledge base.
- **Select Existing Documents**: Users can select from pre-existing JSON documents stored in the knowledge base.
- **Index Configuration**: Users can select text and keyword fields from the document, apply boosting factors to text fields, and configure the search index.
- **Search Functionality**: Users can perform searches on the indexed documents with optional keyword filtering.
- **Persistence**: The application supports saving and loading the search index to/from a pickle file for faster subsequent searches.

## Getting Started

### Prerequisites
- Docker (if you want to run the app inside a Docker container)
- Python 3.10 or higher (if you want to run the app locally without Docker)

### Running the Application with Docker

1. **Build the Docker Image**:
   Navigate to the project directory and build the Docker image.

   ```bash
   docker build -t search-app .
   ```

2. **Run the Docker Container**:
   Start the container and map the port to your local machine.

   ```bash
   docker run -p 8501:8501 search-app
   ```

3. **Access the Application**:
   Open your web browser and go to `http://localhost:8501`.

### Running the Application Locally

1. **Install Dependencies**:
   Install the required Python packages listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   Start the Streamlit application.

   ```bash
   streamlit run app.py
   ```

3. **Access the Application**:
   Open your web browser and go to `http://localhost:8501`.

## How to Use

### 1. Upload or Select a Document
- Choose to upload a new JSON document or select an existing document from the knowledge base.

### 2. Configure Index
- Select the text fields and keyword fields for indexing.
- Optionally, apply boosting factors to specific text fields to prioritize them during searches.
- Configure the index by clicking the "Configure Index" button.

### 3. Perform a Search
- Enter your search query in the provided text box.
- Optionally, filter results based on keyword fields.
- Click "Search" to see the results.

### 4. Persistent Indexing
- If the index is configured for a document, it will be saved to a pickle file for faster future searches.
- You can load the existing index directly if it’s already available.

## Use Cases
- **Knowledge Base Search**: Use this application to search through a collection of FAQ documents or other knowledge bases.
- **Course Material Search**: Index and search through course materials, filtering results by course or section.
- **Document Repository**: Quickly search through large repositories of documents by indexing specific fields and applying custom boosting factors.
