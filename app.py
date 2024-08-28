# import streamlit as st
# import os
# import json
# import pandas as pd
# from flatten_json import flatten
# from Engine.engine import SearchIndex

# DATA_DIR = 'Knowledge_Base'
# PICKLE_DIR = 'VectorStore'

# def file_path(filename):
#     """Construct a file path for the given filename."""
#     return os.path.join(DATA_DIR, filename)

# def pickle_path(filename):
#     """Construct a pickle path for the given filename."""
#     base_filename = os.path.splitext(filename)[0]
#     return os.path.join(PICKLE_DIR, f"{base_filename}.pkl")

# def load_documents(file_path):
#     """Load documents from a JSON file."""
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return json.load(file)

# def save_document(file, content):
#     """Save the document content to a file."""
#     os.makedirs(DATA_DIR, exist_ok=True)
#     with open(file, 'w', encoding='utf-8') as f:
#         json.dump(content, f, ensure_ascii=False, indent=4)

# def is_json(filename):
#     """Check if the file is a JSON file based on its extension."""
#     return filename.lower().endswith('.json')

# def load_existing_documents():
#     """Load existing documents from the knowledge base."""
#     files = [f for f in os.listdir(DATA_DIR) if is_json(f)]
#     return files

# def load_existing_pickles():
#     """Load existing pickle files from the vector store."""
#     pickles = [f for f in os.listdir(PICKLE_DIR) if f.endswith('.pkl')]
#     return pickles

# def load_or_create_index(file_name, text_fields, keyword_fields, boosting_factors):
#     """Load an existing index from a pickle file or create a new one if not available."""
#     pickle_file = pickle_path(file_name)
#     if os.path.exists(pickle_file):
#         st.info(f"Loading index from pickle file: {pickle_file}")
#         search_index = SearchIndex(text_fields, keyword_fields, boosting_factors=boosting_factors)
#         search_index.load_from_pickle(pickle_file)
#     else:
#         st.info("No pickle file found. Indexing the document.")
#         documents = load_documents(file_path(file_name))
#         search_index = SearchIndex(text_fields, keyword_fields, boosting_factors=boosting_factors)
#         search_index.fit(documents, pickle_file=pickle_file)
    
#     st.session_state['search_index'] = search_index
#     st.session_state['keyword_fields'] = keyword_fields
#     st.session_state['index_configured'] = True

# def display_document_fields(file_name):
#     """Display fields from a selected JSON document for indexing configuration."""
#     path = file_path(file_name)
#     with open(path, 'r', encoding='utf-8') as f:
#         content = json.load(f)
#     if isinstance(content, list) and all(isinstance(item, dict) for item in content):
#         flattened_documents = [flatten(item) for item in content]
#         fields = list(flattened_documents[0].keys())

#         col1, col2 = st.columns(2)
#         with col1:
#             text_fields = st.multiselect('Select Text Fields for Indexing', options=fields)
#         with col2:
#             keyword_fields = st.multiselect('Select Keyword Fields for Indexing', options=fields)

#         st.subheader("Set Boost Factors for Text Fields")
#         boost_factors = {}
#         col3, col4 = st.columns(2)
#         for idx, field in enumerate(text_fields):
#             with col3 if idx % 2 == 0 else col4:
#                 boost_factors[field] = st.number_input(f"Boost for {field}", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

#         if st.button('Configure Index'):
#             load_or_create_index(file_name, text_fields, keyword_fields, boost_factors)
#             st.success("Indexing configuration saved and index built! Ready to search.")
#             st.session_state['index_ready'] = True

# def search_interface():
#     """Display the search interface and handle keyword filtering."""
#     st.subheader("Search Documents")
#     query = st.text_input("Enter your search query:")
#     if query:
#         filter_dict = {}
#         if 'keyword_fields' in st.session_state:
#             for field in st.session_state['keyword_fields']:
#                 options = pd.Series([doc[field] for doc in st.session_state['search_index'].documents if field in doc]).unique()
#                 selected_option = st.selectbox(f"Filter by {field}", [''] + list(options))
#                 if selected_option:
#                     filter_dict[field] = selected_option
        
#         if st.button("Search"):
#             results = st.session_state['search_index'].search(query, filter_dict=filter_dict)
#             if results:
#                 st.write("Search Results:", results)
#             else:
#                 st.write("No results found.")

# st.title('Search Engine Using TF-IDF Vectors')

# option = st.radio("Choose an action:", ("Upload New Document", "Select from Existing Documents", "Use Existing Index"))

# if option == "Upload New Document":
#     uploaded_file = st.file_uploader("Choose a file (JSON format)", type='json')
#     if uploaded_file is not None and is_json(uploaded_file.name):
#         document_content = json.load(uploaded_file)
#         file_name = uploaded_file.name
#         save_document(file_path(file_name), document_content)
#         st.success("Document uploaded and saved successfully.")
#         display_document_fields(file_name)

# elif option == "Select from Existing Documents":
#     existing_files = load_existing_documents()
#     file_name = st.selectbox("Select a document:", existing_files)
#     if file_name:
#         display_document_fields(file_name)

# elif option == "Use Existing Index":
#     existing_pickles = load_existing_pickles()
#     selected_pickle = st.selectbox("Select an existing index:", existing_pickles)
#     if selected_pickle:
#         pickle_file = os.path.join(PICKLE_DIR, selected_pickle)
#         search_index = SearchIndex([], [])  
#         search_index.load_from_pickle(pickle_file)
#         st.session_state['search_index'] = search_index
#         st.session_state['keyword_fields'] = search_index.keyword_fields
#         st.session_state['index_configured'] = True
#         st.success(f"Index loaded from {selected_pickle}. Ready to search.")

# if 'index_configured' in st.session_state and st.session_state['index_configured']:
#     search_interface()

import streamlit as st
import os
import json
import pandas as pd
from flatten_json import flatten
from Engine.engine import SearchIndex
from Engine.elasticsearch_engine import ElasticsearchEngine  # Import the Elasticsearch engine

DATA_DIR = 'Knowledge_Base'
PICKLE_DIR = 'VectorStore'

# Utility function to generate a dynamic index name based on the file name or other context
def generate_index_name(file_name):
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_index"

# Function to initialize the Elasticsearch engine with a dynamic index name
def initialize_elasticsearch_engine(file_name):
    es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    index_name = generate_index_name(file_name)
    return ElasticsearchEngine(host=es_host, index_name=index_name)

def file_path(filename):
    return os.path.join(DATA_DIR, filename)

def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_document(file, content):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

def is_json(filename):
    """Check if the file is a JSON file based on its extension."""
    return filename.lower().endswith('.json')

def load_existing_documents():
    """Load existing documents from the knowledge base."""
    files = [f for f in os.listdir(DATA_DIR) if is_json(f)]
    return files

def load_existing_pickles():
    """Load existing pickle files from the vector store."""
    pickles = [f for f in os.listdir(PICKLE_DIR) if f.endswith('.pkl')]
    return pickles

def load_or_create_index(file_name, text_fields, keyword_fields, boosting_factors):
    """Load an existing index from a pickle file or create a new one if not available."""
    pickle_file = pickle_path(file_name)
    if os.path.exists(pickle_file):
        st.info(f"Loading index from pickle file: {pickle_file}")
        search_index = SearchIndex(text_fields, keyword_fields, boosting_factors=boosting_factors)
        search_index.load_from_pickle(pickle_file)
    else:
        st.info("No pickle file found. Indexing the document.")
        documents = load_documents(file_path(file_name))
        search_index = SearchIndex(text_fields, keyword_fields, boosting_factors=boosting_factors)
        search_index.fit(documents, pickle_file=pickle_file)
    
    st.session_state['search_index'] = search_index
    st.session_state['keyword_fields'] = keyword_fields
    st.session_state['index_configured'] = True

def display_document_fields(file_name, use_elasticsearch):
    path = file_path(file_name)
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    if isinstance(content, list) and all(isinstance(item, dict) for item in content):
        flattened_documents = [flatten(item) for item in content]
        fields = list(flattened_documents[0].keys())

        col1, col2 = st.columns(2)
        with col1:
            text_fields = st.multiselect('Select Text Fields for Indexing', options=fields)
        with col2:
            keyword_fields = st.multiselect('Select Keyword Fields for Indexing', options=fields)

        st.subheader("Set Boost Factors for Text Fields")
        boost_factors = {}
        col3, col4 = st.columns(2)
        for idx, field in enumerate(text_fields):
            with col3 if idx % 2 == 0 else col4:
                boost_factors[field] = st.number_input(f"Boost for {field}", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        if st.button('Configure Index'):
            if use_elasticsearch:
                es_engine = initialize_elasticsearch_engine(file_name)
                if es_engine.index_exists():
                    st.info("Index already exists. Skipping indexing.")
                else:
                    documents = load_documents(file_path(file_name))
                    es_engine.index_documents(documents)
                    st.success(f"Documents indexed in Elasticsearch! Index name: {es_engine.index_name}")
            else:
                load_or_create_index(file_name, text_fields, keyword_fields, boost_factors)
            st.session_state['index_ready'] = True

def search_interface(use_elasticsearch, file_name):
    st.subheader("Search Documents")
    query = st.text_input("Enter your search query:")
    if query:
        if use_elasticsearch:
            es_engine = initialize_elasticsearch_engine(file_name)
            fields = st.multiselect('Select Fields to Search', options=["question", "answer"], default=["question", "answer"])
            filter_dict = {}  # Define filters if necessary
            results = es_engine.search_documents(query, fields, filter_dict)
        else:
            filter_dict = {}
            if 'keyword_fields' in st.session_state:
                for field in st.session_state['keyword_fields']:
                    options = pd.Series([doc[field] for doc in st.session_state['search_index'].documents if field in doc]).unique()
                    selected_option = st.selectbox(f"Filter by {field}", [''] + list(options))
                    if selected_option:
                        filter_dict[field] = selected_option
            results = st.session_state['search_index'].search(query, filter_dict=filter_dict)

        if results:
            st.write("Search Results:", results)
        else:
            st.write("No results found.")

st.title('Search Engine Using TF-IDF Vectors or Elasticsearch')

# Add a radio button for selecting the search engine
search_option = st.radio("Choose a search engine:", ("Custom TF-IDF Search Engine", "Elasticsearch"))
use_elasticsearch = search_option == "Elasticsearch"

option = st.radio("Choose an action:", ("Upload New Document", "Select from Existing Documents", "Use Existing Index"))

if option == "Upload New Document":
    uploaded_file = st.file_uploader("Choose a file (JSON format)", type='json')
    if uploaded_file is not None and is_json(uploaded_file.name):
        document_content = json.load(uploaded_file)
        file_name = uploaded_file.name
        save_document(file_path(file_name), document_content)
        st.success("Document uploaded and saved successfully.")
        display_document_fields(file_name, use_elasticsearch)

elif option == "Select from Existing Documents":
    existing_files = load_existing_documents()
    file_name = st.selectbox("Select a document:", existing_files)
    if file_name:
        display_document_fields(file_name, use_elasticsearch)

elif option == "Use Existing Index":
    if use_elasticsearch:
        es_engine = initialize_elasticsearch_engine(file_name)
        st.success(f"Elasticsearch is ready. You can start searching. Index name: {es_engine.index_name}")
        st.session_state['index_ready'] = True
    else:
        existing_pickles = load_existing_pickles()
        selected_pickle = st.selectbox("Select an existing index:", existing_pickles)
        if selected_pickle:
            pickle_file = os.path.join(PICKLE_DIR, selected_pickle)
            search_index = SearchIndex([], [])  
            search_index.load_from_pickle(pickle_file)
            st.session_state['search_index'] = search_index
            st.session_state['keyword_fields'] = search_index.keyword_fields
            st.session_state['index_configured'] = True
            st.success(f"Index loaded from {selected_pickle}. Ready to search.")

if 'index_ready' in st.session_state and st.session_state['index_ready']:
    search_interface(use_elasticsearch, file_name)
