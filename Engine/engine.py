""" 
    Objective:
        - To implement a basic search engine that uses TF-IDF  and cosine similarity. The goal of this search engine is to allow me to search through a collection of documents and return the most relevant results based on the query.
    Core Concept:
        - TF-IDF (Term Frequency-Inverse Document Frequency):
            - This is a statistical measure that is used to evaluate how important a word is to a document in a collection/ corpus of documents. 
            - It is a way to weigh the words so that common words like `the`, `is` etc are given less importance, while rarer bu potentially more meaningful words are given more weight. 
            - Process:
                - TF: This is the `frequency of a term` in a document.
                - IDF: This is inverse of the number of documents containing the term. The more any word or term appears in many document gets a lower score, while terms that appear in fewer documents get a higher score. 
                - Product of these two gives the TF-IDF score, 
                    - This indicates how important is a term to a particular documet in context of the entire corpus.
        - Cosine Similarity:
            - Cosine similarity is used to measure similarity between two non-zero vectors. It’s used here to measure the similarity between the query vector and the document vectors.
            - Cosine similarity computes the cosine of the angle between two vectors in a multidimensional space. A cosine similarity of 1 means the vectors are identical, 0 means they are orthogonal (no similarity), and -1 means they are opposite.

"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SearchIndex:
    """
        This contains the core functionality of the search engine. It manages the inndexing the documents and provides the capability to searhc those documents.
    """
    
    def __init__(self, text_fields: List[str], keyword_fields: List[str], vectorizer_params: Optional[Dict] = None, boosting_factors: Optional[Dict[str, float]] = None):
        """ 
            text_fields: Fields in the JSON documents that contain the text data that we want to index like ['question', 'answer']
            keyword_fields: Fields in the JSON documents that we will use for the exact matching like ['section', 'course']
            vectorizers: A dictionary of the TfidfVectorizer objects, one for each of the text fields. These will be used to convert the text data into TF-IDF Vectors. 
            
        """
        
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vectorizer_params = vectorizer_params if vectorizer_params else {}
        self.boosting_factors = boosting_factors if boosting_factors else {field: 1.0 for field in text_fields}  # Default boost of 1.0

        # Initialize the TF-IDF vectorizers for each text field with provided params
        self.vectorizers = {field: TfidfVectorizer(**self.vectorizer_params) for field in text_fields}
        
        # # Initialize specific vectorizers for known text fields
        # self.question_vectorizer = TfidfVectorizer(**vectorizer_params)
        # self.answer_vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Placeholder to store documents
        self.documents = []
        
        self.keyword_df = None
        self.text_matrices = {}
        
    def save_to_pickle(self, pickle_file: str):
        """Saves the precomputed data to a pickle file for future use."""
        with open(pickle_file, 'wb') as f:
            pickle.dump((self.text_matrices, self.vectorizers, self.keyword_df, self.documents), f)

    def load_from_pickle(self, pickle_file: str):
        """Loads the precomputed data from a pickle file."""
        with open(pickle_file, 'rb') as f:
            self.text_matrices, self.vectorizers, self.keyword_df, self.documents = pickle.load(f)
    
        
    def fit(self, documents: List[Dict[str,str]], pickle_file: str = None) -> 'SearchIndex':
        """ 
            Here, we are trying to process the raw documents and process them in a dataframe for fast searching. 
            Process:
                - Input is the document JSON, which is a list of dictionaries. 
                - Text Fields vs Keyword Fields:
                    - Text Fields contain large blocks of text, like sentences or paragraphs. Examples might be "question" or "answer" in our documents.
                        - The goal with text fields is to enable full-text search, where the user can input a query, and the search engine finds the most relevant documents based on the content of these fields.
                        - Text can be lengthy and varied, with different words and phrases that might mean the same thing. Therefore, a straightforward exact match (like what we might do with keyword fields) wouldn’t work well here.
                        - Text Fields Require more sophisticated processing to understand the content and meaning of the text. This is where TF-IDF and cosine similarity come in—they allow the search engine to understand not just if a word appears in a document, but how important that word is relative to the rest of the text and other documents.
                    - Keyword Fields are categorical or single-word fields, like "course" or "section".
                        - The goal with keyword fields is often to filter the search results. For example, we might want to limit our search to a specific course.
                        - Keyword fields are typically used for exact matches—where you want to check if the document’s field exactly matches the user’s filter criteria.
                        - These fields don’t need complex processing because they’re typically used for exact or categorical matches. We can directly compare these fields to filter results.

        """
        # First save the raw documents so that we can return them when a search is performed.
        self.documents = documents
        
        # Process the text fields
            # - For each field:
            #     - Extract the text from each document. Loop through the documents and extract relevant fields for each of them.
            #     - Convert text into TF-IDF matrix using the TfidfVectorizer. Use the TfidfVectorizer instance that created in the init method to convert the text into matrix.
            #     - store this matrix 
        
        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in documents]
            matrix = self.vectorizers[field].fit_transform(texts)
            # Apply boosting at the indexing stage if specified
            if field in self.boosting_factors:
                matrix *= self.boosting_factors[field]
            self.text_matrices[field] = matrix
            
        # Process keyword fields
        self.keyword_df = pd.DataFrame({field: [doc.get(field, '') for doc in documents] for field in self.keyword_fields})

        if pickle_file:
            self.save_to_pickle(pickle_file)
        return self
    
    def search(self, user_query: str, num_results: int = 5, filter_dict: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """ 
        Objective:
            - Implement a method that takes a query and returns the most relevant documents based on how similar they are to the query.
        Process:
            - Initialize a Scores Array: We need a place to store similarity scores for each document.
            - Iterate Over Text Fields: For each text field (like "question" and "answer"), we need to:
                - Convert the search query into a TF-IDF vector using the same vectorizers that were used to index the documents.
                - Use cosine similarity to compare the query vector with the document vectors, which will give us a measure of how similar each document is to the query.
                - Accumulate these scores.
            - Rank the top documents by their similarity score and return the top results.
        """
        
        # Initialize the array to hold the similarity scores of the documents 
        similarity_scores = np.zeros(len(self.documents))
        
        # Compute the similarity score for each text field in the documents
        for field, vectorizer in self.vectorizers.items():
            # For each text field, transform the query into a TF-IDF vector
            query_vector = vectorizer.transform([user_query])
            
            # Calculate cosine similarity between the query vector and the document vectors
            similarity = cosine_similarity(query_vector, self.text_matrices[field]).flatten()
            
            # Add the boosting factor for the question field
            # Inside the search method, add a boosting factor
            boost = self.boosting_factors.get(field, 1.0)
            
            # Accumulate the scores
            similarity_scores += similarity * boost
        
        # Apply filtering based on filter_dict
        if filter_dict:
            for field, value in filter_dict.items():
                if field in self.keyword_fields and value in self.keyword_df[field].values:
                    mask = self.keyword_df[field] == value
                    similarity_scores *= mask.astype(int).to_numpy()
                else:
                    similarity_scores *= 0  # Apply zero mask if filter does not match any document
    
        # Rank documents by their scores, get the indices of the top results
        top_indices = np.argsort(similarity_scores)[-num_results:][::-1]

        # Retrieve the top documents based on the sorted indices
        top_docs = [self.documents[i] for i in top_indices if similarity_scores[i] > 0]

        return top_docs
