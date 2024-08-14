from Engine.engine import SearchIndex
import json

# Step 1: Load the JSON documents
with open('Knowledge_Base/faq_documents.json', 'r') as file:
    docs = json.load(file)

# Step 2: Initialize the Index class
text_fields = ['question', 'answer']  # Assuming these are the text fields in our documents
keyword_fields = ['course', 'section']  # Assuming these are the keyword fields

index = SearchIndex(text_fields=text_fields, keyword_fields=keyword_fields)

# Step 3: Fit the index with the documents
index.fit(docs)

# Step 4: Inspect the results
# Check the TF-IDF matrix for the 'question' field
print("TF-IDF Matrix for 'question':")
print(index.text_matrices['question'].toarray())  # Convert sparse matrix to dense for inspection

# Check the TF-IDF matrix for the 'answer' field
print("\nTF-IDF Matrix for 'answer':")
print(index.text_matrices['answer'].toarray())  # Convert sparse matrix to dense for inspection

# Check the keywords 
print("Keyword dataframe")
print(index.keyword_df)

# Optionally, check the stored documents
print("\nStored documents:")
print(index.documents[:2])  # Print the first two documents to verify they are stored correctly

# Step 5: Test the search method
# Example queries to test the search functionality
queries = [
    "When will the course start?",
    "What are the prerequisites?",
    "How do I register for the course?",
]

filter_dict = {
    "course": "data-engineering-zoomcamp",
    "section": "General course-related questions"
}
import pprint
# Perform searches and print the results
for query in queries:
    print(f"Query: {query}")
    top_docs = index.search(query, num_result=3, filter_dict=filter_dict)
    for i, doc in enumerate(top_docs):
        print(f"Result {i+1}:")
        pprint.pprint(doc, indent=4)
        
    print("\n" + "-"*50 + "\n")
