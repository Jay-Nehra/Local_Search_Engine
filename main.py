from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict
import json
import os

# Import our SearchIndex class
from Engine.engine import SearchIndex

app = FastAPI()

# Paths to our data files
data_file = 'Knowledge_Base/faq_documents.json'
pickle_file = 'VectorStore/tfidf_data.pkl'

# Initialize the SearchIndex instance
text_fields = ['question', 'answer']
keyword_fields = ['course', 'section']
index = SearchIndex(text_fields=text_fields, keyword_fields=keyword_fields)

# Check if we should load the existing vectors or re-fit the data
if os.path.exists(pickle_file) and os.path.getmtime(pickle_file) >= os.path.getmtime(data_file):
    index.load_from_pickle(pickle_file)
else:
    with open(data_file, 'r') as f:
        docs = json.load(f)
    index.fit(docs, pickle_file=pickle_file)


class SearchQuery(BaseModel):
    query: str
    num_results: int = 5
    filter_dict: Dict[str, str] = None

@app.post("/search/")
def search_documents(search: SearchQuery):
    try:
        results = index.search(search.query, num_results=search.num_results, filter_dict=search.filter_dict)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
