import requests 
import json

faq_document = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(faq_document)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']
    
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
    
    with open('./Knowledge_Base/faq_documents.json', 'w', encoding='utf-8') as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)