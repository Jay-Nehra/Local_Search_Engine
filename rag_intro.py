from Engine.engine import SearchIndex
import json


knowledge_base_document = 'Knowledge_Base/faq_documents.json'

with open(knowledge_base_document, 'r') as file:
    raw_documents = json.load(file)
    
# print(json.dumps(raw_documents[0], indent=4))

index = SearchIndex(
    text_fields=['answer', 'question'],
    keyword_fields=[],
)

user_query = 'can i join a course which has finished?'

index = index.fit(raw_documents)

result = index.search(user_query, 3)

# print(json.dumps(result, indent=4))

from elasticsearch import Elasticsearch

search_client = Elasticsearch("http://localhost:9200")

print(search_client.info())