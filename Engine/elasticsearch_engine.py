from elasticsearch import Elasticsearch

class ElasticsearchEngine:
    def __init__(self, index_name="my_index", host="localhost", port=9200):
        """
        Initializes the ElasticsearchEngine with the given index name, host, and port.
        """
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
        self.index_name = index_name

    def index_documents(self, documents):
        """
        Indexes a list of documents in Elasticsearch.
        Checks if the index already exists; if not, it creates the index and indexes the documents.
        If the index exists, it skips re-indexing.

        :param documents: List of dictionaries, each representing a document to be indexed.
        :return: String message indicating the result of the indexing operation.
        """
        if not self.index_exists():
            self.create_index()
            for i, doc in enumerate(documents):
                self.es.index(index=self.index_name, id=i, body=doc)
            return f"Indexed {len(documents)} documents."
        else:
            return "Index already exists. Skipping indexing."

    def create_index(self, settings=None):
        """
        Creates the Elasticsearch index with optional settings.

        :param settings: Dictionary of settings for the Elasticsearch index.
        """
        if settings is None:
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        self.es.indices.create(index=self.index_name, body=settings)

    def index_exists(self):
        """
        Checks if the index already exists in Elasticsearch.

        :return: Boolean indicating whether the index exists.
        """
        return self.es.indices.exists(index=self.index_name)

    def delete_index(self):
        """
        Deletes the Elasticsearch index.

        :return: Dictionary with the result of the delete operation.
        """
        return self.es.indices.delete(index=self.index_name, ignore=[400, 404])

    def search_documents(self, query, fields, filter_dict=None, num_results=5):
        """
        Searches for documents in Elasticsearch based on a query and optional filters.

        :param query: The search query string.
        :param fields: List of fields to search in (e.g., ['question', 'answer']).
        :param filter_dict: Optional dictionary of filters to apply (e.g., {'section': 'General'}).
        :param num_results: Number of top results to return.
        :return: List of dictionaries representing the search results.
        """
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": fields
                        }
                    },
                    "filter": [
                        {"term": {k: v}} for k, v in filter_dict.items()
                    ] if filter_dict else []
                }
            },
            "size": num_results
        }
        response = self.es.search(index=self.index_name, body=query_body)
        results = [hit['_source'] for hit in response['hits']['hits']]
        return results
