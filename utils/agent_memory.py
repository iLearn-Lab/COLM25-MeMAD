import chromadb
import requests
from typing import List, Tuple, Dict, Callable, Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer


class MemoryMAD_VectorDB:
    def __init__(self, persistent_dir: str, collection_name: str,
                 tokenizer: Optional[Callable[[str], List[str]]] = word_tokenize,
                 stemmer: Optional[Callable[[str], List[str]]] = SnowballStemmer("english"),
                 model_name: str = "bgem3", verbose: bool = True):
        
        self.client = chromadb.PersistentClient(path=persistent_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )

        self.ollama_url = "http://localhost:11434/api/embeddings"

        if model_name == "bgem3":
            self.model_name = "bge-m3:latest"
        elif model_name == "nomic":
            self.model_name = "nomic-embed-text:latest"
        elif model_name == "mxbai":
            self.model_name = "mxbai-embed-large:latest"
        elif model_name == "bm25":
            self.model_name = "bm25"
        else:
            raise NotImplementedError

        self.verbose = verbose

        self.tokenizer = tokenizer
        self.stemmer = stemmer

        print("="*50, f"Embedding Model: {self.model_name}", "="*50)

    def get_embedding(self, text: str) -> List[float]:
        try:

            if self.model_name == "bm25":
                response = requests.post(
                    self.ollama_url,
                    json={"model": "bge-m3", "prompt": text}
                )
            else:
                response = requests.post(
                    self.ollama_url,
                    json={"model": self.model_name, "prompt": text}
                )
            return response.json()["embedding"]
        except Exception as e:
            raise Exception(f"Failed to retrieve embedding: {str(e)}")

    def add_memory(self, db_id: str, embedding_text: str, document_text: str,
                   add_meta_datas: Dict = None) -> None:
        try:
            assert isinstance(embedding_text, str), "embedding_text must be str"
            embedding = self.get_embedding(embedding_text)

            meta_datas = dict()
            if add_meta_datas is not None:
                assert isinstance(add_meta_datas, dict), "meta_datas must be Dict"
                for key, val in add_meta_datas.items():
                    meta_datas[key] = val

            tokens = " ".join([self.stemmer.stem(token) for token in self.tokenizer(embedding_text.lower())])
            meta_datas["key_tokens"] = tokens

            self.collection.upsert(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[meta_datas],
                ids=[db_id]
            )
            if self.verbose:
                print(f"Successfully added memory ID.: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to add memory.: {str(e)}")

    def query_similar(self, query_text: str, n_results: int = 3, filter_metadata: Dict = None) -> List[Tuple[Dict, str, float]]:
        try:
            if self.model_name == "bm25":
                similar_pairs = self.query_by_bm25(query_text=query_text, filter_metadata=filter_metadata, n_results=n_results)
            else:
                query_embedding = self.get_embedding(query_text)

                if filter_metadata is not None:
                    assert isinstance(filter_metadata, dict), "filter_metadata must be Dict"

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"],
                    where=filter_metadata
                )

                similar_pairs = []
                for metadata, document, distance in zip(results['metadatas'][0], results['documents'][0], results["distances"][0]):
                    similar_pairs.append((metadata, document, distance))

            return similar_pairs
        except Exception as e:
            raise Exception(f"Query failed.: {str(e)}")

    def get_by_ids(self, select_ids: List[str]) -> List[Tuple[Dict, str]]:
        try:
            results = self.collection.get(ids=select_ids, include=['metadatas', 'documents'])

            similar_pairs = []
            for metadata, document in zip(results['metadatas'], results['documents']):
                similar_pairs.append((metadata, document))

            return similar_pairs
        except Exception as e:
            raise Exception(f"Query by IDs failed: {str(e)}")
        
    def update_memory(self, db_id: str, embedding_text: str, document_text: str,
                      add_meta_datas: Dict = None) -> None:

        try:
            assert isinstance(embedding_text, str), "embedding_text must be str"
            embedding = self.get_embedding(embedding_text)

            meta_datas = dict()
            if add_meta_datas is not None:
                assert isinstance(add_meta_datas, dict), "meta_datas must be Dict"
                for key, val in add_meta_datas.items():
                    meta_datas[key] = val

            tokens = " ".join([self.stemmer.stem(token) for token in self.tokenizer(embedding_text.lower())])
            meta_datas["key_tokens"] = tokens

            self.collection.update(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[meta_datas],
                ids=[db_id]
            )

            if self.verbose:
                print(f"Successfully updated memory ID: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to update memory: {str(e)}")

    def delete_memory(self, db_id: str) -> None:
        try:
            self.collection.delete(ids=[db_id])
            if self.verbose:
                print(f"Successfully deleted memory ID: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to delete memory: {str(e)}")
        
    def get_memories_count(self) -> int:
        return self.collection.count()

    def get_all_ids(self, filter_metadata=None):
        data = self.collection.get(include=["metadatas"], where=filter_metadata)
        ids = data['ids']
        return ids

    def query_by_bm25(self, query_text, filter_metadata=None, n_results: int = 3):
        try:
            total_data = self.collection.get(include=["metadatas"], where=filter_metadata)
            corpus = [item["key_tokens"].split(" ") for item in total_data['metadatas']]
            db_ids = total_data['ids']

            bm25 = BM25Okapi(corpus)
            query_tokens = [self.stemmer.stem(token) for token in self.tokenizer(query_text.lower())]
            doc_scores = bm25.get_scores(query_tokens)

            scored_docs = sorted(
                zip(db_ids, doc_scores),
                key=lambda x: x[1],
                reverse=True
            )

            selected_db_ids = [db_id for db_id, _ in scored_docs[:n_results]]
            selected_scores = [score for _, score in scored_docs[:n_results]]
            results = self.collection.get(ids=selected_db_ids, include=['metadatas', 'documents'])
            similar_pairs = []
            for metadata, document, score in zip(results['metadatas'], results['documents'], selected_scores):
                similar_pairs.append((metadata, document, score))

            return similar_pairs
        except Exception as e:
            raise Exception(f"Query by BM25 fail: {str(e)}")
