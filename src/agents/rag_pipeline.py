"""
Retrieval-Augmented Generation (RAG) Pipeline
Implements document indexing, retrieval, and LLM generation
for video content question answering.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json


@dataclass
class Document:
    """A document in the RAG index."""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from retrieval query."""
    document: Document
    score: float
    rank: int


class SimpleEmbedder:
    """
    Simple text embedding using bag-of-words + TF-IDF.
    In production, replace with OpenAI embeddings or sentence-transformers.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def _build_vocab(self, documents: List[str]):
        """Build vocabulary from documents."""
        vocab_set = set()
        for doc in documents:
            vocab_set.update(self._tokenize(doc))
        self.vocab = {word: i for i, word in enumerate(sorted(vocab_set))}
        self.dim = len(self.vocab)

    def _compute_idf(self, documents: List[str]):
        """Compute IDF scores."""
        n_docs = len(documents)
        for word in self.vocab:
            df = sum(1 for doc in documents if word in self._tokenize(doc))
            self.idf[word] = np.log(n_docs / (df + 1)) if df > 0 else 0

    def fit(self, documents: List[str]):
        """Fit the embedder on a corpus."""
        self._build_vocab(documents)
        self._compute_idf(documents)

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a vector."""
        embedding = np.zeros(self.dim)
        tokens = self._tokenize(text)
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                embedding[idx] = count * self.idf.get(word, 1.0) / len(tokens)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class VectorStore:
    """
    In-memory vector store for document retrieval.
    Supports cosine similarity search.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = np.array([])

    def add_documents(self, documents: List[Document]):
        """Add documents with embeddings to the store."""
        self.documents.extend(documents)
        new_embeddings = np.array([doc.embedding for doc in documents if doc.embedding is not None])
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        if len(self.embeddings) == 0:
            return np.array([])
        dot_products = self.embeddings @ query_embedding
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms[norms == 0] = 1
        return dot_products / norms

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents."""
        similarities = self.cosine_similarity(query_embedding)
        if len(similarities) == 0:
            return []

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(similarities[idx]),
                rank=rank + 1,
            ))
        return results


class RAGPipeline:
    """
    Full RAG pipeline: Index -> Retrieve -> Augment -> Generate.
    Implements retrieval-augmented generation for question answering
    over video content transcripts and descriptions.
    """

    def __init__(self, embedder: Optional[SimpleEmbedder] = None):
        self.embedder = embedder or SimpleEmbedder()
        self.vector_store = VectorStore()
        self.indexed: bool = False

    def index_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Index a corpus of documents."""
        metadatas = metadatas or [{} for _ in texts]
        self.embedder.fit(texts)

        documents = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
            embedding = self.embedder.embed(text)
            documents.append(Document(
                id=doc_id,
                text=text,
                metadata=meta,
                embedding=embedding,
            ))

        self.vector_store.add_documents(documents)
        self.indexed = True
        print(f"Indexed {len(documents)} documents (dim={self.embedder.dim})")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        if not self.indexed:
            return []

        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def augment_prompt(self, query: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Augment the user query with retrieved context."""
        context = "\n\n".join([
            f"[{r.rank}] (score: {r.score:.3f}) {r.document.text[:500]}"
            for r in retrieved_docs
        ])
        return f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

    def generate(self, query: str, top_k: int = 3) -> Dict:
        """Full RAG generation: retrieve + augment + generate."""
        # Retrieve
        results = self.retrieve(query, top_k=top_k)

        # Augment
        augmented_prompt = self.augment_prompt(query, results)

        # Generate (simulated LLM call)
        if results:
            top_result = results[0].document.text[:200]
            answer = f"Based on retrieved context: {top_result}..."
        else:
            answer = "No relevant context found."

        return {
            "query": query,
            "answer": answer,
            "retrieved": [
                {"text": r.document.text[:100], "score": r.score, "rank": r.rank}
                for r in results
            ],
            "augmented_prompt": augmented_prompt[:300] + "..." if len(augmented_prompt) > 300 else augmented_prompt,
        }


if __name__ == "__main__":
    # Demo RAG pipeline
    print("RAG Pipeline Demo")
    print("=" * 50)

    corpus = [
        "The video shows a person cooking pasta in a kitchen with fresh ingredients.",
        "Scene 2 transitions to a outdoor park where children are playing soccer.",
        "The final scene shows a sunset over the ocean with calm waves.",
        "A tutorial on how to edit videos using AI-powered tools and transitions.",
        "An interview with a tech entrepreneur about AI and content creation.",
    ]

    rag = RAGPipeline()
    rag.index_documents(corpus)

    query = "What happens in the outdoor scene?"
    result = rag.generate(query, top_k=2)

    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['retrieved'])} documents:")
    for r in result["retrieved"]:
        print(f"  [{r['rank']}] score={r['score']:.3f}: {r['text']}")
