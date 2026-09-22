from dataclasses import dataclass

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from config import Settings
from ingestion.embedder import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
	id: str
	source: str
	title: str
	content: str
	page: int | None
	score: float


class Retriever:
	def __init__(self, settings: Settings, client: SearchClient, embedder: Embedder):
		self.settings = settings
		self.client = client
		self.embedder = embedder

	def retrieve(self, question: str, top_k: int | None = None, source: str | None = None) -> list[RetrievedChunk]:
		if not question.strip():
			raise ValueError("Question must not be blank")
		top_k = self.settings.retrieval_top_k if top_k is None else top_k
		if not 1 <= top_k <= 20:
			raise ValueError("top_k must be between 1 and 20")
		vector = self.embedder.embed([question])[0]
		source_filter = "source eq '" + source.replace("'", "''") + "'" if source is not None else None
		results = self.client.search(
			search_text=question,
			search_fields=["content", "title"],
			vector_queries=[VectorizedQuery(vector=vector, fields="content_vector", k_nearest_neighbors=max(50, top_k))],
			filter=source_filter,
			vector_filter_mode="preFilter",
			select=["id", "source", "title", "content", "page"],
			top=top_k,
		)
		chunks = []
		seen = set()
		for result in results:
			if result["id"] in seen or not result.get("content", "").strip():
				continue
			seen.add(result["id"])
			chunks.append(RetrievedChunk(
				result["id"], result["source"], result.get("title", ""),
				result["content"], result.get("page"), float(result.get("@search.score", 0)),
			))
		return chunks
