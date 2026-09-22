import json
from collections.abc import Callable, Iterable
from dataclasses import asdict

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
	HnswAlgorithmConfiguration,
	HnswParameters,
	SearchableField,
	SearchField,
	SearchFieldDataType,
	SearchIndex,
	SimpleField,
	VectorSearch,
	VectorSearchProfile,
)

from config import Settings
from ingestion.chunker import Chunk


def build_index(settings: Settings) -> SearchIndex:
	return SearchIndex(
		name=settings.search_index_name,
		fields=[
			SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
			SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
			SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
			SearchableField(name="title", type=SearchFieldDataType.String),
			SearchableField(name="content", type=SearchFieldDataType.String),
			SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
			SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
			SearchField(
				name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
				searchable=True, hidden=True,
				vector_search_dimensions=settings.foundry_embedding_dimensions,
				vector_search_profile_name="rag-vector-profile",
			),
		],
		vector_search=VectorSearch(
			algorithms=[HnswAlgorithmConfiguration(name="rag-hnsw", parameters=HnswParameters(metric="cosine"))],
			profiles=[VectorSearchProfile(name="rag-vector-profile", algorithm_configuration_name="rag-hnsw")],
		),
	)


class SearchIndexer:
	def __init__(self, settings: Settings, client: SearchClient, index_client: SearchIndexClient):
		self.settings = settings
		self.client = client
		self.index_client = index_client

	def ensure_index(self) -> None:
		desired = build_index(self.settings)
		try:
			current = self.index_client.get_index(self.settings.search_index_name)
		except ResourceNotFoundError:
			try:
				self.index_client.create_index(desired)
				return
			except ResourceExistsError:
				current = self.index_client.get_index(self.settings.search_index_name)
		fields = {field.name: field for field in current.fields}
		for expected in desired.fields:
			actual = fields.get(expected.name)
			if actual is None or actual.type != expected.type:
				raise ValueError(f"Index field {expected.name} is incompatible; use a new SEARCH_INDEX_NAME")
			for flag in ("key", "filterable", "searchable"):
				if getattr(expected, flag, False) and not getattr(actual, flag, False):
					raise ValueError(f"Index field {expected.name} requires {flag}; use a new SEARCH_INDEX_NAME")
		vector = fields["content_vector"]
		if vector.vector_search_dimensions != self.settings.foundry_embedding_dimensions or not vector.vector_search_profile_name:
			raise ValueError("Index vector dimensions/profile are incompatible; use a new SEARCH_INDEX_NAME")

	def _write(self, documents: Iterable[dict], operation: Callable) -> None:
		batch = []
		batch_bytes = 0

		def flush() -> None:
			results = operation(documents=batch)
			failed = [result for result in results if not result.succeeded]
			if len(results) != len(batch) or failed:
				raise RuntimeError("Search indexing was partially unsuccessful; rerun ingestion before querying")

		for document in documents:
			size = len(json.dumps(document, ensure_ascii=False).encode("utf-8"))
			if size > 14 * 1024 * 1024:
				raise ValueError("A search document exceeds the indexing payload limit")
			if batch and (len(batch) >= 64 or batch_bytes + size > 14 * 1024 * 1024):
				flush()
				batch = []
				batch_bytes = 0
			batch.append(document)
			batch_bytes += size
		if batch:
			flush()

	def replace_document(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
		if not chunks or len(chunks) != len(vectors):
			raise ValueError("Expected one embedding for every nonempty chunk")
		document_ids = {chunk.document_id for chunk in chunks}
		if len(document_ids) != 1:
			raise ValueError("Replace one source document at a time")
		if len({chunk.id for chunk in chunks}) != len(chunks):
			raise ValueError("Chunk IDs must be unique")
		if any(len(vector) != self.settings.foundry_embedding_dimensions for vector in vectors):
			raise ValueError("Indexing vector dimensions do not match configuration")
		document_id = chunks[0].document_id.replace("'", "''")
		previous = {result["id"] for result in self.client.search(
			search_text="*", filter=f"document_id eq '{document_id}'", select=["id"],
		)}
		documents = ({**asdict(chunk), "content_vector": vector} for chunk, vector in zip(chunks, vectors, strict=True))
		self._write(documents, self.client.upload_documents)
		obsolete = previous - {chunk.id for chunk in chunks}
		self._write(({"id": identity} for identity in sorted(obsolete)), self.client.delete_documents)
		return len(chunks)
