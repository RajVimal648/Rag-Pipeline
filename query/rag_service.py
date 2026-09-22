import json
import re
from collections.abc import Generator
from contextlib import ExitStack, contextmanager

import tiktoken
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from pydantic import BaseModel

from config import Settings
from ingestion.embedder import Embedder, create_openai_client
from query.llm_client import LLMClient
from query.retriever import RetrievedChunk, Retriever


NO_EVIDENCE = "I couldn't find enough information in the indexed documents to answer that question."


class Citation(BaseModel):
	number: int
	chunk_id: str
	source: str
	title: str
	page: int | None
	content: str
	score: float


class QueryResponse(BaseModel):
	answer: str
	sources: list[Citation]


class RAGService:
	def __init__(self, settings: Settings, retriever: Retriever, llm: LLMClient):
		self.settings = settings
		self.retriever = retriever
		self.llm = llm
		self.encoding = tiktoken.get_encoding("cl100k_base")

	def build_context(self, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
		blocks = []
		sources = []
		for chunk in chunks:
			number = len(sources) + 1
			tokens = self.encoding.encode(chunk.content, disallowed_special=())

			def render(count: int) -> tuple[str, str]:
				text = self.encoding.decode(tokens[:count], errors="ignore")
				block = json.dumps({"source_id": number, "source": chunk.source, "title": chunk.title, "page": chunk.page, "content": text}, ensure_ascii=False)
				return block, text

			lower, upper = 0, len(tokens)
			while lower < upper:
				middle = (lower + upper + 1) // 2
				block, _ = render(middle)
				candidate = "\n".join([*blocks, block])
				if len(self.encoding.encode(candidate, disallowed_special=())) <= self.settings.context_max_tokens:
					lower = middle
				else:
					upper = middle - 1
			block, content = render(lower)
			candidate = "\n".join([*blocks, block])
			if not content.strip() or len(self.encoding.encode(candidate, disallowed_special=())) > self.settings.context_max_tokens:
				continue
			blocks.append(block)
			sources.append(Citation(
				number=number, chunk_id=chunk.id, source=chunk.source, title=chunk.title,
				page=chunk.page, content=content, score=chunk.score,
			))
		return "\n".join(blocks), sources

	def query(self, question: str, top_k: int | None = None, source: str | None = None) -> QueryResponse:
		question = question.strip()
		if not question or len(question) > 4000:
			raise ValueError("Question must contain between 1 and 4000 characters")
		chunks = self.retriever.retrieve(question, top_k, source)
		context, sources = self.build_context(chunks)
		if not sources:
			return QueryResponse(answer=NO_EVIDENCE, sources=[])
		draft = self.llm.generate(question, context)
		used = set(draft.source_ids)
		inline = {int(number) for number in re.findall(r"\[(\d+)\]", draft.answer)}
		available = {citation.number for citation in sources}
		if not used or used != inline or not used.issubset(available):
			return QueryResponse(answer=NO_EVIDENCE, sources=[])
		return QueryResponse(answer=draft.answer, sources=[citation for citation in sources if citation.number in used])


@contextmanager
def create_rag_service(settings: Settings) -> Generator[RAGService, None, None]:
	settings.require("foundry_endpoint", "foundry_embedding_deployment", "foundry_chat_deployment", "search_endpoint")
	with ExitStack() as stack:
		credential = stack.enter_context(DefaultAzureCredential())
		openai_client = stack.enter_context(create_openai_client(settings, credential))
		key = settings.search_api_key.get_secret_value()
		search_client = stack.enter_context(SearchClient(
			settings.search_endpoint, settings.search_index_name,
			AzureKeyCredential(key) if key else credential,
			connection_timeout=settings.request_timeout_seconds, read_timeout=settings.request_timeout_seconds,
		))
		embedder = Embedder(settings, openai_client)
		yield RAGService(settings, Retriever(settings, search_client, embedder), LLMClient(settings, openai_client))
