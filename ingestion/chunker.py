from dataclasses import dataclass
from hashlib import sha256

import tiktoken

from ingestion.document_loader import Document


@dataclass(frozen=True)
class Chunk:
	id: str
	document_id: str
	source: str
	title: str
	content: str
	page: int | None
	chunk_index: int


class TokenChunker:
	def __init__(self, chunk_size: int = 800, overlap: int = 120):
		if chunk_size < 16 or not 0 <= overlap < chunk_size:
			raise ValueError("Chunk size must be >= 16 and overlap must be in [0, chunk size)")
		self.chunk_size = chunk_size
		self.overlap = overlap
		self.encoding = tiktoken.get_encoding("cl100k_base")

	def split(self, document: Document) -> list[Chunk]:
		tokens = self.encoding.encode(document.text, disallowed_special=())
		chunks = []
		start = 0
		while start < len(tokens):
			end = min(start + self.chunk_size, len(tokens))
			while end > start:
				try:
					content = self.encoding.decode(tokens[start:end], errors="strict")
					if len(self.encoding.encode(content, disallowed_special=())) <= self.chunk_size:
						break
				except UnicodeDecodeError:
					pass
				end -= 1
			else:
				raise ValueError("Chunk size is too small to preserve Unicode text")
			if content.strip():
				index = len(chunks)
				identity = f"{document.document_id}:{document.page}:{index}:{content}"
				chunks.append(Chunk(
					sha256(identity.encode("utf-8")).hexdigest(), document.document_id,
					document.source, document.title, content, document.page, index,
				))
			if end == len(tokens):
				break
			next_start = max(start + 1, end - self.overlap)
			while next_start < end and self.encoding.decode_single_token_bytes(tokens[next_start])[0] & 0xC0 == 0x80:
				next_start += 1
			start = next_start
		return chunks
