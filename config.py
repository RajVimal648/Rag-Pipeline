import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


ROOT = Path(__file__).resolve().parent


class Settings(BaseModel):
	model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

	foundry_endpoint: str = ""
	foundry_api_key: SecretStr = SecretStr("")
	foundry_api_version: str = "2024-10-21"
	foundry_chat_deployment: str = ""
	foundry_embedding_deployment: str = ""
	foundry_embedding_dimensions: int = Field(default=1536, ge=1, le=4096)
	foundry_embedding_send_dimensions: bool = True
	doc_intelligence_endpoint: str = ""
	doc_intelligence_key: SecretStr = SecretStr("")
	search_endpoint: str = ""
	search_api_key: SecretStr = SecretStr("")
	search_index_name: str = "rag-documents"
	chunk_size_tokens: int = Field(default=800, ge=16, le=8000)
	chunk_overlap_tokens: int = Field(default=120, ge=0)
	embedding_batch_size: int = Field(default=16, ge=1, le=128)
	retrieval_top_k: int = Field(default=5, ge=1, le=20)
	context_max_tokens: int = Field(default=6000, ge=128, le=100000)
	max_answer_tokens: int = Field(default=1000, ge=1, le=16000)
	request_timeout_seconds: float = Field(default=60, gt=0, le=600)
	max_file_size_mb: int = Field(default=20, ge=1, le=500)
	api_key: SecretStr = SecretStr("")

	@model_validator(mode="after")
	def validate_settings(self) -> "Settings":
		if self.chunk_overlap_tokens >= self.chunk_size_tokens:
			raise ValueError("CHUNK_OVERLAP_TOKENS must be less than CHUNK_SIZE_TOKENS")
		if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,126}[a-z0-9]", self.search_index_name):
			raise ValueError("SEARCH_INDEX_NAME must be 2-128 lowercase letters, digits or hyphens")
		for name in ("foundry_endpoint", "search_endpoint", "doc_intelligence_endpoint"):
			value = getattr(self, name)
			if value:
				parsed = urlparse(value)
				if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
					raise ValueError(f"{name.upper()} must be an HTTPS service endpoint")
				allowed_paths = {"", "openai/v1"} if name == "foundry_endpoint" else {""}
				if parsed.path.strip("/") not in allowed_paths:
					raise ValueError(f"{name.upper()} must be a resource root or, for Foundry, an /openai/v1 URL")
		return self

	def require(self, *names: str) -> None:
		missing = [name.upper() for name in names if not getattr(self, name)]
		if missing:
			raise ValueError("Missing configuration: " + ", ".join(missing))


@lru_cache
def get_settings() -> Settings:
	values = {**dotenv_values(ROOT / ".env"), **os.environ}
	return Settings.model_validate({name: values[name.upper()] for name in Settings.model_fields if name.upper() in values})
