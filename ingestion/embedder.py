import math
from collections.abc import Sequence
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI

from config import Settings


def create_openai_client(settings: Settings, credential: DefaultAzureCredential) -> OpenAI:
	settings.require("foundry_endpoint")
	key = settings.foundry_api_key.get_secret_value()
	token_provider = None if key else get_bearer_token_provider(
		credential, "https://cognitiveservices.azure.com/.default",
	)
	if urlparse(settings.foundry_endpoint).path.strip("/") == "openai/v1":
		return OpenAI(
			base_url=settings.foundry_endpoint.rstrip("/") + "/",
			api_key=key or token_provider,
			timeout=settings.request_timeout_seconds,
			max_retries=3,
		)
	return AzureOpenAI(
		azure_endpoint=settings.foundry_endpoint,
		api_version=settings.foundry_api_version,
		api_key=key or None,
		azure_ad_token_provider=token_provider,
		timeout=settings.request_timeout_seconds,
		max_retries=3,
	)


class Embedder:
	def __init__(self, settings: Settings, client: OpenAI):
		settings.require("foundry_embedding_deployment")
		self.settings = settings
		self.client = client

	def embed(self, texts: Sequence[str]) -> list[list[float]]:
		if any(not text.strip() for text in texts):
			raise ValueError("Cannot embed empty text")
		vectors = []
		for start in range(0, len(texts), self.settings.embedding_batch_size):
			batch = list(texts[start:start + self.settings.embedding_batch_size])
			options = {"dimensions": self.settings.foundry_embedding_dimensions} if self.settings.foundry_embedding_send_dimensions else {}
			response = self.client.embeddings.create(
				model=self.settings.foundry_embedding_deployment,
				input=batch,
				encoding_format="float",
				**options,
			)
			items = sorted(response.data, key=lambda item: item.index)
			if [item.index for item in items] != list(range(len(batch))):
				raise RuntimeError("Embedding service returned an incomplete or invalid batch")
			for item in items:
				if len(item.embedding) != self.settings.foundry_embedding_dimensions:
					raise ValueError("Embedding dimensions do not match FOUNDRY_EMBEDDING_DIMENSIONS")
				if not all(math.isfinite(value) for value in item.embedding):
					raise ValueError("Embedding service returned non-finite values")
				vectors.append(item.embedding)
		return vectors
