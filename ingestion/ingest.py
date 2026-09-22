import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from config import get_settings
from ingestion.chunker import TokenChunker
from ingestion.document_loader import SUPPORTED_EXTENSIONS, load_document
from ingestion.embedder import Embedder, create_openai_client
from ingestion.indexer import SearchIndexer


logger = logging.getLogger(__name__)


def ingest_path(path: Path, source_prefix: str = "") -> dict:
	path = path.resolve(strict=True)
	settings = get_settings()
	settings.require("search_endpoint", "foundry_endpoint", "foundry_embedding_deployment")
	files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink() and item.suffix.lower() in SUPPORTED_EXTENSIONS) if path.is_dir() else [path]
	if not files:
		raise ValueError("No supported documents found")
	root = path if path.is_dir() else path.parent
	with ExitStack() as stack:
		credential = stack.enter_context(DefaultAzureCredential())
		openai_client = stack.enter_context(create_openai_client(settings, credential))
		key = settings.search_api_key.get_secret_value()
		search_credential = AzureKeyCredential(key) if key else credential
		client = stack.enter_context(SearchClient(
			settings.search_endpoint, settings.search_index_name, search_credential,
			connection_timeout=settings.request_timeout_seconds, read_timeout=settings.request_timeout_seconds,
		))
		index_client = stack.enter_context(SearchIndexClient(
			settings.search_endpoint, search_credential,
			connection_timeout=settings.request_timeout_seconds, read_timeout=settings.request_timeout_seconds,
		))
		indexer = SearchIndexer(settings, client, index_client)
		indexer.ensure_index()
		embedder = Embedder(settings, openai_client)
		chunker = TokenChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
		report = {"documents": 0, "chunks": 0, "failed": []}
		for file in files:
			relative = file.relative_to(root).as_posix()
			source = f"{source_prefix.rstrip('/')}/{relative}" if source_prefix else relative
			try:
				documents = load_document(file, settings, credential, source)
				chunks = [chunk for document in documents for chunk in chunker.split(document)]
				vectors = embedder.embed([chunk.content for chunk in chunks])
				count = indexer.replace_document(chunks, vectors)
				report["documents"] += 1
				report["chunks"] += count
				logger.info("Indexed %s (%s chunks)", source, count)
			except Exception as error:
				logger.error("Failed to ingest %s (%s)", source, type(error).__name__)
				report["failed"].append({"source": source, "error": type(error).__name__})
		return report


def main() -> int:
	parser = argparse.ArgumentParser(description="Ingest local documents into Azure AI Search")
	parser.add_argument("path", type=Path, help="Document or directory to ingest recursively")
	parser.add_argument("--source-prefix", default="", help="Stable collection prefix to avoid source-name collisions")
	arguments = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	logging.getLogger("azure").setLevel(logging.WARNING)
	logging.getLogger("httpx").setLevel(logging.WARNING)
	if not arguments.path.exists():
		logger.error("Input path does not exist: %s. Add documents at this path or pass an existing file or directory.", arguments.path.absolute())
		return 1
	try:
		report = ingest_path(arguments.path, arguments.source_prefix)
	except Exception as error:
		logger.error("Ingestion could not start (%s). Check configuration and service access.", type(error).__name__)
		return 1
	print(json.dumps(report, indent=2))
	return 1 if report["failed"] else 0


if __name__ == "__main__":
	raise SystemExit(main())
