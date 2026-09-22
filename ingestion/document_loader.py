from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from config import Settings


TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | OCR_EXTENSIONS


@dataclass(frozen=True)
class Document:
	document_id: str
	source: str
	title: str
	text: str
	page: int | None = None


def load_document(path: Path, settings: Settings, credential: DefaultAzureCredential | None = None,
				  source: str | None = None) -> list[Document]:
	path = path.resolve(strict=True)
	if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
		raise ValueError(f"Unsupported document: {path.name}")
	if path.stat().st_size > settings.max_file_size_mb * 1024 * 1024:
		raise ValueError(f"Document exceeds MAX_FILE_SIZE_MB: {path.name}")
	source = source or path.name
	document_id = sha256(source.encode("utf-8")).hexdigest()
	if path.suffix.lower() in TEXT_EXTENSIONS:
		text = path.read_text(encoding="utf-8-sig")
		documents = [Document(document_id, source, path.name, text)]
	else:
		settings.require("doc_intelligence_endpoint")
		key = settings.doc_intelligence_key.get_secret_value()
		if not key and credential is None:
			raise ValueError("Document Intelligence requires a key or an Azure credential")
		with DocumentIntelligenceClient(
			settings.doc_intelligence_endpoint,
			AzureKeyCredential(key) if key else credential,
			connection_timeout=settings.request_timeout_seconds,
			read_timeout=settings.request_timeout_seconds,
		) as client, path.open("rb") as stream:
			result = client.begin_analyze_document(
				"prebuilt-layout", body=stream, content_type="application/octet-stream",
				string_index_type="unicodeCodePoint",
			).result(timeout=settings.request_timeout_seconds * 5)
		documents = []
		for page in result.pages or []:
			text = "".join((result.content or "")[span.offset:span.offset + span.length] for span in page.spans or [])
			if text.strip():
				documents.append(Document(document_id, source, path.name, text, page.page_number))
		if not documents and result.content:
			documents = [Document(document_id, source, path.name, result.content)]
	if not any(document.text.strip() for document in documents):
		raise ValueError(f"No readable text in document: {path.name}")
	return documents
