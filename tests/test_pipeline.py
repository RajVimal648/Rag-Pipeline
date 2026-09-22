import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from azure.core.exceptions import ResourceNotFoundError
from fastapi.testclient import TestClient
from openai import NotFoundError

from api.main import create_app
from config import Settings
from ingestion.chunker import TokenChunker
from ingestion.document_loader import Document, load_document
from ingestion.embedder import Embedder, create_openai_client
from ingestion.ingest import ingest_path, main as ingest_main
from ingestion.indexer import SearchIndexer, build_index
from query.llm_client import AnswerDraft, LLMClient
from query.rag_service import NO_EVIDENCE, QueryResponse, RAGService
from query.retriever import RetrievedChunk, Retriever


class ChunkerTests(unittest.TestCase):
    def test_size_overlap_and_stable_ids(self):
        chunker = TokenChunker(32, 8)
        document = Document("document", "example.txt", "example", "A useful sentence about retrieval. " * 100)
        chunks = chunker.split(document)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks, chunker.split(document))
        for chunk in chunks:
            self.assertLessEqual(len(chunker.encoding.encode(chunk.content)), 32)
        self.assertEqual(chunker.encoding.encode(chunks[0].content)[-8:], chunker.encoding.encode(chunks[1].content)[:8])

    def test_unicode_is_preserved_without_overlap(self):
        chunker = TokenChunker(16, 0)
        text = "\u4f60\u597d\U0001f30d\u00e9\u0645\u0631\u062d\u0628\u0627 " * 100
        chunks = chunker.split(Document("document", "unicode.txt", "unicode", text))
        self.assertEqual("".join(chunk.content for chunk in chunks), text)
        self.assertTrue(all(len(chunker.encoding.encode(chunk.content)) <= 16 for chunk in chunks))

    def test_empty_and_special_token_text(self):
        chunker = TokenChunker()
        self.assertEqual(chunker.split(Document("document", "empty", "empty", "  ")), [])
        chunks = chunker.split(Document("document", "special", "special", "literal <|endoftext|> text"))
        self.assertEqual(chunks[0].content, "literal <|endoftext|> text")

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            TokenChunker(32, 32)
        with self.assertRaises(ValueError):
            Settings(chunk_size_tokens=32, chunk_overlap_tokens=32)


class AzureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(foundry_embedding_deployment="test", foundry_embedding_dimensions=2, embedding_batch_size=2)

    def test_embedding_batch_order_and_dimensions(self):
        client = MagicMock()
        client.embeddings.create.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[3.0, 4.0]), SimpleNamespace(index=0, embedding=[1.0, 2.0])]),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[5.0, 6.0])]),
        ]
        embedder = Embedder(self.settings, client)
        self.assertEqual(embedder.embed(["one", "two", "three"]), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(client.embeddings.create.call_count, 2)
        client.embeddings.create.side_effect = None
        client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])
        with self.assertRaises(ValueError):
            embedder.embed(["bad dimensions"])

    def test_openai_endpoint_formats(self):
        for suffix in ("", "/openai/v1", "/openai/v1/"):
            settings = Settings(foundry_endpoint="https://example.openai.azure.com" + suffix, foundry_api_key="test")
            with create_openai_client(settings, MagicMock()) as client:
                expected = "/openai/v1/" if suffix else "/openai/"
                self.assertTrue(str(client.base_url).endswith(expected))
        with self.assertRaises(ValueError):
            Settings(foundry_endpoint="https://example.services.ai.azure.com/api/projects/project")

    def test_index_schema_and_dimension_mismatch(self):
        index_client = MagicMock()
        index_client.get_index.return_value = build_index(self.settings)
        indexer = SearchIndexer(self.settings, MagicMock(), index_client)
        indexer.ensure_index()
        index_client.create_index.assert_not_called()
        index_client.get_index.return_value.fields[-1].vector_search_dimensions = 3
        with self.assertRaises(ValueError):
            indexer.ensure_index()

    def test_replace_deletes_stale_only_after_success(self):
        client = MagicMock()
        client.search.return_value = [{"id": "stale"}]
        client.upload_documents.return_value = [SimpleNamespace(succeeded=False)]
        chunk = TokenChunker().split(Document("document", "file", "file", "Some text"))[0]
        indexer = SearchIndexer(self.settings, client, MagicMock())
        with self.assertRaises(RuntimeError):
            indexer.replace_document([chunk], [[1.0, 2.0]])
        client.delete_documents.assert_not_called()
        client.upload_documents.return_value = [SimpleNamespace(succeeded=True)]
        client.delete_documents.return_value = [SimpleNamespace(succeeded=True)]
        self.assertEqual(indexer.replace_document([chunk], [[1.0, 2.0]]), 1)
        client.delete_documents.assert_called_once_with(documents=[{"id": "stale"}])

    def test_hybrid_query_escapes_filter_and_deduplicates(self):
        client = MagicMock()
        result = {"id": "one", "source": "it's.txt", "title": "title", "content": "text", "page": 1, "@search.score": 0.02}
        client.search.return_value = [result, result]
        embedder = MagicMock()
        embedder.embed.return_value = [[1.0, 2.0]]
        chunks = Retriever(self.settings, client, embedder).retrieve("question", source="it's.txt")
        self.assertEqual(len(chunks), 1)
        options = client.search.call_args.kwargs
        self.assertEqual(options["search_text"], "question")
        self.assertEqual(options["filter"], "source eq 'it''s.txt'")
        self.assertEqual(options["vector_queries"][0].vector, [1, 2])


class DocumentLoadingTests(unittest.TestCase):
    def test_local_text_and_invalid_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("\ufeffLocal knowledge", encoding="utf-8")
            document = load_document(path, Settings(), source="collection/notes.txt")[0]
            self.assertEqual(document.text, "Local knowledge")
            self.assertEqual(document.source, "collection/notes.txt")
            identity = document.document_id
            path.write_text("Updated knowledge", encoding="utf-8")
            self.assertEqual(load_document(path, Settings(), source=document.source)[0].document_id, identity)
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_document(path, Settings())
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                load_document(path, Settings(max_file_size_mb=1))

    @patch("ingestion.document_loader.DocumentIntelligenceClient")
    def test_pdf_page_spans_use_unicode_code_points(self, client_class):
        text = "\U0001f30d First page. Second page."
        offset = text.index("Second")
        result = SimpleNamespace(content=text, pages=[
            SimpleNamespace(page_number=1, spans=[SimpleNamespace(offset=0, length=offset)]),
            SimpleNamespace(page_number=2, spans=[SimpleNamespace(offset=offset, length=len(text) - offset)]),
        ])
        client = client_class.return_value.__enter__.return_value
        client.begin_analyze_document.return_value.result.return_value = result
        with TemporaryDirectory() as directory:
            path = Path(directory) / "file.pdf"
            path.write_bytes(b"pdf fixture")
            documents = load_document(path, Settings(doc_intelligence_endpoint="https://example.cognitiveservices.azure.com", doc_intelligence_key="test"))
        self.assertEqual([document.page for document in documents], [1, 2])
        self.assertEqual("".join(document.text for document in documents), text)
        self.assertEqual(client.begin_analyze_document.call_args.kwargs["string_index_type"], "unicodeCodePoint")

    def test_ingestion_missing_path_reports_local_error(self):
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing"
            with patch("sys.argv", ["ingest", str(missing_path)]), \
                    patch("ingestion.ingest.ingest_path") as ingest_mock, \
                    self.assertLogs("ingestion.ingest", level="ERROR") as logs:
                self.assertEqual(ingest_main(), 1)
            ingest_mock.assert_not_called()
            self.assertIn("Input path does not exist", logs.output[0])
            self.assertIn(str(missing_path), logs.output[0])
            self.assertNotIn("service access", logs.output[0])

    def test_ingestion_orchestrates_real_text_chunks(self):
        settings = Settings(
            foundry_endpoint="https://example.openai.azure.com", foundry_api_key="test",
            foundry_embedding_deployment="test", foundry_embedding_dimensions=2,
            search_endpoint="https://example.search.windows.net", search_api_key="test",
        )
        with TemporaryDirectory() as directory, \
                patch("ingestion.ingest.get_settings", return_value=settings), \
                patch("ingestion.ingest.DefaultAzureCredential"), \
                patch("ingestion.ingest.create_openai_client") as openai_factory, \
                patch("ingestion.ingest.SearchClient") as search_factory, \
                patch("ingestion.ingest.SearchIndexClient") as index_factory:
            (Path(directory) / "notes.txt").write_text("RAG retrieves supporting documents.", encoding="utf-8")
            openai_client = openai_factory.return_value.__enter__.return_value
            openai_client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])])
            search_client = search_factory.return_value.__enter__.return_value
            search_client.search.return_value = []
            search_client.upload_documents.return_value = [SimpleNamespace(succeeded=True)]
            index_factory.return_value.__enter__.return_value.get_index.return_value = build_index(settings)
            report = ingest_path(Path(directory), "knowledge")
        self.assertEqual(report, {"documents": 1, "chunks": 1, "failed": []})
        payload = search_client.upload_documents.call_args.kwargs["documents"][0]
        self.assertEqual(payload["source"], "knowledge/notes.txt")
        self.assertEqual(payload["content_vector"], [1.0, 2.0])


class LLMTests(unittest.TestCase):
    def test_v1_responses_answer_and_invalid_output(self):
        client = MagicMock()
        response = SimpleNamespace(status="completed", output_text='{"answer":"Evidence [1]", "source_ids":[1]}')
        client.responses.create.return_value = response
        llm = LLMClient(Settings(foundry_endpoint="https://example.services.ai.azure.com/openai/v1/", foundry_chat_deployment="test"), client)
        self.assertEqual(llm.generate("Question", "Context").source_ids, [1])
        options = client.responses.create.call_args.kwargs
        self.assertEqual(options["text"], {"format": {"type": "json_object"}})
        self.assertEqual(options["model"], "test")
        self.assertEqual(options["max_output_tokens"], llm.settings.max_answer_tokens)
        self.assertFalse(options["store"])
        self.assertIn("untrusted", options["instructions"])
        self.assertIn("Context", options["input"])
        client.chat.completions.create.assert_not_called()
        for status, content in [("incomplete", response.output_text), ("failed", response.output_text), ("completed", "")]:
            with self.subTest(status=status, content=content):
                response.status, response.output_text = status, content
                with self.assertRaises(RuntimeError):
                    llm.generate("Question", "Context")
        response.status, response.output_text = "completed", "not JSON"
        with self.assertRaises(ValueError):
            llm.generate("Question", "Context")

    def test_structured_answer_and_incomplete_output(self):
        client = MagicMock()
        choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"answer":"Evidence [1]", "source_ids":[1]}'))
        client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
        llm = LLMClient(Settings(foundry_chat_deployment="test"), client)
        self.assertEqual(llm.generate("Question", "Context").source_ids, [1])
        options = client.chat.completions.create.call_args.kwargs
        self.assertEqual(options["response_format"], {"type": "json_object"})
        self.assertIn("untrusted", options["messages"][0]["content"])
        choice.finish_reason = "length"
        with self.assertRaises(RuntimeError):
            llm.generate("Question", "Context")


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.retriever = MagicMock()
        self.llm = MagicMock()
        self.service = RAGService(Settings(context_max_tokens=128), self.retriever, self.llm)
        self.chunk = RetrievedChunk("one", "guide.txt", "Guide", "Retrieval finds relevant documents. " * 100, 2, 0.02)

    def test_no_sources_skips_model(self):
        self.retriever.retrieve.return_value = []
        self.assertEqual(self.service.query("Question").answer, NO_EVIDENCE)
        self.llm.generate.assert_not_called()

    def test_context_budget_and_citations(self):
        context, sources = self.service.build_context([self.chunk])
        self.assertLessEqual(len(self.service.encoding.encode(context)), 128)
        self.assertEqual(len(sources), 1)
        self.assertLess(len(sources[0].content), len(self.chunk.content))
        self.retriever.retrieve.return_value = [self.chunk]
        self.llm.generate.return_value = AnswerDraft(answer="Retrieval finds documents [1].", source_ids=[1])
        response = self.service.query("What is retrieval?")
        self.assertEqual(response.sources[0].page, 2)
        self.assertEqual(response.sources[0].content, sources[0].content)

    def test_invalid_or_missing_citations_fall_back(self):
        self.retriever.retrieve.return_value = [self.chunk]
        for answer, identifiers in [("Unsupported [99]", [99]), ("Uncited", [1]), ("No evidence", []), ("Wrong [2]", [1])]:
            self.llm.generate.return_value = AnswerDraft(answer=answer, source_ids=identifiers)
            response = self.service.query("Question")
            self.assertEqual(response.answer, NO_EVIDENCE)
            self.assertEqual(response.sources, [])


class APITests(unittest.TestCase):
    def test_swagger_example_searches_all_sources(self):
        service = MagicMock()
        service.query.return_value = QueryResponse(answer=NO_EVIDENCE, sources=[])
        with TestClient(create_app(Settings(api_key="test-key"), service)) as client:
            schema = client.get("/openapi.json").json()["components"]["schemas"]["QueryRequest"]
            example = schema["examples"][0]
            self.assertNotIn("source", example)
            response = client.post("/query", headers={"X-API-Key": "test-key"}, json=example)
            self.assertEqual(response.status_code, 200)
            service.query.assert_called_once_with(example["question"], example["top_k"], None)

    def test_auth_validation_and_query(self):
        service = MagicMock()
        service.query.return_value = QueryResponse(answer=NO_EVIDENCE, sources=[])
        with TestClient(create_app(Settings(api_key="test-key"), service)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.post("/query", json={"question": "What?"}).status_code, 401)
            headers = {"X-API-Key": "test-key"}
            for payload in [{"question": "  "}, {"question": "What?", "top_k": 0}, {"question": "What?", "extra": True}]:
                self.assertEqual(client.post("/query", headers=headers, json=payload).status_code, 422)
            response = client.post("/query", headers=headers, json={"question": "What?", "top_k": 3})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sources"], [])
            service.query.assert_called_once_with("What?", 3, None)

    def test_upstream_failure_does_not_leak_details(self):
        service = MagicMock()
        service.query.side_effect = RuntimeError("secret credential in upstream exception")
        with TestClient(create_app(Settings(), service)) as client:
            response = client.post("/query", json={"question": "What?"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("secret", response.text)

    def test_missing_upstream_resources_have_safe_actionable_errors(self):
        response = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
        failures = [
            (NotFoundError("secret upstream details", response=response, body={"code": "DeploymentNotFound"}), "FOUNDRY_EMBEDDING_DEPLOYMENT"),
            (ResourceNotFoundError("secret upstream details"), "SEARCH_INDEX_NAME"),
        ]
        for error, setting in failures:
            with self.subTest(setting=setting):
                service = MagicMock()
                service.query.side_effect = error
                with TestClient(create_app(Settings(), service)) as client:
                    result = client.post("/query", json={"question": "What?"})
                    self.assertEqual(result.status_code, 502)
                    self.assertIn(setting, result.json()["detail"])
                    self.assertNotIn("secret", result.text)

    def test_unconfigured_api_still_has_health_endpoint(self):
        with TestClient(create_app(Settings())) as client:
            self.assertFalse(client.get("/health").json()["configured"])
            self.assertEqual(client.post("/query", json={"question": "What?"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()