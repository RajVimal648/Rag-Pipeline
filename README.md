# Document Intelligence RAG Pipeline

A standalone Python backend application for asking questions about your own documents and receiving answers with supporting source excerpts. It uses Azure OpenAI deployments in Microsoft Foundry, Azure AI Search, Azure Document Intelligence, and FastAPI.

This is a general-purpose document question-answering project, not a library-management system or a reusable Python library package. It can work with project documentation, business reports, policies, technical manuals, and other supported documents. The subject matter comes from the documents you ingest; it is not tied to books, catalogs, or any particular industry.

## What the Application Does

- Ingests local documents through a command-line workflow, extracting text or using OCR for PDFs and images.
- Splits text into token-sized chunks and stores their embeddings in Azure AI Search.
- Retrieves relevant excerpts using hybrid keyword and vector search.
- Exposes a FastAPI endpoint that returns document-grounded answers with citations.
- Returns an insufficient-evidence answer when no usable sources are found.

The current interface is a backend API with Swagger for interactive requests. A dedicated chat frontend, user-account management, and document-upload API are not included.

## Architecture

```text
Local documents -> text/OCR -> token chunks -> embeddings -> Azure AI Search
Question -> embedding -> hybrid keyword + vector search -> bounded context
				 -> Azure OpenAI -> answer with inline citations and source excerpts
```

### Project Structure

| Path | Responsibility |
| --- | --- |
| `config.py` | Environment settings and configuration validation |
| `ingestion/` | Document loading, chunking, embeddings, indexing, and ingestion CLI |
| `query/` | Retrieval, model requests, context limits, and citation validation |
| `api/main.py` | FastAPI application, request validation, and optional API-key authentication |
| `tests/test_pipeline.py` | Automated tests using mocked Azure clients |

## Prerequisites

- Python 3.11 or newer. Local tests have been run on Python 3.14.
- An Azure AI Search service with vector search support.
- An Azure OpenAI chat deployment supporting Responses API JSON mode for v1 endpoints, or Chat Completions JSON mode and `max_completion_tokens` for resource-root endpoints.
- An embedding deployment, preferably `text-embedding-3-small` (1536 dimensions) or `text-embedding-3-large` (3072 dimensions by default).
- Document Intelligence only if ingesting PDFs or images. Plain text does not use OCR.

This repository does not provision Azure services or deploy models. Ingestion and queries use billable Azure APIs.

## Setup

Run commands from the repository root, not from `api/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The existing `.env` is preserved. Use `.env.example` as the reference for available settings. Shell environment variables override `.env`, which is always loaded relative to `config.py`, regardless of the working directory.

Required values:

| Setting | Meaning |
| --- | --- |
| `FOUNDRY_ENDPOINT` | Azure OpenAI resource root or `https://name.openai.azure.com/openai/v1/`; not a Foundry project URL |
| `FOUNDRY_CHAT_DEPLOYMENT` | Actual deployed chat model deployment name |
| `FOUNDRY_EMBEDDING_DEPLOYMENT` | Actual embedding deployment name, shared by ingestion and queries |
| `FOUNDRY_EMBEDDING_DIMENSIONS` | Must match generated vectors and the search index |
| `SEARCH_ENDPOINT` | Search service root, such as `https://name.search.windows.net/` |
| `SEARCH_INDEX_NAME` | Dedicated index name; created by ingestion if absent |
| `DOC_INTELLIGENCE_ENDPOINT` | Required for PDFs/images only |

`FOUNDRY_EMBEDDING_SEND_DIMENSIONS=true` requests the configured dimensions from embedding-3 models. For older models such as `text-embedding-ada-002`, set it to `false` and use that model's native dimension count. Changing embedding models requires a new index and re-ingestion even if dimensions are unchanged.

For `/openai/v1` endpoints the standard OpenAI client uses the v1 API without a dated API version. Answers use the Responses API with JSON output and `store=False`. Resource-root endpoints use `AzureOpenAI` with `FOUNDRY_API_VERSION` and Chat Completions. Set the base URL ending in `/openai/v1`, not the operation URL ending in `/responses`. Both `openai.azure.com` and `services.ai.azure.com` resource hosts are supported.

The chat deployment cannot replace the embedding deployment. If embeddings return `DeploymentNotFound`, deploy an embedding model in the configured resource or set `FOUNDRY_EMBEDDING_DEPLOYMENT` to its existing deployment name. Match `FOUNDRY_EMBEDDING_DIMENSIONS` to the embedding model, then rerun ingestion and restart the API after changing configuration.

### Authentication

If service keys are supplied in `FOUNDRY_API_KEY`, `SEARCH_API_KEY`, or `DOC_INTELLIGENCE_KEY`, the corresponding client uses them. Otherwise clients use `DefaultAzureCredential`, supporting managed identity and developer credentials. Never commit keys.

For passwordless access, grant the calling identity appropriate roles at the resource scope: Cognitive Services OpenAI User for inference; Search Index Data Reader for querying; Search Index Data Contributor and Search Service Contributor for ingestion/index creation; Cognitive Services User for Document Intelligence. Search must allow role-based access. The query API does not need index-management permissions.

`API_KEY` is a separate application secret, not an Azure key. When set, `/query` requires the `X-API-Key` header. When unset, use loopback only. For a shared deployment, add HTTPS, proper identity-based authorization, rate limits and request-body limits at the gateway. A source filter is not a tenant access-control boundary.

## Ingest Documents

Supported formats: UTF-8 `.txt`, `.md`, `.csv`; PDFs and `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` via Document Intelligence. CSV is indexed as text, not treated as a relational table. PDF/image page numbers are retained for citations.

```powershell
python -m ingestion.ingest .\data --source-prefix knowledge
python -m ingestion.ingest .\data\orion-launch-plan.pdf --source-prefix knowledge
```

The CLI reports document/chunk counts and failures as JSON, with a nonzero exit code if anything fails. It bounds file size, batches embedding/index requests, validates embedding dimensions, and refuses incompatible existing index schemas. It never deletes or recreates an existing index automatically.

Sources are the prefix plus the path relative to the ingestion root. Use a stable root and prefix across runs. For example, ingesting a nested file alone requires its parent path in the prefix to preserve its source identity. Prefixes distinguish separate collections with matching filenames.

Re-ingestion upserts deterministic chunk IDs, then removes obsolete chunks for that source only after all new uploads succeed. Upload failures retain old chunks, but Azure Search has no transaction across batches: partial uploads may be visible until a successful rerun. Writes are eventually consistent. Do not run concurrent ingestions for the same source. Files removed from disk are not automatically deleted from the index; use a fresh index for a full corpus rebuild.

## Run the API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- `GET /health`: process liveness and local client-configuration status, not a live Azure connectivity check.
- `POST /query`: retrieve context and generate a cited answer.

```powershell
$body = @{ question = "What must Project Orion complete before its pilot launch?"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType 'application/json' -Body $body
```

When `API_KEY` is configured, add `-Headers @{ 'X-API-Key' = '<your-application-key>' }`. Optionally include `source` in the request to restrict retrieval to an exact indexed source, such as `knowledge/orion-launch-plan.pdf`. Omit `source` to search all indexed documents; a placeholder such as `"string"` is treated as a real filter, not as an unset value.

Illustrative response for a fictional Project Orion launch plan, not a live API result. The example PDF is not bundled with this repository; ingest your own documents and adapt the question. Chunk IDs and scores are generated at runtime.

```json
{
	"answer": "Before its pilot launch, Project Orion must pass security review, validate the data migration, and assign an on-call owner for each service [1].",
	"sources": [
		{
			"number": 1,
			"chunk_id": "stable-chunk-hash",
			"source": "knowledge/orion-launch-plan.pdf",
			"title": "orion-launch-plan.pdf",
			"page": 3,
			"content": "Pilot launch gate: security review must pass, data migration must be validated, and every service must have an assigned on-call owner before Project Orion can launch.",
			"score": 0.032
		}
	]
}
```

The context budget includes serialized source metadata and excerpts. It does not include the question, system prompt, model-specific message overhead or output tokens. Configure it below the deployed model's context window, allowing room for those. Token counting uses `cl100k_base`; tokenizer differences between models require additional headroom.

If retrieval returns nothing, the model is not called. Missing, unknown or inconsistent citation identifiers produce an insufficient-evidence answer. Citations prove which excerpts were referenced, not that every model claim is correct. Prompts treat retrieved text as untrusted, but prompts alone cannot eliminate prompt injection or hallucination; evaluate answers on representative documents before production use. Hybrid RRF scores are ranking signals, not calibrated confidence probabilities.

Upstream failures return sanitized errors: 429 for model rate limits, 504 for model timeouts, 502 for other pipeline failures, and 503 when required configuration is missing. Invalid request bodies return 422. Index writes remain CLI-only; the API does not accept arbitrary server-side file paths.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests exercise token limits/Unicode, loading and page spans, embedding batching, index validation, stale-chunk cleanup, hybrid queries, context budgets, citation checks, model output handling, API validation/authentication, and ingestion orchestration. Azure clients are mocked: this does not verify real deployment access, model capabilities, OCR quality or live search behavior. The tokenizer may download its public encoding cache on first use.
