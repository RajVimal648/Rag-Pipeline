import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from openai import APIError, APITimeoutError, NotFoundError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from config import Settings, get_settings
from query.rag_service import QueryResponse, RAGService, create_rag_service


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class QueryRequest(BaseModel):
	model_config = ConfigDict(
		extra="forbid", str_strip_whitespace=True,
		json_schema_extra={"examples": [{"question": "Who was India's Prime Minister on 10 June 2024?", "top_k": 5}]},
	)
	question: str = Field(min_length=1, max_length=4000)
	top_k: int | None = Field(default=None, ge=1, le=20)
	source: str | None = Field(default=None, min_length=1, max_length=1024, description="Optional exact source identifier, such as knowledge/test.txt. Omit to search all documents.")

	@field_validator("question")
	@classmethod
	def validate_question(cls, value: str) -> str:
		if not value.strip():
			raise ValueError("Question must not be blank")
		return value


def get_service(request: Request) -> RAGService:
	service = request.app.state.service
	if service is None:
		raise HTTPException(status_code=503, detail="RAG service is not configured")
	return service


def authorize(request: Request, supplied: Annotated[str | None, Depends(api_key_header)]) -> None:
	expected = request.app.state.settings.api_key.get_secret_value()
	if expected and (supplied is None or not secrets.compare_digest(supplied.encode(), expected.encode())):
		raise HTTPException(status_code=401, detail="Invalid or missing API key")


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
	@asynccontextmanager
	async def lifespan(app: FastAPI):
		app.state.settings = settings or get_settings()
		app.state.service = service
		if not app.state.settings.api_key.get_secret_value():
			logger.warning("API_KEY is unset; bind to loopback only. Authentication is required before sharing this API.")
		if service is not None:
			yield
			return
		manager = create_rag_service(app.state.settings)
		try:
			app.state.service = await run_in_threadpool(manager.__enter__)
		except ValueError:
			logger.error("RAG configuration is incomplete or invalid; /query will return 503")
			yield
			return
		try:
			yield
		finally:
			await run_in_threadpool(manager.__exit__, None, None, None)
			app.state.service = None

	app = FastAPI(title="Document RAG API", version="1.0.0", lifespan=lifespan)

	@app.get("/health")
	def health():
		return {"status": "ok", "configured": app.state.service is not None}

	@app.post("/query", response_model=QueryResponse, dependencies=[Depends(authorize)])
	def query(payload: QueryRequest, rag: Annotated[RAGService, Depends(get_service)]):
		try:
			return rag.query(payload.question, payload.top_k, payload.source)
		except RateLimitError:
			raise HTTPException(status_code=429, detail="Model service rate limit reached; retry later", headers={"Retry-After": "30"}) from None
		except APITimeoutError:
			raise HTTPException(status_code=504, detail="Model service timed out") from None
		except NotFoundError:
			logger.warning("Model endpoint or deployment was not found (404)")
			raise HTTPException(status_code=502, detail="Model endpoint or deployment not found; verify FOUNDRY_ENDPOINT, FOUNDRY_EMBEDDING_DEPLOYMENT and FOUNDRY_CHAT_DEPLOYMENT, then restart the API") from None
		except ResourceNotFoundError:
			logger.warning("Search index was not found (404)")
			raise HTTPException(status_code=502, detail="Search index not found; verify SEARCH_ENDPOINT and SEARCH_INDEX_NAME, then run document ingestion to create and populate the index") from None
		except (APIError, HttpResponseError):
			logger.warning("An upstream Azure request failed")
			raise HTTPException(status_code=502, detail="Upstream service failed; check deployments, index and permissions") from None
		except Exception as error:
			logger.error("RAG request failed (%s)", type(error).__name__)
			raise HTTPException(status_code=502, detail="The RAG pipeline could not complete this request") from None

	return app


app = create_app()
