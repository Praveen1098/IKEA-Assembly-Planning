# pyright: reportCallIssue=false
# pyright: reportArgumentType=false

import os
from dataclasses import dataclass
from typing import Literal

import google.oauth2.credentials
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

# ChatVertexAI is the correct class for enterprise/proxy Gemini endpoints (Vertex AI-compatible).
# langchain-google-genai's ChatGoogleGenerativeAI hardcodes /v1beta/models/... paths and cannot
# use Fujitsu's endpoint format. The deprecation warning is suppressed at call site via
# catch_warnings(); ChatVertexAI will not be removed until LangChain 4.0.
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

# Models that do not accept temperature or max_tokens in API requests
_NO_SAMPLING_PARAMS = {"GPT-5.1"}


def get_bind_kwargs(llm, temperature: float | None = None, max_tokens: int | None = None) -> dict:
    """Return .bind() kwargs appropriate for the given LLM.

    For GPT-5.1, temperature is dropped and max_tokens is remapped to max_completion_tokens.
    """
    if getattr(llm, "model_name", None) in _NO_SAMPLING_PARAMS:
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        return kwargs
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def get_model_config(config):
    engine = config["model"]["engine"]
    name = config["model"]["name"]

    if engine == "Gemini":
        return Gemini(
            name=name,
            endpoint=config["endpoint"][name],
            max_retries=config["model"]["max_retries"],
            temperature=config["model"]["temperature"],
            max_tokens=config["model"]["max_tokens"],
            thinking_budget=config["model"].get("thinking_budget", -1),
        )
    elif engine == "GPT":
        return GPT(
            name=name,
            endpoint=config["endpoint"][name],
            max_retries=config["model"]["max_retries"],
            temperature=config["model"].get("temperature", 0),
            max_tokens=config["model"]["max_tokens"],
            api_version=config["model"].get("api_version", ""),
        )
    elif engine == "Claude":
        return Claude(
            name=name,
            max_retries=config["model"]["max_retries"],
            temperature=config["model"]["temperature"],
            max_tokens=config["model"]["max_tokens"],
        )
    elif engine == "Ollama":
        num_gpu = config["model"]["num_gpu"]
        return Ollama(name=name, num_gpu=num_gpu, temperature=config["model"]["temperature"])
    else:
        raise ValueError(f"Unknown model engine: {engine}")


@dataclass
class Gemini:
    name: Literal["GEMINI-2.5-pro", "GEMINI-2.5-flash"]
    endpoint: str
    max_retries: int = 1
    temperature: float = 0
    max_tokens: int = 10000
    thinking_budget: int = -1  # -1 = no override; 0 = disable thinking


@dataclass
class Claude:
    name: Literal["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    max_retries: int = 1
    temperature: float = 0
    max_tokens: int = 10000


@dataclass
class GPT:
    name: Literal["GPT-4o", "GPT-4o-mini", "GPT-5.1"]
    endpoint: str
    max_retries: int = 1
    temperature: float = 0
    max_tokens: int = 10000
    api_version: str = ""


@dataclass
class Ollama:
    name: Literal["gemma3n:e4b", "deepseek-r1:70b", "deepseek-r1:32b", "deepseek-r1:1.5b"]
    temperature: float
    num_gpu: int = 1  # Default to 0 GPUs


class LLM:
    @staticmethod
    def get_model(model):
        if isinstance(model, Gemini):
            import warnings as _warnings
            mkwargs = {}
            if model.thinking_budget >= 0:
                mkwargs["thinking_config"] = {"thinking_budget": model.thinking_budget}
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=r".*ChatGoogleGenerativeAI.*")
                return ChatVertexAI(
                    model=model.name,
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                    max_retries=model.max_retries,
                    credentials=google.oauth2.credentials.Credentials(os.getenv("API_KEY")),
                    api_transport="rest",
                    api_endpoint=model.endpoint,  # type: ignore
                    project="bpmn",
                    model_kwargs=mkwargs,
                )

        if isinstance(model, GPT):
            gpt_kwargs = dict(
                model=model.name,
                max_retries=model.max_retries,
                azure_endpoint=model.endpoint,  # type: ignore
                api_key=SecretStr(os.getenv("API_KEY")),  # type: ignore
                api_version=model.api_version,
            )
            # GPT-5.1 uses max_completion_tokens instead of max_tokens, and has no temperature
            if model.name in _NO_SAMPLING_PARAMS:
                gpt_kwargs["max_completion_tokens"] = model.max_tokens
            else:
                gpt_kwargs["temperature"] = model.temperature
                gpt_kwargs["max_tokens"] = model.max_tokens
            return AzureChatOpenAI(**gpt_kwargs)

        if isinstance(model, Claude):
            return ChatAnthropic(
                model=model.name,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                max_retries=model.max_retries,
                timeout=None,
                api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
            )

        if isinstance(model, Ollama):
            return OllamaLLM(
                model=model.name,
                num_gpu=model.num_gpu,
                temperature=model.temperature,
            )

        raise ValueError(f"Unsupported model: {model.name}")


def load_llm_from_recipe(recipe_path, model_id: str):
    """Load a LangChain LLM from recipe.yaml by model id.

    Args:
        recipe_path: Path to recipe.yaml (str or Path).
        model_id: The 'id' field in recipe.yaml's models list, e.g. 'gpt4o', 'gemini-pro'.

    Returns:
        A LangChain chat model instance (AzureChatOpenAI, ChatVertexAI, etc.)
    """
    import yaml
    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    models_by_id = {m["id"]: m for m in recipe["models"]}
    if model_id not in models_by_id:
        raise ValueError(
            f"Model id '{model_id}' not found in recipe. "
            f"Available: {list(models_by_id.keys())}"
        )
    config = {"endpoint": recipe["endpoint"], "model": models_by_id[model_id]}
    return LLM.get_model(get_model_config(config))
