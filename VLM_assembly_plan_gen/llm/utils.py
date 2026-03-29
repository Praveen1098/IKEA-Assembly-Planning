from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate


class TokenUsageHandler(BaseCallbackHandler):
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self._initialized = True

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # Handle OpenAI format
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_tokens += usage.get("total_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)

        # Handle Anthropic format
        elif response.llm_output and "usage" in response.llm_output:
            usage = response.llm_output["usage"]
            self.prompt_tokens += usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("output_tokens", 0)
            self.total_tokens += self.prompt_tokens + self.completion_tokens

        # Handle VertexAI Gemini format - usage_metadata on response object
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens)

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

        # Handle alternative VertexAI format - check generations for usage
        elif hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "generation_info") and generation.generation_info:
                        usage = generation.generation_info.get("usage_metadata")
                        if usage:
                            prompt_tokens = usage.get("prompt_token_count", 0)
                            completion_tokens = usage.get("candidates_token_count", 0)
                            total_tokens = usage.get(
                                "total_token_count", prompt_tokens + completion_tokens
                            )

                            self.prompt_tokens += prompt_tokens
                            self.completion_tokens += completion_tokens
                            self.total_tokens += total_tokens
                            break

    def reset_counters(self):
        """Reset token counters to zero."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_usage_summary(self):
        """Get current token usage summary."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def invoke_multimodal(llm, prompt_text: str, base64_images: list, mime_type: str = "image/jpeg") -> str:
    """Call the LLM with text + optional base64 images.

    Works identically for AzureChatOpenAI (GPT) and ChatVertexAI (Gemini).
    The 'detail' field is intentionally omitted — it is OpenAI-specific and
    causes errors with the Gemini LangChain adapter.

    Args:
        llm: A LangChain chat model instance.
        prompt_text: The text portion of the prompt.
        base64_images: List of base64-encoded image strings (may be empty for text-only calls).
        mime_type: MIME type for all images, e.g. 'image/png' or 'image/jpeg'.

    Returns:
        The model's response as a plain string.
    """
    token_handler = TokenUsageHandler()
    content = [{"type": "text", "text": prompt_text}]
    for b64 in base64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    response = llm.invoke([HumanMessage(content=content)], config={"callbacks": [token_handler]})
    return response.content


def process_with_llm(llm, prompt, **prompt_kwargs) -> str:
    """General method for LLM processing with prompts"""
    token_handler = TokenUsageHandler()

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt["system"]),
            ("human", prompt["user"]),
        ]
    )
    formatted_prompt = chat_prompt.format_messages(**prompt_kwargs)
    return llm.invoke(formatted_prompt, config={"callbacks": [token_handler]}).content
