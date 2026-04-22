"""Adapter binding `llm/utils.invoke_multimodal` to the VLMClient protocol.

The free function `llm.utils.invoke_multimodal(llm, prompt, images, mime)` takes
a LangChain LLM instance as its first positional argument. Our `stage2_v2_plan`
module's `VLMClient` protocol expects the LLM to be carried as state and a
two-arg `invoke_multimodal(prompt, images)` to be the call site. This adapter
bridges the two.

Default model: Gemini 2.5 Pro (recipe id `gemini-pro` in
`VLM_assembly_plan_gen/recipe.yaml`). This is the recipe's own
`default_model` field. Pass `model_id="gpt4o"` etc. to override.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llm.model import load_llm_from_recipe
from llm.utils import TokenUsageHandler, invoke_multimodal


DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[1] / "recipe.yaml"
)
DEFAULT_MODEL_ID = "gemini-pro"  # Gemini 2.5 Pro


class RecipeVLMClient:
    """Adapter that satisfies `stage2_v2_plan.VLMClient`.

    Attributes:
        model_id: the recipe id used to load the LLM (e.g. 'gemini-pro').
        llm: the underlying LangChain LLM instance (ChatVertexAI for Gemini).
        mime_type: default MIME for base64 images ('image/png' recommended for
            rendered manual pages; 'image/jpeg' for photos).
        call_count: number of invoke_multimodal calls issued (for telemetry).
    """

    def __init__(
        self,
        recipe_path: str | os.PathLike = DEFAULT_RECIPE_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        mime_type: str = "image/png",
    ):
        self.model_id = model_id
        self.recipe_path = str(recipe_path)
        self.mime_type = mime_type
        self.llm = load_llm_from_recipe(self.recipe_path, self.model_id)
        self.call_count = 0

    def invoke_multimodal(self, prompt: str, base64_images: list[str]) -> str:
        self.call_count += 1
        return invoke_multimodal(
            self.llm,
            prompt_text=prompt,
            base64_images=base64_images,
            mime_type=self.mime_type,
        )

    def token_usage(self) -> dict[str, int]:
        """Return cumulative token usage across all calls (via singleton handler)."""
        return TokenUsageHandler().get_usage_summary()

    def __repr__(self) -> str:
        return (
            f"RecipeVLMClient(model_id={self.model_id!r}, "
            f"calls={self.call_count}, tokens={self.token_usage()})"
        )


def load_default_vlm_client(**kwargs: Any) -> RecipeVLMClient:
    """Convenience factory returning a Gemini 2.5 Pro client.

    Override via kwargs: `model_id="gpt4o"` or `recipe_path=...`.
    """
    return RecipeVLMClient(**kwargs)
