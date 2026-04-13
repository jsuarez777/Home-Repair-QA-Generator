#!/usr/bin/env python3
from openai import OpenAI
import os, sys
from typing import Optional

class MyOpenAIClient:
    """
    Factory to produce a configured OpenAI client.
    Reads OPENAI_API_KEY and OPENAI_ORG from the environment if not provided.
    """
    def __init__(self, model: str, api_key: Optional[str] = None):
        if not model:
            print("Error: 'model' is required to initialize MyOpenAIClient.", file=sys.stderr)
            raise ValueError("'model' parameter is required.")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None
        self.model: str = model

    def get_client(self) ->OpenAI:
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def query(self, *, input: str | list, model: Optional[str] = None, **kwargs):
        """Convenience wrapper that uses the configured default model unless overridden.

        Parameters:
            input: The prompt string or messages to send to the responses API.
            model: Optional model name to override the factory default.
            **kwargs: Passed through to `client.responses.create`.
        """
        client = self.get_client()
        model_to_use = model or self.model
        if not model_to_use:
            raise ValueError("No model specified: pass `model=` or set `model` when constructing MyOpenAIClient.")
        return client.responses.create(model=model_to_use, input=input, **kwargs)