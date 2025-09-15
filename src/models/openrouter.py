import logging
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI


class OpenRouterModel:
    """
    Wrapper class for calling OpenRouter models with logging, retries, and configs.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "openrouter/auto",
        temperature: float = 0.3,
        max_tokens: int = 512,
        log_level: int = logging.INFO,
        retries: int = 3,
        retry_delay: float = 2.0,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the wrapper.

        Args:
            api_key (str): OpenRouter API key.
            model_name (str): Model name, e.g. "openai/gpt-4o-mini".
            temperature (float): Sampling temperature.
            max_tokens (int): Max tokens for generation.
            log_level (int): Logging level (default INFO).
            retries (int): Number of retries for failed requests.
            retry_delay (float): Seconds between retries.
            base_url (str): API base URL.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_delay = retry_delay

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Setup OpenRouter client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send chat messages to the model.

        Args:
            messages (List[Dict[str, str]]): [{"role": "user", "content": "..."}].
            temperature (float, optional): Override default temperature.
            max_tokens (int, optional): Override default max tokens.

        Returns:
            str: Model's reply.
        """
        attempt = 0
        while attempt < self.retries:
            try:
                self.logger.info(
                    f"Requesting model={self.model_name}, attempt={attempt+1}"
                )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )

                reply = response.choices[0].message.content
                self.logger.info(f"Response received (len={len(reply)} chars)")
                return reply

            except Exception as e:
                attempt += 1
                self.logger.error(f"Error on attempt {attempt}: {e}")
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Use model in completion (non-chat) mode.

        Args:
            prompt (str): Input text.
            temperature (float, optional): Override temperature.
            max_tokens (int, optional): Override max tokens.

        Returns:
            str: Completion text.
        """
        attempt = 0
        while attempt < self.retries:
            try:
                self.logger.info(
                    f"Requesting completion with model={self.model_name}, attempt={attempt+1}"
                )

                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )

                text = response.choices[0].text
                self.logger.info(f"Completion received (len={len(text)} chars)")
                return text

            except Exception as e:
                attempt += 1
                self.logger.error(f"Error on attempt {attempt}: {e}")
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                else:
                    raise
