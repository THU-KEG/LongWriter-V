import json
from anthropic import Anthropic
from config import config

class Claude_Interface:

    client = Anthropic(
        api_key=config.claude_api_key,
        base_url=config.claude_base_url
    )

    @classmethod
    def call(cls, model, messages, **kwargs):
        messags = cls.client.messages.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return messags.content[0].text
    