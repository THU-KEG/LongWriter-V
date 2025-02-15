import json
from anthropic import Anthropic

class Claude_Interface:
    with open("config.json", "r") as f:
        config = json.load(f)
    client = Anthropic(
        api_key=config["claude_api_key"],
        base_url=config["claude_base_url"]
    )

    @classmethod
    def call(cls, model, messages, **kwargs):
        messags = cls.client.messages.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return messags.content[0].text
    