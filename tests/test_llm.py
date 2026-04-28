from linkedin_scrapper.config import Settings
from linkedin_scrapper.services import llm


class StubChatOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_cv_parser_chat_model_uses_dedicated_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setattr(llm, "ChatOpenAI", StubChatOpenAI)
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-mini",
        OPENAI_CV_PARSER_MODEL="gpt-4.1",
        OPENAI_CV_PARSER_REASONING_EFFORT="high",
    )

    chat_model = llm.build_cv_parser_chat_model(settings)

    assert chat_model.kwargs["model"] == "gpt-4.1"
    assert chat_model.kwargs["reasoning_effort"] == "high"


def test_default_chat_model_does_not_use_cv_parser_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setattr(llm, "ChatOpenAI", StubChatOpenAI)
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-mini",
        OPENAI_CV_PARSER_MODEL="gpt-4.1",
        OPENAI_CV_PARSER_REASONING_EFFORT="high",
    )

    chat_model = llm.build_chat_model(settings)

    assert chat_model.kwargs["model"] == "gpt-4.1-mini"
    assert "reasoning_effort" not in chat_model.kwargs


def test_cv_parser_chat_model_omits_blank_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setattr(llm, "ChatOpenAI", StubChatOpenAI)
    settings = Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_CV_PARSER_MODEL="gpt-4.1",
        OPENAI_CV_PARSER_REASONING_EFFORT=" ",
    )

    chat_model = llm.build_cv_parser_chat_model(settings)

    assert "reasoning_effort" not in chat_model.kwargs
