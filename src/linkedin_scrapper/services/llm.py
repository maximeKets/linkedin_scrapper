from langchain_openai import ChatOpenAI

from linkedin_scrapper.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required to build the chat model.")

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def build_cv_parser_chat_model(settings: Settings) -> ChatOpenAI:
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required to build the CV parser chat model.")

    reasoning_effort = settings.openai_cv_parser_reasoning_effort
    model_kwargs = {}
    if reasoning_effort and reasoning_effort.strip():
        model_kwargs["reasoning_effort"] = reasoning_effort.strip()

    return ChatOpenAI(
        model=settings.openai_cv_parser_model,
        api_key=settings.openai_api_key.get_secret_value(),
        **model_kwargs,
    )
