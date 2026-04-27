import resend

from linkedin_scrapper.config import Settings


def configure_resend(settings: Settings) -> None:
    if settings.resend_api_key is None:
        raise RuntimeError("RESEND_API_KEY is required to configure Resend.")

    resend.api_key = settings.resend_api_key.get_secret_value()
