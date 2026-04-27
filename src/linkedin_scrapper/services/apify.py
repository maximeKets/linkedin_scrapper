from apify_client import ApifyClient

from linkedin_scrapper.config import Settings


def build_apify_client(settings: Settings) -> ApifyClient:
    if settings.apify_api_token is None:
        raise RuntimeError("APIFY_API_TOKEN is required to build the Apify client.")

    return ApifyClient(settings.apify_api_token.get_secret_value())
