from dataclasses import dataclass


@dataclass(frozen=True)
class LinkedInLocation:
    label: str
    geo_id: str


LOCAL_SEARCH_LOCATION = LinkedInLocation(
    label="Montpellier, Occitanie, France",
    geo_id="106719766",
)
REMOTE_SEARCH_LOCATION = LinkedInLocation(
    label="France",
    geo_id="105015875",
)
