from linkedin_scrapper.search_locations import LOCAL_SEARCH_LOCATION, REMOTE_SEARCH_LOCATION


def test_linkedin_search_locations_define_public_geo_ids() -> None:
    assert LOCAL_SEARCH_LOCATION.label == "Montpellier, Occitanie, France"
    assert LOCAL_SEARCH_LOCATION.geo_id == "106719766"
    assert REMOTE_SEARCH_LOCATION.label == "France"
    assert REMOTE_SEARCH_LOCATION.geo_id == "105015875"
