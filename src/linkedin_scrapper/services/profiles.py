from sqlalchemy.orm import Session

from linkedin_scrapper.cv_parser import ParsedCandidateProfile
from linkedin_scrapper.models import CandidateProfile


def save_candidate_profile(
    session: Session,
    parsed_profile: ParsedCandidateProfile,
) -> CandidateProfile:
    profile = CandidateProfile(
        cv_text=parsed_profile.cv_text,
        target_roles=parsed_profile.target_roles,
        skills=parsed_profile.skills,
        locations=parsed_profile.locations,
        remote_preference=parsed_profile.remote_preference,
        seniority=parsed_profile.seniority,
        exclusions=parsed_profile.exclusions,
        profile_payload=parsed_profile.profile_payload,
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile
