from sqlalchemy.orm import Session

from linkedin_scrapper.cv_parser import CandidateSkill, ParsedCandidateProfile
from linkedin_scrapper.models import CandidateProfile


def save_candidate_profile(
    session: Session,
    parsed_profile: ParsedCandidateProfile,
) -> CandidateProfile:
    profile = CandidateProfile(
        cv_text=parsed_profile.cv_text,
        target_roles=parsed_profile.target_roles,
        locations=parsed_profile.locations,
        total_years_of_experience=parsed_profile.total_years_of_experience,
        remote_preference=[preference.value for preference in parsed_profile.remote_preference],
        languages_spoken=[language.value for language in parsed_profile.languages_spoken],
        industries_experienced=parsed_profile.industries_experienced,
        skills=[_dump_skill(skill) for skill in parsed_profile.skills],
        seniority=parsed_profile.seniority,
        exclusions=parsed_profile.exclusions,
        profile_payload=parsed_profile.profile_payload,
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)
    return profile


def _dump_skill(skill: CandidateSkill) -> dict:
    return skill.model_dump(mode="json")
