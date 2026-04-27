from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class SearchRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ApplicationStatus(StrEnum):
    NEW = "new"
    SHORTLISTED = "shortlisted"
    APPLIED = "applied"
    REJECTED = "rejected"
    IGNORED = "ignored"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CandidateProfile(TimestampMixin, Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    cv_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_roles: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    locations: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    remote_preference: Mapped[str | None] = mapped_column(String(50))
    seniority: Mapped[str | None] = mapped_column(String(100))
    exclusions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    profile_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    search_runs: Mapped[list[SearchRun]] = relationship(
        back_populates="candidate_profile",
        cascade="all, delete-orphan",
    )


class SearchRun(TimestampMixin, Base):
    __tablename__ = "search_runs"
    __table_args__ = (
        CheckConstraint(
            "status in ('pending', 'running', 'succeeded', 'failed')",
            name="ck_search_runs_status",
        ),
        Index("ix_search_runs_profile_status", "profile_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("candidate_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(String(255), nullable=False)
    linkedin_url: Mapped[str] = mapped_column(Text, nullable=False)
    actor_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30),
        default=SearchRunStatus.PENDING.value,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)

    candidate_profile: Mapped[CandidateProfile] = relationship(back_populates="search_runs")
    job_links: Mapped[list[SearchRunJob]] = relationship(
        back_populates="search_run",
        cascade="all, delete-orphan",
    )


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_jobs_external_id"),
        UniqueConstraint("url", name="uq_jobs_url"),
        Index("ix_jobs_company", "company"),
        Index("ix_jobs_location", "location"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    external_id: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    company: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text, nullable=False)
    apply_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    salary: Mapped[str | None] = mapped_column(String(255))
    remote: Mapped[bool | None]
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    score: Mapped[JobScore | None] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    application: Mapped[Application | None] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    search_run_links: Mapped[list[SearchRunJob]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class SearchRunJob(Base):
    __tablename__ = "search_run_jobs"

    search_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("search_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    search_run: Mapped[SearchRun] = relationship(back_populates="job_links")
    job: Mapped[Job] = relationship(back_populates="search_run_links")


class JobScore(TimestampMixin, Base):
    __tablename__ = "job_scores"
    __table_args__ = (
        CheckConstraint("score >= 0 and score <= 100", name="ck_job_scores_score_range"),
        UniqueConstraint("job_id", name="uq_job_scores_job_id"),
        Index("ix_job_scores_score", "score"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    match_reasons: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    missing_skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    risk_flags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    scoring_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    job: Mapped[Job] = relationship(back_populates="score")


class Application(TimestampMixin, Base):
    __tablename__ = "applications"
    __table_args__ = (
        CheckConstraint(
            "status in ('new', 'shortlisted', 'applied', 'rejected', 'ignored')",
            name="ck_applications_status",
        ),
        UniqueConstraint("job_id", name="uq_applications_job_id"),
        Index("ix_applications_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(30),
        default=ApplicationStatus.NEW.value,
        nullable=False,
    )
    notes: Mapped[str | None] = mapped_column(Text)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    job: Mapped[Job] = relationship(back_populates="application")
