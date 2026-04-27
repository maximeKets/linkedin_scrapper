# LinkedIn Scrapper

POC backend for a LinkedIn job matching assistant.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env` with local credentials before running non-dry-run commands.

## Commands

Run a configuration-safe dry run:

```bash
linkedin-scrapper run-pipeline --dry-run
```

Run the pipeline once runtime services are configured:

```bash
linkedin-scrapper run-pipeline --cv-path ./path/to/cv.pdf
```

Inspect resolved non-secret settings:

```bash
linkedin-scrapper config
```

Create the database schema:

```bash
linkedin-scrapper init-db
```

Parse a CV without saving:

```bash
linkedin-scrapper parse-cv ./path/to/cv.pdf
```

Parse and save a candidate profile:

```bash
linkedin-scrapper parse-cv ./path/to/cv.pdf --save
```
