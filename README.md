# Resume Matching Engine

A Python-based resume matching engine built for the Redrob AI Campus Hackathon.

## Problem Statement

Build a system that:
- normalizes noisy resume skills
- computes TF-IDF vectors for resumes
- creates binary vectors for job descriptions
- calculates cosine similarity
- ranks the Top 3 matching candidates for each job description

## Features

- Skill normalization using alias mapping
- Deduplication of resume skills
- Shared vocabulary generation
- Manual TF-IDF implementation
- Binary JD vector generation
- Cosine similarity ranking
- Top candidate recommendation system

## Technologies Used

- Python
- Standard Python libraries only
  - math

## Project Structure

```text
resume_matching_engine/
│
├── main.py
├── utils.py
├── data.py
├── prompts.txt
├── notes.txt
├── README.md
└── .gitignore
```

## Workflow

1. Normalize resume skills
2. Deduplicate skills
3. Build shared vocabulary
4. Compute TF-IDF vectors
5. Build JD binary vectors
6. Compute cosine similarity
7. Rank Top 3 candidates

## How to Run

```bash
python main.py
```

## Constraints Followed

- Used only Python standard libraries
- No external ML/NLP libraries
- Manual implementation of TF-IDF and cosine similarity

## Output Format

```text
JD-1 — Kakao (ML Engineer)
Candidate Name (score)
Candidate Name (score)
Candidate Name (score)
```

## Author

Built for the Redrob AI Campus Hackathon powered by McKinley Rice.