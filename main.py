from data import *
from utils import *


def main():

    # Normalize resumes
    normalized_resumes = []

    for resume in RESUMES:

        normalized = normalize_skills(
            resume["skills"],
            SKILL_ALIASES
        )

        normalized_resumes.append(normalized)

    # Build vocabulary
    vocabulary = build_vocabulary(
        normalized_resumes
    )

    # Compute IDF
    idf = compute_idf(
        normalized_resumes,
        vocabulary
    )

    # Build resume TF-IDF vectors
    resume_vectors = []

    for resume_skills in normalized_resumes:

        vector = compute_tfidf_vector(
            resume_skills,
            vocabulary,
            idf
        )

        resume_vectors.append(vector)

    # Candidate names
    candidate_names = [
        resume["name"]
        for resume in RESUMES
    ]

    # Process each JD
    for jd in JDS:

        normalized_jd = normalize_skills(
            jd["skills"],
            SKILL_ALIASES
        )

        jd_vector = build_jd_vector(
            normalized_jd,
            vocabulary
        )

        top_candidates = rank_candidates(
            candidate_names,
            resume_vectors,
            jd_vector
        )

        print(f"{jd['id']} — {jd['company']} ({jd['role']})")

        for name, score in top_candidates:
            print(f"{name} ({score:.2f})")

        print()


if __name__ == "__main__":
    main()