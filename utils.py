import math

def normalize_skills(skill_string, alias_mapping):
    """
    Normalize comma-separated skills.

    Steps:
    - split on commas
    - lowercase
    - strip whitespace
    - apply alias mapping
    - discard unknown skills
    - remove duplicates
    - return sorted canonical skills
    """

    skills = skill_string.split(",")

    canonical_skills = set()

    for skill in skills:
        skill = skill.strip().lower()

        # Apply alias mapping
        if skill in alias_mapping:
            canonical_skill = alias_mapping[skill]
            canonical_skills.add(canonical_skill)

    return sorted(list(canonical_skills))

def build_vocabulary(normalized_resumes):
    """
    Build a sorted vocabulary from normalized resumes.
    """

    vocabulary = set()

    for skills in normalized_resumes:
        for skill in skills:
            vocabulary.add(skill)

    return sorted(list(vocabulary))



def compute_idf(normalized_resumes, vocabulary):
    """
    Compute IDF values for each skill in vocabulary.
    """

    total_resumes = len(normalized_resumes)

    idf = {}

    for skill in vocabulary:

        document_count = 0

        for resume in normalized_resumes:
            if skill in resume:
                document_count += 1

        idf[skill] = math.log(total_resumes / document_count)

    return idf


def compute_tfidf_vector(resume_skills, vocabulary, idf):
    """
    Compute TF-IDF vector for one resume.
    """

    tfidf_vector = []

    total_skills = len(resume_skills)

    for skill in vocabulary:

        if skill in resume_skills:
            tf = 1 / total_skills
            tfidf = tf * idf[skill]
        else:
            tfidf = 0

        tfidf_vector.append(tfidf)

    return tfidf_vector

def build_jd_vector(jd_skills, vocabulary):
    """
    Build binary vector for a job description.
    """

    jd_vector = []

    for skill in vocabulary:

        if skill in jd_skills:
            jd_vector.append(1)
        else:
            jd_vector.append(0)

    return jd_vector

def cosine_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors.
    """

    dot_product = 0

    for a, b in zip(vector1, vector2):
        dot_product += a * b

    magnitude1 = math.sqrt(sum(x * x for x in vector1))
    magnitude2 = math.sqrt(sum(x * x for x in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

def rank_candidates(candidate_names, resume_vectors, jd_vector):
    """
    Rank candidates using cosine similarity.
    """

    candidate_scores = []

    for name, vector in zip(candidate_names, resume_vectors):

        score = cosine_similarity(vector, jd_vector)

        candidate_scores.append(
            (name, round(score, 2))
        )

    ranked_candidates = sorted(
        candidate_scores,
        key=lambda x: (-x[1], x[0])
    )

    return ranked_candidates[:3]