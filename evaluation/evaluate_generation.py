from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.generation.generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_questions.json"
)

class AnswerJudgeResult(BaseModel):
    score: int = Field(
        ge=0,
        le=3,
    )
    reason: str


class FaithfulnessJudgeResult(BaseModel):
    faithful: bool

    score: int = Field(
        ge=0,
        le=3,
    )

    unsupported_claims: List[str]

    reason: str


class CitationCorrectnessResult(BaseModel):
    correct: bool

    score: int = Field(
        ge=0,
        le=3,
    )

    reason: str


class CitationCompletenessResult(BaseModel):
    requires_citation: bool
    has_citation: bool
    complete: bool
    reason: str

# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions() -> List[Dict[str, Any]]:
    with open(
        GENERATION_QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def check_expected_terms(
    answer: str,
    expected_terms: List[str],
) -> bool:

    if not expected_terms:
        return True

    normalized_answer = normalize_text(
        answer
    )

    return any(
        normalize_text(term)
        in normalized_answer
        for term in expected_terms
    )


def check_citations_present(
    answer: str,
) -> bool:

    return bool(
        re.search(
            r"\[\d+]",
            answer,
        )
    )


def check_abstention(
    answer: str,
) -> bool:

    expected_message = (
        "i could not find sufficient evidence "
        "in the provided documents."
    )

    return (
        expected_message
        in normalize_text(answer)
    )


# ============================================================
# ANSWER QUALITY JUDGE
# ============================================================

def judge_answer(
    question: str,
    answer: str,
    reference_answer: str | None,
) -> Dict[str, Any]:

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    structured_llm = (
        llm.with_structured_output(
            AnswerJudgeResult
        )
    )

    reference_section = ""

    if reference_answer:
        reference_section = f"""
REFERENCE ANSWER:
{reference_answer}
"""

    prompt = f"""
You are evaluating the quality of an answer produced by a RAG assistant.

Evaluate whether the GENERATED ANSWER correctly and directly
answers the QUESTION.

Important:

- Do not penalize correct supporting detail simply because it is
  not written word-for-word in the reference answer.
- Do penalize unnecessary or distracting information when it makes
  the answer less direct.
- Evaluate answer quality only.
- Do not evaluate citation formatting here.

{reference_section}

QUESTION:
{question}

GENERATED ANSWER:
{answer}

Scoring:

0 = incorrect or does not answer the question
1 = partially correct or substantially over-scoped
2 = mostly correct with minor issues
3 = fully correct, direct, and appropriately scoped
""".strip()

    result = structured_llm.invoke(
        prompt
    )

    return result.model_dump()


# ============================================================
# FAITHFULNESS JUDGE
# ============================================================

def judge_faithfulness(
    question: str,
    answer: str,
    retrieved_context: str,
) -> Dict[str, Any]:

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    structured_llm = (
        llm.with_structured_output(
            FaithfulnessJudgeResult
        )
    )

    prompt = f"""
You are evaluating the faithfulness of an answer produced
by a Retrieval-Augmented Generation system.

Determine whether every material factual claim in the
GENERATED ANSWER is supported by the RETRIEVED EVIDENCE.

Rules:

1. Use ONLY the retrieved evidence.

2. Do NOT use outside knowledge.

3. Evaluate grounding only.
   Do NOT penalize verbosity, style, relevance, or citation formatting.

4. A claim is supported if the evidence directly supports it
   or reasonably entails it.

5. A claim is unsupported if it introduces information that is
   absent from or contradicted by the retrieved evidence.

6. If every material factual claim is supported:
   faithful MUST be true
   and score MUST be 3.

7. Score 0 only when there is a major unsupported or contradictory
   factual claim.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{retrieved_context}

GENERATED ANSWER:
{answer}

Scoring:

0 = major unsupported or contradictory claims
1 = several material unsupported claims
2 = mostly supported with minor unsupported detail
3 = every material factual claim is supported
""".strip()

    result = structured_llm.invoke(
        prompt
    )

    return result.model_dump()


# ============================================================
# CLAIM + CITATION EXTRACTION
# ============================================================

def extract_claims_and_citations(
    answer: str,
) -> List[Dict[str, Any]]:
    """
    Break answer into factual claims and capture citations
    attached to each claim.

    Example:

        "Participation is voluntary. [1]"
        "Subjects may withdraw at any time. [2][3]"

    becomes roughly:

        [
          {
            "claim": "Participation is voluntary.",
            "citation_ids": [1]
          },
          {
            "claim": "Subjects may withdraw at any time.",
            "citation_ids": [2, 3]
          }
        ]

    This is heuristic-based.

    The semantic judge later determines whether a sentence
    is actually a factual claim.
    """

    # Convert bullet boundaries into sentence-like boundaries.
    text = answer.replace(
        "\n- ",
        "\n"
    )

    # Split on newlines and normal sentence endings.
    segments = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    claims = []

    for segment in segments:

        segment = segment.strip()

        if not segment:
            continue

        citation_ids = [
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                segment,
            )
        ]

        clean_claim = re.sub(
            r"\[\d+\]",
            "",
            segment,
        ).strip()

        clean_claim = re.sub(
            r"^[•\-]\s*",
            "",
            clean_claim,
        ).strip()

        if not clean_claim:
            continue

        claims.append(
            {
                "claim": clean_claim,
                "citation_ids": citation_ids,
            }
        )

    return claims


# ============================================================
# BUILD CITED EVIDENCE
# ============================================================

def build_evidence_for_citation_ids(
    citation_ids: List[int],
    retrieved_chunks: List[Dict[str, Any]],
) -> str:

    chunks_by_id = {
        chunk["source_id"]: chunk
        for chunk in retrieved_chunks
    }

    evidence_blocks = []

    for citation_id in citation_ids:

        chunk = chunks_by_id.get(
            citation_id
        )

        if not chunk:
            continue

        block = f"""
[SOURCE {citation_id}]
Document: {chunk.get("source")}
Page: {chunk.get("page")}
Chunk: {chunk.get("chunk_id")}

Content:
{chunk.get("text")}
""".strip()

        evidence_blocks.append(
            block
        )

    return "\n\n".join(
        evidence_blocks
    )


# ============================================================
# CITATION CORRECTNESS JUDGE
# ============================================================

def judge_citation_correctness(
    claim: str,
    citation_ids: List[int],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not citation_ids:
        return {
            "correct": False,
            "score": 0,
            "reason": (
                "No citation attached to claim."
            ),
        }

    evidence = build_evidence_for_citation_ids(
        citation_ids=citation_ids,
        retrieved_chunks=retrieved_chunks,
    )

    if not evidence:
        return {
            "correct": False,
            "score": 0,
            "reason": (
                "Citation IDs could not be resolved "
                "to retrieved evidence."
            ),
        }

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    structured_llm = (
        llm.with_structured_output(
            CitationCorrectnessResult
        )
    )

    prompt = f"""
You are evaluating citation correctness in a RAG system.

Determine whether the CITED EVIDENCE actually supports the CLAIM.

Rules:

1. Use ONLY the cited evidence.

2. Do NOT use outside knowledge.

3. The citation is correct only if the evidence directly supports
   or reasonably entails the claim.

4. Related subject matter is not enough.

5. Do not reward a citation merely because it comes from the
   correct document.

CLAIM:
{claim}

CITED EVIDENCE:
{evidence}

Scoring:

0 = cited evidence does not support the claim
1 = weak or partial support
2 = mostly supports the claim
3 = directly and fully supports the claim
""".strip()

    result = structured_llm.invoke(
        prompt
    )

    return result.model_dump()


# ============================================================
# CITATION COMPLETENESS JUDGE
# ============================================================

def judge_citation_completeness(
    claim: str,
    citation_ids: List[int],
) -> Dict[str, Any]:

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    structured_llm = (
        llm.with_structured_output(
            CitationCompletenessResult
        )
    )

    prompt = f"""
You are evaluating citation completeness in a RAG answer.

Determine whether the following text contains a material factual
claim that requires support from retrieved evidence.

CLAIM:
{claim}

Attached citation IDs:
{citation_ids}

Rules:

1. Definitions require citations.

2. Requirements, recommendations, document claims, factual
   assertions, and numeric statements require citations.

3. Pure headings, transition phrases, introductions, or formatting
   labels do NOT require citations.

4. If the text requires a citation and citation_ids is empty:
   has_citation must be false
   complete must be false.

5. If the text does not require a citation:
   complete must be true.

6. If the text requires a citation and at least one citation exists:
   has_citation must be true
   complete must be true.
""".strip()

    result = structured_llm.invoke(
        prompt
    )

    return result.model_dump()


# ============================================================
# EVALUATE ALL CITATIONS IN ANSWER
# ============================================================

def evaluate_answer_citations(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    claims = extract_claims_and_citations(
        answer
    )

    claim_results = []

    correctness_scores = []

    required_claims = 0
    cited_required_claims = 0

    for claim_item in claims:

        claim = claim_item[
            "claim"
        ]

        citation_ids = claim_item[
            "citation_ids"
        ]

        completeness = (
            judge_citation_completeness(
                claim=claim,
                citation_ids=citation_ids,
            )
        )

        requires_citation = (
            completeness.get(
                "requires_citation",
                True,
            )
        )

        if requires_citation:
            required_claims += 1

            if citation_ids:
                cited_required_claims += 1

        correctness = None

        # Only judge citation correctness when
        # a citation is actually present.
        if citation_ids:

            correctness = (
                judge_citation_correctness(
                    claim=claim,
                    citation_ids=citation_ids,
                    retrieved_chunks=retrieved_chunks,
                )
            )

            correctness_scores.append(
                correctness.get(
                    "score",
                    0,
                )
            )

        claim_results.append(
            {
                "claim": claim,
                "citation_ids": citation_ids,
                "requires_citation": (
                    requires_citation
                ),
                "citation_complete": (
                    completeness.get(
                        "complete",
                        False,
                    )
                ),
                "citation_correctness": (
                    correctness
                ),
            }
        )

    # --------------------------------------------------------
    # Citation completeness
    # --------------------------------------------------------

    if required_claims == 0:
        completeness_rate = 1.0

    else:
        completeness_rate = (
            cited_required_claims
            / required_claims
        )

    # --------------------------------------------------------
    # Citation correctness
    # --------------------------------------------------------

    if correctness_scores:

        average_correctness_score = (
            sum(correctness_scores)
            / len(correctness_scores)
        )

        fully_correct_count = sum(
            1
            for score in correctness_scores
            if score == 3
        )

        correctness_rate = (
            fully_correct_count
            / len(correctness_scores)
        )

    else:

        average_correctness_score = 0
        correctness_rate = 0

    return {
        "claims": claim_results,
        "required_claims": required_claims,
        "cited_required_claims": (
            cited_required_claims
        ),
        "citation_completeness_rate": (
            completeness_rate
        ),
        "average_citation_correctness_score": (
            average_correctness_score
        ),
        "citation_correctness_rate": (
            correctness_rate
        ),
    }


# ============================================================
# SINGLE QUESTION
# ============================================================

def evaluate_question(
    item: Dict[str, Any],
) -> Dict[str, Any]:

    question = item["question"]

    answerable = item.get(
        "answerable",
        True,
    )

    expected_terms = item.get(
        "expected_answer_terms",
        [],
    )

    reference_answer = item.get(
        "reference_answer"
    )

    result = generate_answer(
        question=question,
        retrieval_k=5,
        rerank_k=5,
        use_reranker=False,
    )

    answer = result["answer"]

    retrieved_context = result.get(
        "retrieved_context",
        "",
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    citation_validation = result.get(
        "citation_validation",
        {},
    )

    citations = result.get(
        "citations",
        {},
    )

    evaluation = {
        "id": item.get("id"),

        "question": question,

        "query_type": item.get(
            "query_type",
            "unknown",
        ),

        "answer": answer,

        "answerable": answerable,
    }

    # ========================================================
    # ANSWERABLE
    # ========================================================

    if answerable:

        evaluation[
            "abstained"
        ] = check_abstention(
            answer
        )

        evaluation[
            "expected_terms_found"
        ] = check_expected_terms(
            answer=answer,
            expected_terms=expected_terms,
        )

        evaluation[
            "citations_present"
        ] = check_citations_present(
            answer
        )

        evaluation[
            "citation_ids_valid"
        ] = citation_validation.get(
            "valid",
            False,
        )

        evaluation[
            "citation_count"
        ] = len(
            citations
        )

        # ----------------------------------------------------
        # Answer quality
        # ----------------------------------------------------

        answer_judge = judge_answer(
            question=question,
            answer=answer,
            reference_answer=reference_answer,
        )

        evaluation[
            "answer_score"
        ] = answer_judge.get(
            "score",
            0,
        )

        evaluation[
            "answer_reason"
        ] = answer_judge.get(
            "reason",
            "",
        )

        # ----------------------------------------------------
        # Faithfulness
        # ----------------------------------------------------

        faithfulness = (
            judge_faithfulness(
                question=question,
                answer=answer,
                retrieved_context=(
                    retrieved_context
                ),
            )
        )

        evaluation[
            "faithfulness_score"
        ] = faithfulness.get(
            "score",
            0,
        )

        evaluation[
            "faithful"
        ] = faithfulness.get(
            "faithful",
            False,
        )

        evaluation[
            "unsupported_claims"
        ] = faithfulness.get(
            "unsupported_claims",
            [],
        )

        evaluation[
            "faithfulness_reason"
        ] = faithfulness.get(
            "reason",
            "",
        )

        # ----------------------------------------------------
        # Citation semantic evaluation
        # ----------------------------------------------------

        citation_semantics = (
            evaluate_answer_citations(
                answer=answer,
                retrieved_chunks=(
                    retrieved_chunks
                ),
            )
        )

        evaluation[
            "citation_completeness_rate"
        ] = citation_semantics[
            "citation_completeness_rate"
        ]

        evaluation[
            "average_citation_correctness_score"
        ] = citation_semantics[
            "average_citation_correctness_score"
        ]

        evaluation[
            "citation_correctness_rate"
        ] = citation_semantics[
            "citation_correctness_rate"
        ]

        evaluation[
            "citation_claims"
        ] = citation_semantics[
            "claims"
        ]

    # ========================================================
    # UNANSWERABLE
    # ========================================================

    else:

        abstained = check_abstention(
            answer
        )

        evaluation[
            "abstained"
        ] = abstained

        evaluation[
            "correct_abstention"
        ] = abstained

        evaluation[
            "citation_count"
        ] = len(
            citations
        )

    return evaluation


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:

    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    summary = {}

    if answerable_results:

        total = len(
            answerable_results
        )

        summary[
            "expected_term_accuracy"
        ] = (
            sum(
                result[
                    "expected_terms_found"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "citation_presence_rate"
        ] = (
            sum(
                result[
                    "citations_present"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "citation_validity_rate"
        ] = (
            sum(
                result[
                    "citation_ids_valid"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "average_answer_score"
        ] = (
            sum(
                result[
                    "answer_score"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "average_faithfulness_score"
        ] = (
            sum(
                result[
                    "faithfulness_score"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "faithfulness_rate"
        ] = (
            sum(
                result[
                    "faithful"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "average_citation_completeness"
        ] = (
            sum(
                result[
                    "citation_completeness_rate"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "average_citation_correctness_score"
        ] = (
            sum(
                result[
                    "average_citation_correctness_score"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "citation_correctness_rate"
        ] = (
            sum(
                result[
                    "citation_correctness_rate"
                ]
                for result
                in answerable_results
            )
            / total
        )

        summary[
            "false_abstention_rate"
        ] = (
            sum(
                result[
                    "abstained"
                ]
                for result
                in answerable_results
            )
            / total
        )

    if unanswerable_results:

        summary[
            "abstention_accuracy"
        ] = (
            sum(
                result[
                    "correct_abstention"
                ]
                for result
                in unanswerable_results
            )
            / len(
                unanswerable_results
            )
        )

    return summary


# ============================================================
# PRINT ANSWERABLE RESULT
# ============================================================

def print_answerable_result(
    result: Dict[str, Any],
):

    print(
        "\nExpected terms found:",
        result[
            "expected_terms_found"
        ],
    )

    print(
        "Citations present:",
        result[
            "citations_present"
        ],
    )

    print(
        "Citation IDs valid:",
        result[
            "citation_ids_valid"
        ],
    )

    print(
        "Answer score:",
        result[
            "answer_score"
        ],
    )

    print(
        "Faithfulness score:",
        result[
            "faithfulness_score"
        ],
    )

    print(
        "Faithful:",
        result[
            "faithful"
        ],
    )

    print(
        "Citation completeness:",
        f"{result['citation_completeness_rate']:.2%}",
    )

    print(
        "Citation correctness score:",
        f"{result['average_citation_correctness_score']:.2f} / 3",
    )

    print(
        "Fully correct citation rate:",
        f"{result['citation_correctness_rate']:.2%}",
    )

    # --------------------------------------------------------
    # Detailed claim-level citation debugging
    # --------------------------------------------------------

    print(
        "\nCitation claim analysis:"
    )

    for claim_result in result[
        "citation_claims"
    ]:

        print(
            "\nClaim:",
            claim_result[
                "claim"
            ],
        )

        print(
            "Citation IDs:",
            claim_result[
                "citation_ids"
            ],
        )

        print(
            "Requires citation:",
            claim_result[
                "requires_citation"
            ],
        )

        print(
            "Citation complete:",
            claim_result[
                "citation_complete"
            ],
        )

        correctness = claim_result.get(
            "citation_correctness"
        )

        if correctness:

            print(
                "Citation correctness:",
                correctness.get(
                    "score"
                ),
                "/ 3",
            )

            print(
                "Reason:",
                correctness.get(
                    "reason"
                ),
            )


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def print_summary(
    summary: Dict[str, float],
):

    print("\n")
    print("#" * 100)
    print("GENERATION EVALUATION RESULTS")
    print("#" * 100)

    percentage_metrics = {
        "expected_term_accuracy",
        "citation_presence_rate",
        "citation_validity_rate",
        "faithfulness_rate",
        "average_citation_completeness",
        "citation_correctness_rate",
        "false_abstention_rate",
        "abstention_accuracy",
    }

    score_metrics = {
        "average_answer_score",
        "average_faithfulness_score",
        "average_citation_correctness_score",
    }

    for metric, value in summary.items():

        if metric in percentage_metrics:

            print(
                f"{metric}: "
                f"{value:.2%}"
            )

        elif metric in score_metrics:

            print(
                f"{metric}: "
                f"{value:.2f} / 3"
            )

        else:

            print(
                f"{metric}: "
                f"{value}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    questions = load_questions()

    print(
        f"Loaded "
        f"{len(questions)} "
        f"generation evaluation questions."
    )

    results = []

    for item in questions:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"QUESTION: "
            f"{item['question']}"
        )

        print(
            "=" * 100
        )

        result = evaluate_question(
            item
        )

        results.append(
            result
        )

        print("\nANSWER:")

        print(
            result["answer"]
        )

        if result[
            "answerable"
        ]:

            print_answerable_result(
                result
            )

        else:

            print(
                "\nCorrect abstention:",
                result[
                    "correct_abstention"
                ],
            )

    summary = calculate_summary(
        results
    )

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()