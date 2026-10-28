# Document-grounded answer quality evaluation (v1)

Run date: 2026-08-24. This evaluation reuses the 60 stored LangGraph responses
from the 150-case policy run. It reruns only the current FAISS retrieval and
uses RAGAS 0.4.3 with the configured OpenAI model as the judge.

## Scope

- **54 cases:** 27 PDF seed questions with reference answers and their 27
  meaning-preserving variants.
- **Gold context:** PDF chunk IDs were manually reviewed against the local
  FAISS docstore.
- **Not in this score:** 40 conditional cases remain in
  `conditional_annotation_queue_v1.jsonl` until a human confirms their
  reference answer, required facts, and gold chunks.
- **Excluded:** three seed questions and their variants are explicit policy
  restrictions (easy payment, savings, and cash withdrawal). They belong to
  the policy evaluation rather than the PDF-answer quality aggregate.
- **Important:** three answerable questions still reached an
  `insufficient_pdf_evidence` response. The `answered_from_pdf` row is
  therefore the appropriate quality figure for generated PDF answers.

## Results

| Cohort | n | MRR@5 | Recall@5 | NDCG@5 | Source hit | Faithfulness | Answer correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All reference-labeled PDF-answer cases | 54 | 0.7238 | 0.7963 | 0.6681 | 0.9259 | 0.8657 | 0.5622 |
| `answered_from_pdf` only | 51 | 0.7663 | 0.8431 | 0.7074 | 0.9804 | 0.8657 | 0.5952 |

`Faithfulness` is available for 51 of 54 cases because three
`insufficient_pdf_evidence` responses intentionally had no cited PDF context.
`AnswerCorrectness` uses RAGAS factuality only (`weights=[1.0, 0.0]`), not
embedding similarity, to penalize missing or contradictory policy conditions.

## Interpretation

The evaluation confirms a meaningful gap between citation grounding and
complete answer quality. For the 51 document-answer cases, 98.04% cited at
least one reviewed gold chunk and the average Faithfulness score was 0.8657,
but factual Answer Correctness was 0.5952. This indicates that answers often
stay within their cited context while omitting or incompletely expressing
facts required by the reference answer.

This is not a production accuracy claim. It is a reproducible local benchmark
observation tied to the 2026-08-22 stored responses, the current local FAISS
index, and an OpenAI judge run on 2026-08-24.

## Next actions

1. Manually adjudicate the 40 conditional cases before adding them to content
   quality aggregates.
2. Split `critical_facts` into atomic required claims and report coverage in
   addition to RAGAS Answer Correctness.
3. Fix the low-score retrieval/evidence cases and rerun this exact benchmark.
4. Use `policy_safety_v2_150_failures.json` to add minimal-pair regressions for
   policy lexicon gaps and insufficient-evidence route leaks.

## Rejected prompt experiment (2026-08-24)

We tested a completion-oriented instruction in the production
`POLICY_CONTEXT_SYSTEM_PROMPT`. It asked the model to preserve explicit dates,
amounts, procedures, exceptions, and evidence requirements. The same 60
answerable inputs were regenerated and the same 54 reference-labeled cases
were judged with the same RAGAS configuration.

| Cohort | Faithfulness | Answer correctness | Avg generation latency |
| --- | ---: | ---: | ---: |
| Baseline, all 54 cases | 0.8657 | 0.5622 | 21,866 ms |
| Completion-prompt candidate, all 54 cases | 0.8520 | 0.5218 | 5,520 ms |

The candidate decreased Answer Correctness by **0.0404 points** and
Faithfulness by **0.0137 points**. Its faster latency does not compensate for
the quality regression. In the 60 generated responses, two status outcomes
also differed from the original run, so the prompt was reverted and this
candidate is not part of the deployed configuration. Result artifacts are
`answerable_v4_completion_guard.jsonl` and
`content_quality_v4_completion_guard.jsonl`.
