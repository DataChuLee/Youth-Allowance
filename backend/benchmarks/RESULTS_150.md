# 150-case policy and safety regression

Run date: 2026-08-22. The configured local FAISS index and OpenAI-backed
LangGraph workflow were executed for all 150 cases. This is an end-to-end
functional regression, not a concurrent-load test and not a semantic-answer
correctness evaluation.

| Scenario | Cases | Expected route match | Wrong `answered_from_pdf` route |
| --- | ---: | ---: | ---: |
| Answerable | 60 | 85.0% (51/60) | 0/60 |
| Conditional | 40 | 87.5% (35/40) | 0/40 |
| Policy blocked | 25 | 84.0% (21/25) | 1/25 (4.0%) |
| Insufficient evidence | 25 | 84.0% (21/25) | 1/25 (4.0%) |
| **Total** | **150** | **85.33% (128/150)** | **2/50 safe-path cases (4.0%)** |

Observed end-to-end latency: average **14.94 s**, p50 **16.55 s**, p95
**26.37 s**. The raw per-case output and machine-readable score summary are
kept locally under `benchmarks/results/` and are deliberately ignored by Git.

## Limitations and next actions

- The 150 cases carry `review_status`; 30 PDF questions are seed questions and
  120 cases require human adjudication before a portfolio claims definitive
  accuracy.
- No `acceptable_chunk_ids` were annotated for this expanded set. Accordingly,
  `source_hit_rate` and `required_term_rate` are reported as `null` (not
  measured), rather than as zero.
- The 22 route mismatches are retained as regression cases. Priority fixes are
  four policy-blocked variants that bypassed the policy route and four
  insufficient-evidence variants that reached a general/PDF answer route.
