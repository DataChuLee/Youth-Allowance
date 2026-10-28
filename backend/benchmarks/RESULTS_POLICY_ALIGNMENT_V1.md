# Policy answer-to-PDF alignment evaluation (v1)

Run date: 2026-08-24. This evaluation scores the stored 150-case benchmark
responses without generating new answers. The evaluated cohort contains 54
answerable cases: 27 manually reviewed source questions and 27
meaning-preserving variants.

## Why this replaces the previous correctness interpretation

RAGAS Answer Correctness compared each response against a single reference
answer. That made it unsuitable as the headline accuracy measure for policy
answers with both a restriction and an exception. For example, overseas
generative-AI subscriptions cannot be paid with the Youth Allowance card, but
can be reimbursed after personal-card payment with the required evidence. The
correct policy outcome is therefore `conditional`, not simply allowed or
blocked.

Each case in `datasets/policy_alignment_v1.jsonl` has a manually reviewed:

- policy resolution: `allowed`, `blocked`, `conditional`, or `informational`;
- list of 1–3 atomic required facts; and
- PDF chunk IDs containing those facts.

An OpenAI JSON judge evaluates only the service answer against these facts and
the gold PDF excerpts. A non-`answered_from_pdf` response is deterministically
marked `insufficient` with every required fact omitted, so the judge cannot
give credit merely because the gold PDF contains the fact.

## Results

| Metric | Result | Meaning |
| --- | ---: | --- |
| Policy resolution accuracy | 87.04% | Correct `allowed`/`blocked`/`conditional`/`informational` outcome |
| Required-fact coverage | 82.10% | Mean share of manually defined facts stated correctly |
| Contradiction-free rate | 94.44% | Answer did not state an opposite PDF fact |
| Unsupported-claim-free rate | 92.59% | No PDF-unsupported claim found |
| Actionable policy pass rate | 83.33% | Correct policy outcome and no contradiction for allowed/blocked/conditional cases |

By outcome type, resolution accuracy / fact coverage were: allowed 50.00% /
50.00% (n=4), blocked 85.71% / 71.43% (n=14), conditional 88.89% / 90.74%
(n=18), and informational 94.44% / 88.89% (n=18).

## Example: overseas generative-AI subscription

For `answerable-018`, the service answer was judged `conditional` and its
policy resolution was correct. It correctly stated both that direct Youth
Allowance card payment is unavailable and that personal-card payment can be
reimbursed. It omitted the third required fact: the minimum two supporting
documents. This is recorded as **2/3 fact coverage**, not as a wrong answer.

## Limits

- The 54 cases are manually specified, but the OpenAI judge is still an
  automated semantic judge. High-impact failures should receive human review.
- The 40 conditional benchmark cases remain outside this aggregate until their
  expected resolutions and atomic facts are manually confirmed.
- These are reproducible local benchmark observations, not a production SLA.
