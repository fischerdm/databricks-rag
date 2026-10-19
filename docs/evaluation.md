# Evaluation

This document explains how the RAG backend is evaluated with a golden dataset: what the dataset contains, how it is built, what the metrics mean, and what the evaluation has found so far. It builds on [architecture.md](architecture.md), especially the sections on filter extraction and search.

**Contents**

1. [Why evaluate](#1-why-evaluate)
2. [The golden dataset](#2-the-golden-dataset)
3. [Metrics](#3-metrics)
4. [Why filters are inferred, and why that can fail](#4-why-filters-are-inferred-and-why-that-can-fail)
5. [Running the evaluation](#5-running-the-evaluation)
6. [Findings](#6-findings)
7. [Pitfalls](#7-pitfalls)

---

## 1. Why evaluate

A RAG system has many settings that change answer quality: the prompts, the models, chunk size, `top_k`, the filter rules. Trying a few questions in the UI after a change shows whether *those* questions still work, not whether the system as a whole got better or worse.

A **golden dataset** is a fixed, reviewed set of questions with known right answers. Running all of them after every change turns 'it seems fine' into numbers that can be compared between runs. It is regression testing for RAG.

So far the evaluation scores **retrieval**: does the search find the article that answers the question? Retrieval comes first because the LLM can only answer from the passages it is given. If the right article is not among them, no prompt can fix the answer. Scoring the answers themselves (faithfulness, correctness, refusals) is the next step.

## 2. The golden dataset

[evals/golden.jsonl](../evals/golden.jsonl) holds one case per line:

```json
{"id": "q0007",
 "question": "What did USDA project for U.S. corn stocks on Aug. 31, 1987?",
 "expected_doc_ids": ["reuters-15853", "reuters-15875"],
 "reference_answer": "USDA projected U.S. corn stocks on Aug. 31, 1987, at 5,240 mln bushels, ...",
 "key_facts": ["5,240"],
 "style": "undated",
 "source_title": "U.S. SUPPLY/DEMAND HIGHLIGHTS",
 "source_date": "1987-04-09",
 "generated_by": "gpt-5.4-mini",
 "reviewed": true}
```

| Field | Meaning |
|---|---|
| `question` | What a user would type |
| `expected_doc_ids` | Articles that answer the question. Retrieval is scored against these |
| `reference_answer`, `key_facts` | A correct answer and the facts it must contain. Not used yet; for answer scoring later |
| `style` | `dated` if the question mentions when the news was reported, `undated` if it mentions no date |
| `source_title`, `source_date` | The article the question was generated from, to make review easier |
| `reviewed` | Whether a person has checked the case |
| `notes` | Optional. What the review decided and why, e.g. 'Added reuters-3065: same story, same day' |

### Step 1: draft cases

[evals/generate.py](../evals/generate.py) builds cases from the corpus itself:

1. Sample articles that are indexed, have a date and a topic, and are at least 400 characters long. No single topic may make up more than 25% of the cases; otherwise about 40% would be earnings reports.
2. Give each article to an LLM (`gpt-5.4-mini`) with instructions to write one short question that this article answers, specific enough that it is the best match among 19,000 articles, plus a reference answer.
3. Randomly make about half the questions `dated` and the rest `undated`. Dated questions test filter extraction; undated ones test plain similarity search.

The trick is that **the article the question was written from is the expected answer**, so no labelling is needed. Running the script again appends new cases and skips articles already in the file.

```bash
.venv/bin/python -m evals.generate --n 50
```

`gpt-4o-mini` was tried first and ignored the date rules ('this week', 'next week' instead of a date); `gpt-5.4-mini` followed them.

### Step 2: review the cases

Generated cases are drafts. Review is what makes the dataset *golden*. For each case with `"reviewed": false`:

- **Delete or rewrite vague questions** that many articles answer equally well.
- **Add equally good articles to `expected_doc_ids`.** Reuters often has several versions of one story, such as a short highlights item and a detailed table. If the second one is retrieved, that is not a miss.
- **Check the reference answer** against the article.
- **Never change a case to make a system failure go away.** If the right article was excluded by a wrong filter, the case is correct and the system is not.
- Set `"reviewed": true`, and write what you decided in `notes` if you changed anything.

**Start with the misses.** A case the system already ranks first rarely hides a problem with `expected_doc_ids`. For a miss, the articles retrieved instead are the candidates for equally good answers, so you only judge those few, never the whole corpus. Search benchmarks have been built this way for decades ('pooling'). An equally good article that no run ever retrieves stays unknown; it does not distort the scores, and it shows up for review once a better version of the system retrieves it.

## 3. Metrics

For each question, the search returns the `top_k` (default 4) closest chunks. Chunks of the same article are merged, which gives a ranked list of articles. Every metric is based on **where the expected article lands in that list**.

| Metric | Per question | Over all questions |
|---|---|---|
| **hit@1** | 1 if an expected article is ranked **first**, else 0 | Share of questions where it came first |
| **hit@k** | 1 if an expected article is **among the retrieved ones**, else 0 | Share of questions where it was retrieved at all |
| **MRR** (Mean Reciprocal Rank) | 1/rank: 1st → 1, 2nd → 0.5, 3rd → 0.33, 4th → 0.25, not retrieved → 0 | Average of those values |
| **filt.out** | 1 if the applied filters **excluded** every expected article | Share of such questions |

A worked example with four questions:

| Question | Rank of expected article | hit@1 | hit@4 | 1/rank |
|---|---|---|---|---|
| A | 1 | 1 | 1 | 1.00 |
| B | 3 | 0 | 1 | 0.33 |
| C | 1 | 1 | 1 | 1.00 |
| D | not retrieved | 0 | 0 | 0 |
| **Score** | | **50%** | **75%** | **MRR 0.58** |

How to read them:

- **hit@k** is the minimum requirement: the LLM sees only these passages.
- **hit@1** is stricter: with the best source on top, a good answer is more likely.
- **MRR** combines the two into one number; second place counts as half a hit instead of a miss.
- **filt.out** is a diagnostic, not a quality score. It says *why* a miss happened: the article was removed by a filter before the search started, so it could not be found at any rank. A high value points at filter extraction rather than at the embeddings.

## 4. Why filters are inferred, and why that can fail

Similarity search compares *meaning*. It finds 'articles about oil prices' well, but a date in the question only shifts the embedding slightly; it does not act as a constraint. The corpus has 566 crude-oil articles spread over eight months, and without filters the question 'What happened to oil prices on October 19, 1987?' retrieved articles from March and April.

Metadata filters are exact: `date_int` between 19871019 and 19871020, `topics` contains `crude`. But users type free text, not form fields. So a small LLM first **translates the question into filters**, and the database applies them before ranking:

```
"What happened to oil prices on October 19, 1987?"
   ↓ LLM (filter extraction)
date 1987-10-19 – 1987-10-20, topic = crude
   ↓ database keeps only matching articles
   ↓ similarity search ranks the rest
```

Explicit filters from the UI settings are the alternative. They always win over inferred ones (see [architecture.md](architecture.md#servicesfilterspy--filter-extraction)).

A filter is all or nothing. When the inferred value is right, it removes noise. When it is wrong, the right article becomes **impossible to find**, and the fallback does not help as long as the wrong filter still matches *some* articles:

```
question ─► wrong filter ─► right article removed ─► 4 other articles found ─► no fallback
```

- **Dates are low risk.** The question states them, so the LLM mostly copies them.
- **Topics are high risk.** Topic codes were assigned by Reuters editors in 1987, from a fixed list of 119. The LLM has to guess which code an editor chose, and that guess is subjective.

## 5. Running the evaluation

[evals/evaluate.py](../evals/evaluate.py) runs each question through the same retrieval pipeline as `/query` ([app/services/retrieval.py](../app/services/retrieval.py)), so the scores describe the real system and not a copy of it.

```bash
.venv/bin/python -m evals.evaluate                    # as /query runs
.venv/bin/python -m evals.evaluate --no-auto-filters  # plain similarity search, for comparison
.venv/bin/python -m evals.evaluate --reviewed-only    # only reviewed cases
.venv/bin/python -m evals.evaluate --top-k 8          # retrieve more chunks
```

Output: the metrics overall and per `style`, then every case not ranked first with its applied filters and the retrieved titles. The full results and the settings used are written to `evals/runs/` (git-ignored), so two runs can be compared.

Cases whose articles are not in the index (e.g. after `ingest_reuters --limit 500`) are skipped and counted separately, not scored as misses.

## 6. Findings

Each entry records what was measured, on which data, and what follows from it.

### 2026-10-19: baseline, inferred topic filter hurts retrieval

**Setup.** 50 generated cases, **not yet reviewed**. `top_k` 4, `text-embedding-3-small`, `gpt-5.4-nano` for filter extraction, full corpus indexed.

| | n | hit@1 | hit@4 | MRR | filt.out |
|---|---|---|---|---|---|
| **With auto filters** (as `/query` runs) | 50 | 74% | 86% | 0.80 | 10% |
| dated | 23 | 87% | 96% | 0.91 | 4% |
| undated | 27 | 63% | 78% | 0.70 | 15% |
| **Without auto filters** | 50 | 80% | 96% | 0.88 | 0% |
| dated | 23 | 87% | 100% | 0.93 | 0% |
| undated | 27 | 74% | 93% | 0.83 | 0% |

**Observation.** Filter extraction makes retrieval *worse* overall (hit@4 86% vs 96%). The loss is in the undated questions, which contain no date, so the filter responsible must be the topic. The five filtered-out cases confirm it:

| Question | Inferred filter | Article's labels | Cause |
|---|---|---|---|
| What U.S. joint venture did Nippon Mining sign with Conoco Inc? | `acq` | `crude` | wrong topic |
| What is Bank of British Columbia's revised shareholder distribution estimate after selling its assets? | `acq` | `earn` | wrong topic |
| Why is Cehave NV Veghel challenging the European Community grain co-responsibility levy? | `grain` | `meal-feed` | wrong topic |
| What pact did California Micro Devices and Grumman's Tachonics sign in March? | March, `acq` | `strategic-metal` | wrong topic |
| How much beef did Argentina export in Jan/Feb 1987? | Jan–Feb | published Mar 11 | event date ≠ publication date |

Take the Nippon Mining case. The LLM read 'joint venture … sign with' and chose `acq` (Mergers/Acquisitions), a reasonable guess. But Reuters labelled the story `crude`, because it is about oil. The filter removed the article before the search started; four unrelated `acq` articles were found instead, so the fallback never ran. Without filters, the article was retrieved.

**Interpretation.** The inferred date filter is neutral to helpful (dated questions: 87% hit@1 either way). The inferred topic filter is the main source of misses: 4 of 5 filtered-out cases. This is the limitation listed in [architecture.md](architecture.md#7-known-limitations) ('the fallback only triggers on zero results'), now with a number.

**Caveats.** The cases are unreviewed and 50 is a small sample: one case moves a score by 2 percentage points. The direction is clear, though: each filtered-out case was checked against the article's labels by hand.

**Options to evaluate next:**

- Infer only dates; drop the inferred topic.
- Use the topic to boost matching articles in the ranking instead of excluding the others.
- Fall back to an unfiltered search not only on zero results but also when the filtered results are poor matches.

### 2026-10-19: review of the misses, and filter extraction is not deterministic

**Setup.** The 13 cases not ranked first in the baseline were reviewed (now `reviewed: true`, with `notes`); the other 37 are not yet. Same settings as the baseline. The auto-filter evaluation was run three times, plain search once.

**Review.** Only one of the 13 questions needed rewriting. The misses fell into four groups:

| Group | Cases | Fix |
|---|---|---|
| **Filter excluded the article.** Plain search ranks it first | q0014, q0015, q0017, q0024, q0033 | None. The case is right and the system is wrong; these are the tests for a filter fix |
| **Same story twice.** Reuters ran an update or repeat, and the system found the other copy | q0008, q0027, q0028, q0040 | Added the copies to `expected_doc_ids` |
| **Genuine ranking miss** | q0004, q0012, q0039 | None |
| **Question too vague**, answered by dozens of articles | q0021 | Rewritten, plus a duplicate added. Still a genuine miss: the answer is in the second chunk of a long G-7 statement, and analysis articles about the statement rank higher |

The review also caught a broken reference answer (q0028, cut off mid-sentence).

**Results after the review:**

| | hit@1 | hit@4 | MRR | filt.out |
|---|---|---|---|---|
| With auto filters, run 1 | 82% | 86% | 0.84 | 10% |
| With auto filters, run 2 | 80% | 84% | 0.82 | 12% |
| With auto filters, run 3 | 78% | 82% | 0.80 | 14% |
| **Without auto filters** | **88%** | **96%** | **0.92** | 0% |

**Finding 1: the gap is now entirely the filter.** Fixing the duplicates raised both configurations, and the remaining genuine misses (q0004, q0012, q0021, q0039) occur with and without filters. What separates the two is the cases where filters excluded the right article: hit@4 is 96% without filter extraction and 82–86% with it.

**Finding 2: filter extraction gives different results on identical runs.** Nothing changed between the three runs, yet:

| Case | Inferred topic in runs 1 / 2 / 3 | Article's labels |
|---|---|---|
| q0014 California Micro / Grumman pact | none / `acq` / none | `strategic-metal` |
| q0024 Nippon Mining / Conoco | none / none / `acq` | `crude` |
| q0027 Saudi Arabia / Petrobras credit | `instal-debt` in all three (none in the baseline) | `crude` |

The search itself is deterministic: same question, same embedding, same ranking. The variation comes from the extraction LLM. Consequences:

- A difference of a few points between two single runs can be noise. Compare configurations over several runs.
- A user asking the same question twice can get different sources.
- q0027 shows the failure mode again: the model matched 'credit guarantees' to `instal-debt` (Instalment Debt/Consumer Credit) on shared words, against the prompt's instruction to match on meaning.

**Finding 3: the topic filter also helps, but less often than it hurts.** In q0025 (Allis-Chalmers restructuring) and q0026 (Dauphin Deposit acquisition), the inferred `acq` filter moved the right article from rank 2 to rank 1, in all three runs. Against that, 5–7 cases per run lose the right article entirely. Any fix should keep the benefit if it can, e.g. by using the topic to boost matching articles rather than to exclude the others.

**Next.** Review the remaining 37 cases, then compare the filter options from the baseline entry over several runs each.

## 7. Pitfalls

- **Don't tune on the cases you report.** If you change prompts until every case passes, the scores stop telling you anything. Keep some cases aside and look at them only for the final comparison.
- **LLM steps vary between runs.** Filter extraction can give different filters for the same question on two runs, so scores move by a few points without any change. Run the evaluation several times before concluding that a change helped.
- **A miss can be a wrong case.** Check the retrieved titles before blaming the system: often a second article answers the question just as well and belongs in `expected_doc_ids`.
- **Dated questions use the publication date.** Filters match the date an article was published. A user asking about 'the week ending March 19' gets a filter that excludes the article published on April 1 that reports it. The generator avoids such questions; real users won't, so add a few by hand.
- **A generated question is easier than a real one.** It was written from the article, often with the article's own words, which flatters similarity search. Add real questions from the UI over time, especially the ones that failed.
