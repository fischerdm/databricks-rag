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
| `style` | Describes the question, not the article: `dated` if it mentions a date or period ('on March 24', 'in Jan/Feb 1987'), `undated` if it mentions none |
| `source_title`, `source_date` | The article the question was generated from, to make review easier |
| `reviewed` | Whether a person has checked the case |
| `notes` | Optional. What the review decided and why, e.g. 'Added reuters-3065: same story, same day' |

### Step 1: draft cases

[evals/generate.py](../evals/generate.py) builds cases from the corpus itself:

1. Sample articles that are indexed, have a date and a topic, and are at least 400 characters long. No single topic may make up more than 25% of the cases; otherwise about 40% would be earnings reports.
2. Give each article to an LLM (`gpt-5.4-mini`) and ask for a structured reply with three fields, all written from the article in one call:

   | Field | Instruction to the LLM |
   |---|---|
   | `question` | One short question (at most 20 words) that this article answers, specific enough that it is the best match among 19,000 articles |
   | `reference_answer` | One or two sentences, only facts stated in the article |
   | `key_facts` | One to three facts the answer must contribute beyond the question: verbatim from the article, as short as possible, never repeating the question |

3. Randomly make about half the questions `dated` and the rest `undated`. Dated questions test filter extraction; undated ones test plain similarity search.

The trick is that **the article the question was written from is the expected answer**, so no labelling is needed. Running the script again appends new cases and skips articles already in the file.

**Key facts come from the article, not from the question.** They are what a correct answer must *add* to the question. Key facts do not affect the retrieval scores. They are for scoring answers later, where a cheap first check is whether an answer contains them ('28,526', '17.6 pct'). That only works if they follow four rules:

| Rule | Bad | Good | Why |
|---|---|---|---|
| Verbatim from the article | 'Malaysia 28,526 tonnes' (the article says 'Malaysia has been allocated an export quota of 28,526 tonnes') | '28,526' | Checked by exact string matching |
| Never repeats the question | 'Fidelcor Business Credit Corp' for 'What did Fidelcor Business Credit Corp agree to buy…?' | 'loan portfolio' | Every answer repeats the question's names, so the check would always pass |
| As short as possible | 'most of Lazere's loan portfolio' | 'loan portfolio' | A correct answer may word the rest differently ('most of its loan portfolio') |
| Essential | the group total '96,000 tonnes' for a question about Malaysia's and Indonesia's quotas | (left out) | A good answer that leaves it out would be counted as wrong |

**Shape versus content.** The reply's *shape* (a question, a reference answer, a list of key facts) is enforced by OpenAI's structured outputs, from the Pydantic model `GeneratedCase`. The *content* rules need the article, which a JSON schema cannot express, so a Pydantic validator checks them after parsing, with the article passed as validation context. The first three rules are checked by [evals/checks.py](../evals/checks.py); if a reply breaks one, the generator sends the error back to the LLM ('key fact "Aldo Hofmeister" only repeats the question') and asks again, up to three attempts, and skips the article if it still fails. Whether a fact is essential cannot be checked automatically; that stays with the prompt and the review. The same check marks key facts ✓ or ✗ in the review UI.

The 'repeats the question' check rejects a fact only if *all* its words are in the question (ignoring function words and company suffixes such as 'Corp'), so '30.50 dlrs per share' passes for a question about the price per share: '30.50' is new.

```bash
.venv/bin/python -m evals.generate --n 50
```

`gpt-4o-mini` was tried first and ignored the date rules ('this week', 'next week' instead of a date); `gpt-5.4-mini` followed them.

### Step 2: review the cases

Generated cases are drafts. Review is what makes the dataset *golden*. The review UI shows one case at a time, with the full text of the expected articles (key facts highlighted, and flagged if not found verbatim) and the articles the latest evaluation runs retrieved instead. Saving writes straight to `evals/golden.jsonl`, so `git diff` shows what the review changed:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/streamlit run evals/review_app.py
```

For each case with `"reviewed": false`:

- **Delete or rewrite vague questions** that many articles answer equally well.
- **Add equally good articles to `expected_doc_ids`.** Reuters often has several versions of one story, such as a short highlights item and a detailed table. If the second one is retrieved, that is not a miss. See [deciding on another article](#deciding-on-another-article) below.
- **Check the reference answer** against the article: correct, complete, only facts from the article.
- **Check the key facts** against the four rules above. The UI marks each one ✓ or ✗ for the first three; whether it is essential is your call. Remove facts a good answer could leave out.
- **Check the style:** an `undated` question must not mention a date or period such as 'Jan/Feb'.
- **Never change a case to make a system failure go away.** If the right article was excluded by a wrong filter, the case is correct and the system is not.
- Set `"reviewed": true`, and write what you decided in `notes` if you changed anything.

After editing `golden.jsonl` by hand, check all cases against the rules (exit code 1 on violations, so it can run in CI):

```bash
.venv/bin/python -m evals.checks
```

**Start with the misses.** A case the system already ranks first rarely hides a problem with `expected_doc_ids`. For a miss, the articles retrieved instead are the candidates for equally good answers, so you only judge those few, never the whole corpus. Search benchmarks have been built this way for decades ('pooling'). An equally good article that no run ever retrieves stays unknown; it does not distort the scores, and it shows up for review once a better version of the system retrieves it.

#### Deciding on another article

In the review UI, the articles a run retrieved are listed under 'Evaluation runs'; open one to read it with the key facts highlighted. The column without auto filters is usually the better candidate list, since no wrong filter has removed anything. Then ask one question: **would a user asking this question be fully satisfied with this article alone?**

| Add it if… | Don't add it if… |
|---|---|
| It is the same story again: an update, a repeat, a short version | It is only on the same subject |
| It contains the facts of the reference answer. Many highlighted key facts are a strong sign | It answers only part of the question |
| A user would not care which of the two they got | It gives different figures, or another source's view |

**Every expected article must contain every key fact.** Both fields describe the correct answer: `expected_doc_ids` for retrieval scoring, `key_facts` for answer scoring. If an expected article lacked a key fact, retrieving it would count as a hit while an answer built from it could not contain the fact. When the two disagree, one of them is wrong, and you decide which: remove the article if it does not fully answer the question, or shorten the fact if it is worded too specifically ('2.5 percent' versus '2.5 pct' → '2.5'). The review UI shows `key facts n/m` for every article and warns on saving; nothing is changed automatically.

**Duplicates no run retrieved.** The candidates from the runs only cover what a run happened to retrieve. The review UI also lists **possible duplicates** of the expected articles: articles whose first chunk lies within distance 0.15 of an expected article's first chunk. Known duplicates were at most 0.115 apart, related but different stories at least 0.197. Distance alone misleads for recurring reports: last week's cocoa purchases or another quarter's housing starts lie as close as a true duplicate but carry other figures. So candidates containing all key facts are shown, and the others are collapsed. For an overview of all cases:

```bash
.venv/bin/python -m evals.duplicates
```

Examples from the first review:

- **q0008, added.** reuters-3065 has the same title, the same day and the same '2.5 pct' growth figure: the same story.
- **q0004, not added.** The higher-ranked Colombia articles are about coffee export registrations, but none states the registration price the question asks for.
- **q0039, not added.** The higher-ranked article is a growth forecast from the DIW institute, not the Bundesbank: same subject, different answer.

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

### 2026-10-19: key facts repeated the question

**Observation.** During review, key facts turned out to name what the question already names: 'Fidelcor Business Credit Corp' for 'What did Fidelcor Business Credit Corp agree to buy…?'. A scan of all 50 cases flagged 41; most listed the company or person from the question, five facts did not appear verbatim in the article, and many were longer than needed or not essential. The cases already marked reviewed were affected too: the review had focused on questions and expected articles.

**Cause.** The generator prompt asked for 'facts a correct answer must contain'. A correct answer does contain the question's names, because it repeats them. Structured outputs enforced the reply's shape but nothing about its content.

**Fix.**

- All 50 cases: key facts rewritten by hand against the article (49 changed; the old facts are kept in each case's `notes`), and verified by [evals/checks.py](../evals/checks.py).
- Generator: rules in the prompt, plus a Pydantic validator that rejects facts that are not verbatim or only repeat the question, with up to two retries that send the errors back. In a test with 8 new cases, all passed on the first attempt.
- Review UI and checklist: the same check, and the four rules in 'How to review'.

Retrieval scores are unaffected: key facts are only used for answer scoring, which comes next.

## 7. Pitfalls

- **Don't tune on the cases you report.** If you change prompts until every case passes, the scores stop telling you anything. Keep some cases aside and look at them only for the final comparison.
- **LLM steps vary between runs.** Filter extraction can give different filters for the same question on two runs, so scores move by a few points without any change. Run the evaluation several times before concluding that a change helped.
- **A miss can be a wrong case.** Check the retrieved titles before blaming the system: often a second article answers the question just as well and belongs in `expected_doc_ids`.
- **Dated questions use the publication date.** Filters match the date an article was published. A user asking about 'the week ending March 19' gets a filter that excludes the article published on April 1 that reports it. The generator avoids such questions; real users won't, so add a few by hand.
- **A generated question is easier than a real one.** It was written from the article, often with the article's own words, which flatters similarity search. Add real questions from the UI over time, especially the ones that failed.
