"""Generate draft golden-dataset cases from indexed Reuters articles.

Usage:
    python -m evals.generate [--n 50] [--seed 0] [--out evals/golden.jsonl] [--model gpt-5.4-mini]

Samples articles from the corpus and asks an LLM to write one question per article that the
article answers, plus a reference answer. The article's id becomes the expected source, so
retrieval can be scored without labelling anything by hand.

New cases are appended with "reviewed": false. Read them, fix or delete bad ones, add other
articles that answer a question equally well to expected_doc_ids, then set "reviewed": true.
Articles already in the file are skipped, so running the script again grows the set.
"""

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from app.services import vector
from app.services.llm import _client
from evals.checks import DEFAULT_OUT, fact_problem, load_cases, style_problem
from scripts.ingest_reuters import download_archive, iter_articles

# gpt-4o-mini (the default chat model) ignored the date rules below in testing; gpt-5.4-mini kept to them.
DEFAULT_MODEL = "gpt-5.4-mini"
# Short articles rarely hold enough facts for a specific question.
MIN_LENGTH = 400
# Earnings reports are about 40% of the corpus; cap any single topic so the set stays varied.
MAX_TOPIC_SHARE = 0.25

Style = Literal["dated", "undated"]


# Attempts per article: the first answer plus retries that send the validation errors back.
MAX_ATTEMPTS = 3


class GeneratedCase(BaseModel):
    """The LLM's reply. The JSON schema fixes its shape; the validator checks the content.

    The content rules need the article and the style, passed as validation context. OpenAI's SDK
    parses the reply without context, so the check runs in a second, explicit validation step.
    """

    question: str = Field(description="The question a user would type")
    reference_answer: str = Field(description="One or two sentences answering it, using only the article")
    key_facts: list[str] = Field(
        description="1-3 short facts the answer adds beyond the question, copied verbatim from the article"
    )

    @field_validator("question")
    @classmethod
    def check_question(cls, question: str, info: ValidationInfo) -> str:
        if info.context and (problem := style_problem(question, info.context["style"])):
            raise ValueError(f"the question {problem}")
        return question

    @field_validator("key_facts")
    @classmethod
    def check_key_facts(cls, facts: list[str], info: ValidationInfo) -> list[str]:
        if not info.context:
            return facts
        if not 1 <= len(facts) <= 3:
            raise ValueError(f"give one to three key facts, not {len(facts)}")
        problems = [
            f"key fact {fact!r} {problem}"
            for fact in facts
            if (problem := fact_problem(fact, info.context["article"], info.data.get("question", "")))
        ]
        if problems:
            raise ValueError("; ".join(problems))
        return facts


SYSTEM_PROMPT = """You write test questions for a search assistant over an archive of 19,000
Reuters business news articles from February to October 1987.

You get one article. Write one question that a user of the archive might ask and that this
article answers.
- Make it specific enough that this article is the best match in the archive: name the company,
  person, country, commodity or figure involved. Avoid questions that hundreds of articles answer.
- Write it as a user would: one short question about one thing, at most 20 words.
  Never refer to "the article", "the report" or "the text".
- It must be answerable from the article alone.

reference_answer: one or two sentences, only facts stated in the article.

key_facts: one to three facts a correct answer must contribute beyond the question.
- Copy each one verbatim from the article, so it can be found by exact string matching.
- Keep each as short as possible: usually a number ("28,526"), a name, or a phrase of two to four words.
- Never repeat names or terms that are already in the question: the answer repeats them anyway.
- Only facts a good answer cannot leave out; no background details."""

STYLE_RULES: dict[Style, str] = {
    # The article's date (second line) is its publication date, which is what date filters match.
    "dated": (
        "Mention when it was reported, using the article's date on its second line, as a user would: "
        "the day ('on April 7') or the month ('in March'). Do not use other dates from the text."
    ),
    "undated": "Do not mention any date, month, quarter, year or period such as 'the first three months'.",
}


def sample_articles(n: int, seed: int, exclude: set[str]) -> list[tuple[str, str, dict]]:
    """Pick n random indexed articles with a date and a topic, capping each topic's share."""
    candidates = [
        article
        for article in iter_articles()
        if article[0] not in exclude
        and len(article[1]) >= MIN_LENGTH
        and "date" in article[2]
        and article[2].get("topics")
    ]
    random.Random(seed).shuffle(candidates)

    max_per_topic = max(1, int(n * MAX_TOPIC_SHARE))
    per_topic: Counter[str] = Counter()
    picked = []
    # Check the index in batches: the script may have indexed only part of the corpus (--limit).
    for start in range(0, len(candidates), 200):
        batch = candidates[start : start + 200]
        indexed = vector.document_metadata([doc_id for doc_id, _, _ in batch])
        for article in batch:
            topic = article[2]["topics"][0]
            if article[0] in indexed and per_topic[topic] < max_per_topic:
                picked.append(article)
                per_topic[topic] += 1
                if len(picked) == n:
                    return picked
    return picked


async def generate_case(article: tuple[str, str, dict], style: Style, model: str) -> GeneratedCase | None:
    """Ask for a case; if it breaks the style or key-fact rules, send the errors back and ask again.

    Returns None if the reply still breaks the rules after MAX_ATTEMPTS.
    """
    doc_id, content, _ = article
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{STYLE_RULES[style]}"},
        {"role": "user", "content": content},
    ]
    for _ in range(MAX_ATTEMPTS):
        completion = await _client().chat.completions.parse(
            model=model, messages=messages, response_format=GeneratedCase
        )
        draft = completion.choices[0].message.parsed
        try:
            return GeneratedCase.model_validate(draft.model_dump(), context={"article": content, "style": style})
        except ValidationError as exc:
            problems = "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
            messages += [
                {"role": "assistant", "content": draft.model_dump_json()},
                {"role": "user", "content": f"Your answer breaks the rules: {problems}. Answer again, fixing this."},
            ]
    print(f"Skipped {doc_id}: still breaks the rules after {MAX_ATTEMPTS} attempts ({problems}).")
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="number of cases to add")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling and styles")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL file to append to")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model that writes the questions")
    args = parser.parse_args()

    download_archive()
    existing = load_cases(args.out)
    seen = {doc_id for case in existing for doc_id in case["expected_doc_ids"]}
    articles = sample_articles(args.n, args.seed, exclude=seen)
    rng = random.Random(args.seed)
    styles: list[Style] = [rng.choice(["dated", "undated"]) for _ in articles]

    semaphore = asyncio.Semaphore(8)

    async def limited(article, style):
        async with semaphore:
            return await generate_case(article, style, args.model)

    drafts = await asyncio.gather(*(limited(a, s) for a, s in zip(articles, styles, strict=True)))
    generated = [(a, s, case) for a, s, case in zip(articles, styles, drafts, strict=True) if case is not None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for i, ((doc_id, _, metadata), style, case) in enumerate(generated):
            record = {
                "id": f"q{len(existing) + i + 1:04d}",
                "question": case.question,
                "expected_doc_ids": [doc_id],
                "reference_answer": case.reference_answer,
                "key_facts": case.key_facts,
                "style": style,
                "source_title": metadata["title"],
                "source_date": metadata["date"],
                "generated_by": args.model,
                "reviewed": False,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Added {len(generated)} cases to {args.out} ({len(existing) + len(generated)} in total).")


if __name__ == "__main__":
    asyncio.run(main())
