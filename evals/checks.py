"""Rules a case must follow, shared by the generator, the review UI and dataset checks.

A key fact is something a correct answer must contribute beyond the question, copied verbatim
from the article so it can be checked by string matching:
- it appears in the article (ignoring case and line breaks);
- it does not just repeat the question: at least one of its words is not in the question.
Whether a fact is essential cannot be checked automatically; that is left to review.

Every expected article should contain every key fact. Otherwise retrieval would count that
article as a hit while an answer built from it could not contain the fact. When they disagree,
a reviewer decides which side is wrong: the article (remove it) or the fact (shorten it).

The style describes the question: 'dated' if it names a date or period, 'undated' if not.

Usage, to check the whole dataset:
    python -m evals.checks [--golden evals/golden.jsonl]

It needs no API key, no index and nothing beyond the standard library, so it runs in CI
(.github/workflows/ci.yml); it downloads the Reuters archive on first run.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.ingest_reuters import download_archive, iter_articles

DEFAULT_OUT = Path("evals/golden.jsonl")

# Words that carry no answer on their own: function words, question words and company suffixes,
# so that "Volkswagen AG" counts as repeating a question that names "Volkswagen".
# Not formatted: the formatter would put each word on its own line.
# fmt: off
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "much", "many", "of", "on", "or", "over",
    "said", "say", "that", "the", "their", "this", "to", "was", "were", "what", "when", "which",
    "who", "why", "will", "with",
    "ag", "co", "corp", "corporation", "inc", "ltd", "nv", "plc", "sa", "group", "company",
}
# fmt: on

# A date or period in a question: a month or weekday ('on April 7', 'Jan/Feb'), a year, or a
# relative period ('third-quarter', 'the 53 weeks to'). Month names are matched capitalised only,
# so the verbs 'may' and 'march' do not count; a name such as 'May Department Stores' still does.
MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
WEEKDAYS = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
PERIODS = r"(?i:quarter|qtr|weeks?|months?|(?:this|last|next|fiscal) year|(?:first|second) half|today|yesterday)"
DATE_PATTERN = re.compile(rf"\b(?:{MONTHS}|{WEEKDAYS}|19\d\d|{PERIODS})\b")


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def content_words(text: str) -> set[str]:
    """Lower-case words without stopwords, tickers like <MYM>, and possessive 's."""
    text = re.sub(r"<[^>]*>", " ", text.lower())
    words = set()
    for word in re.findall(r"[a-z0-9][a-z0-9.,'/-]*", text):
        word = re.sub(r"'s$", "", word).strip(".,'/-")
        if word and word not in STOPWORDS:
            words.add(word)
    return words


def fact_problem(fact: str, article: str, question: str) -> str | None:
    """Why a key fact breaks the rules, or None if it is fine."""
    if not fact.strip():
        return "is empty"
    if normalize(fact) not in normalize(article):
        return "does not appear verbatim in the article"
    if content_words(fact) <= content_words(question):
        return "only repeats the question"
    return None


def style_problem(question: str, style: str) -> str | None:
    """Why a question does not match its style, or None if it does."""
    mentions = DATE_PATTERN.findall(question)
    if style == "undated" and mentions:
        return f"is undated but mentions {', '.join(repr(m) for m in mentions)}"
    if style == "dated" and not mentions:
        return "is dated but mentions no date or period"
    return None


def missing_facts(facts: list[str], article: str) -> list[str]:
    """The key facts that do not appear verbatim in this article."""
    text = normalize(article)
    return [fact for fact in facts if normalize(fact) not in text]


# Reuters' telex abbreviations in the key facts, and the spellings the answer model uses for them,
# each written as the one spelling on the left. Fixed from the key facts before measuring on stored
# answers, so it is not tuned to the misses. Numbers written as words ('nine cts' vs '9 cents') and
# other number formats ('330,000,000' vs '330 million') are left to the judge.
NUMBER = r"\d+(?:[.,]\d+)*"
ABBREVIATIONS = [
    # Symbols first, so that '$1.15' becomes '1.15 dollars' and then '1.15 dlr'.
    (" pct", r"(?<=\d)\s?%"),
    (r"\1\2 dlr", rf"\$\s?({NUMBER})(\s(?:million|mln|billion|bln))?"),
    ("pct", r"\b(?:percent|per cent)\b"),
    ("mln", r"\bmillions?\b"),
    ("billion", r"\bbln\b"),
    ("dlr", r"\b(?:dollars?|dlrs?)\b"),
    ("ct", r"\b(?:cents?|cts)\b"),
    ("lb", r"\b(?:pounds?|lbs?)\b"),
    ("new zealand", r"\bn\.z\.(?!\w)"),
]


def canonical(text: str) -> str:
    """Text lower-cased, with collapsed whitespace and every abbreviation written one way."""
    text = normalize(text)
    for replacement, pattern in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def missing_normalized(facts: list[str], answer: str) -> list[str]:
    """The key facts that do not appear in the answer, once both are written canonically."""
    text = canonical(answer)
    return [fact for fact in facts if canonical(fact) not in text]


def case_problems(case: dict, texts: dict[str, str]) -> list[str]:
    """Every rule a case breaks, given the texts of its expected articles keyed by id."""
    problems = [f"unknown article {doc_id}" for doc_id in case["expected_doc_ids"] if doc_id not in texts]
    if problem := style_problem(case["question"], case["style"]):
        problems.append(f"question {problem}")
    if not case.get("key_facts"):
        problems.append("no key facts")
    for fact in case.get("key_facts", []):
        if content_words(fact) <= content_words(case["question"]):
            problems.append(f"key fact {fact!r} only repeats the question")
    for doc_id in case["expected_doc_ids"]:
        if doc_id in texts and (missing := missing_facts(case.get("key_facts", []), texts[doc_id])):
            problems.append(f"{doc_id} lacks key facts {missing}: remove the article or shorten the facts")
    return problems


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_OUT, help="golden dataset (JSONL)")
    args = parser.parse_args()

    download_archive()
    cases = load_cases(args.golden)
    needed = {doc_id for case in cases for doc_id in case["expected_doc_ids"]}
    texts = {doc_id: content for doc_id, content, _ in iter_articles() if doc_id in needed}

    n_bad = 0
    for case in cases:
        if problems := case_problems(case, texts):
            n_bad += 1
            print(f"{case['id']}: {case['question']}")
            for problem in problems:
                print(f"  - {problem}")
    print(f"{n_bad} of {len(cases)} cases break a rule." if n_bad else f"All {len(cases)} cases pass.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
