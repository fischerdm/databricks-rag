"""Rules a key fact must follow, shared by the generator, the review UI and dataset checks.

A key fact is something a correct answer must contribute beyond the question, copied verbatim
from the article so it can be checked by string matching:
- it appears in the article (ignoring case and line breaks);
- it does not just repeat the question: at least one of its words is not in the question.
Whether a fact is essential cannot be checked automatically; that is left to review.
"""

import re

# Words that carry no answer on their own: function words, question words and company suffixes,
# so that "Volkswagen AG" counts as repeating a question that names "Volkswagen".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "much", "many", "of", "on", "or", "over",
    "said", "say", "that", "the", "their", "this", "to", "was", "were", "what", "when", "which",
    "who", "why", "will", "with",
    "ag", "co", "corp", "corporation", "inc", "ltd", "nv", "plc", "sa", "group", "company",
}


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
