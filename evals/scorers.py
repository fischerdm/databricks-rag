"""Answer scorers for mlflow.genai.evaluate: does an answer contain the key facts, and is it grounded?

Four checks, from cheapest to most expensive:
- key_facts_verbatim (code): every key fact appears verbatim in the answer, as evals.checks
  matches them in articles. Free and deterministic, but misses paraphrases ('2.5 percent' for '2.5 pct').
- key_facts_normalized (code): the same after writing Reuters abbreviations and the answer model's
  spellings one way on both sides (evals.checks.ABBREVIATIONS), so '17.6%' matches '17.6 pct'. Free too.
- fact_coverage (LLM judge): a judge decides per fact whether the answer states it, in any wording,
  and quotes the words that do. The quote is checked against the answer, so a judge that claims a
  fact the answer does not contain is caught. MLflow's built-in Correctness judge also takes the
  key facts (as expected_facts) but returns one verdict per answer and no quotes.
- groundedness (MLflow built-in judge): every claim in the answer is supported by the retrieved
  passages. Catches invented facts, which the key facts cannot.

The judges are Claude (settings.judge_model), the answers come from OpenAI (settings.chat_model).
"""

import json
from functools import lru_cache

import anthropic
from mlflow.entities import Feedback
from mlflow.genai.scorers import RetrievalGroundedness, scorer
from pydantic import BaseModel, Field

from app.config import get_settings
from evals.checks import missing_facts, missing_normalized, normalize

FACT_PROMPT = """You check whether an answer from a news assistant states given facts.

Question: {question}

Answer:
<answer>
{answer}
</answer>

Key facts (each is a short phrase copied from the source article):
{facts}

For each key fact, in the order given, decide whether the answer states it.
- A fact counts as stated if the answer gives the same information in any wording: abbreviations
  ('pct' = 'percent', 'mln' = 'million', 'dlrs' = 'dollars'), other number formats
  ('5,240 mln' = '5.24 billion'), or a paraphrase.
- It does not count if the answer gives a different value, only hints at it, or says the
  information is not available.
- If stated, copy into `quote` the shortest passage of the answer that states it, character for
  character, so it can be found in the answer by string search. If not stated, leave `quote` empty.
- `reason`: one short sentence."""


class FactVerdict(BaseModel):
    fact: str = Field(description="The key fact, as given")
    stated: bool = Field(description="Whether the answer states this fact, in any wording")
    quote: str = Field(description="Exact passage of the answer that states it; empty if not stated")
    reason: str = Field(description="One short sentence")


class FactJudgement(BaseModel):
    verdicts: list[FactVerdict]


@lru_cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def judge_facts(question: str, answer: str, facts: list[str]) -> list[dict]:
    """One verdict per key fact: stated, quote, reason, plus whether the quote is in the answer."""
    response = _client().messages.parse(
        model=get_settings().judge_model,
        max_tokens=4000,
        # A judge must stay the same model across runs, so no fallbacks to another model.
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": FACT_PROMPT.format(
                    question=question, answer=answer, facts="\n".join(f"- {fact}" for fact in facts)
                ),
            }
        ],
        output_format=FactJudgement,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"Judge stopped with {response.stop_reason}")
    verdicts = response.parsed_output.verdicts
    if len(verdicts) != len(facts):
        raise RuntimeError(f"Judge returned {len(verdicts)} verdicts for {len(facts)} facts")
    return [
        {
            "fact": fact,
            "stated": v.stated,
            "quote": v.quote,
            # A quote the answer does not contain means the judge's 'stated' cannot be trusted.
            "quote_found": bool(v.quote.strip()) and normalize(v.quote) in normalize(answer),
            "reason": v.reason,
        }
        for fact, v in zip(facts, verdicts, strict=True)
    ]


@scorer
def key_facts_verbatim(outputs: str, expectations: dict) -> Feedback:
    facts = expectations["key_facts"]
    missing = missing_facts(facts, outputs or "")
    return Feedback(
        value=not missing,
        rationale=f"Missing verbatim: {missing}" if missing else "All key facts appear verbatim.",
        metadata={"missing": json.dumps(missing)},
    )


@scorer
def key_facts_normalized(outputs: str, expectations: dict) -> Feedback:
    facts = expectations["key_facts"]
    missing = missing_normalized(facts, outputs or "")
    return Feedback(
        value=not missing,
        rationale=f"Missing after normalising abbreviations: {missing}" if missing else "All key facts appear.",
        metadata={"missing": json.dumps(missing)},
    )


@scorer
def fact_coverage(inputs: dict, outputs: str, expectations: dict) -> Feedback:
    verdicts = judge_facts(inputs["question"], outputs or "", expectations["key_facts"])
    not_stated = [v["fact"] for v in verdicts if not v["stated"]]
    return Feedback(
        value=not not_stated,
        rationale="; ".join(f"{v['fact']}: {v['reason']}" for v in verdicts),
        metadata={"verdicts": json.dumps(verdicts)},
    )


def groundedness() -> RetrievalGroundedness:
    return RetrievalGroundedness(name="groundedness", model=f"anthropic:/{get_settings().judge_model}")
