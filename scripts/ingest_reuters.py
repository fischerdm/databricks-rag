"""Index the Reuters-21578 corpus (1987 Reuters newswire) into the vector store.

Usage:
    python -m scripts.ingest_reuters [--limit N] [--batch-size N] [--dry-run]

Downloads the original SGML archive from the UCI repository on first run and reads it
directly from the tarball. Articles without a body are skipped. Document ids are derived
from the corpus NEWID, so re-running the script updates existing entries instead of
duplicating them.
"""

import argparse
import html
import re
import tarfile
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

ARCHIVE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
ARCHIVE_PATH = Path("data/reuters21578/reuters21578.tar.gz")

LABEL_FIELDS = ("topics", "places")

ARTICLE_RE = re.compile(r"<REUTERS[^>]*NEWID=\"(\d+)\"[^>]*>(.*?)</REUTERS>", re.DOTALL)


def _tag(name: str, text: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return html.unescape(match.group(1)).strip() if match else ""


def _labels(name: str, text: str) -> list[str]:
    """Return the <D> entries of a label field (TOPICS, PLACES, ...)."""
    return re.findall(r"<D>(.*?)</D>", _tag(name, text))


def _parse_date(raw: str) -> datetime | None:
    # Raw values look like "26-FEB-1987 15:01:01.79", with 1- or 2-digit days and some junk.
    match = re.match(r"\s*(\d{1,2}-[A-Z]{3}-\d{4})", raw)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d-%b-%Y")
    except ValueError:
        return None


def download_archive() -> None:
    if ARCHIVE_PATH.exists():
        return
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARCHIVE_URL} ...")
    urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)


def iter_articles(archive_path: Path = ARCHIVE_PATH) -> Iterator[tuple[str, str, dict]]:
    """Yield (document_id, content, metadata) for every article that has a body."""
    with tarfile.open(archive_path) as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if not member.name.endswith(".sgm"):
                continue
            raw = archive.extractfile(member).read().decode("latin-1")
            for new_id, article in ARTICLE_RE.findall(raw):
                body = _tag("BODY", article)
                if not body:
                    continue
                # Drop the trailing " Reuter" sign-off and control characters.
                body = re.sub(r"\s*Reuter\s*$", "", body.replace("\x03", "")).strip()
                title = _tag("TITLE", article)
                date = _parse_date(_tag("DATE", article))

                metadata = {
                    "source": "reuters-21578",
                    "reuters_id": int(new_id),
                    "title": title,
                }
                # Chroma rejects empty lists, so label fields are only set when present.
                for key in LABEL_FIELDS:
                    if labels := _labels(key.upper(), article):
                        metadata[key] = labels
                if date:
                    metadata["date"] = date.date().isoformat()
                    # Numeric copy so Chroma can range-filter ($gte/$lte only work on numbers).
                    metadata["date_int"] = int(date.strftime("%Y%m%d"))

                header = f"{title}\n{date.date().isoformat()}\n\n" if date else f"{title}\n\n"
                yield f"reuters-{new_id}", header + body, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="index at most N articles")
    parser.add_argument("--batch-size", type=int, default=100, help="articles per embedding request")
    parser.add_argument("--dry-run", action="store_true", help="parse and count without indexing")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="rewrite metadata of already indexed articles without re-embedding (no OpenAI cost)",
    )
    args = parser.parse_args()

    download_archive()

    batch: list[tuple[str, str, dict]] = []
    n_articles = n_chunks = 0

    def flush() -> None:
        nonlocal n_chunks
        if batch and not args.dry_run:
            from app.services import vector

            if args.metadata_only:
                # Explicit None deletes label keys left over from an earlier index format.
                batch[:] = [(i, c, {**dict.fromkeys(LABEL_FIELDS), **m}) for i, c, m in batch]
                n_chunks += vector.update_metadata(batch)
            else:
                n_chunks += vector.add_documents(batch)
        batch.clear()
        print(f"\r{n_articles} articles, {n_chunks} chunks", end="", flush=True)

    for document in iter_articles():
        if args.limit and n_articles >= args.limit:
            break
        batch.append(document)
        n_articles += 1
        if len(batch) >= args.batch_size:
            flush()
    flush()
    print("\nDone.")


if __name__ == "__main__":
    main()
