"""Parse, chunk and embed the Reuters-21578 archive into a Delta table (Databricks job).

Usage (as a bundle job, see databricks.yml):
    databricks bundle run ingest_chunks [--params limit=200]

Reads the archive from a Unity Catalog volume (serverless compute cannot reach the UCI
download), reuses the local parser and chunker, embeds each chunk with OpenAI and writes
one row per chunk. Each run replaces the table's contents; the table itself, its schema
and its Change Data Feed setting (needed by the Vector Search index) are kept.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from databricks.sdk.runtime import dbutils
from openai import OpenAI
from pyspark.sql import SparkSession

COLUMNS = """
    chunk_id STRING NOT NULL COMMENT 'document_id:chunk, the primary key for Vector Search',
    document_id STRING NOT NULL,
    chunk INT NOT NULL,
    text STRING NOT NULL,
    title STRING,
    date DATE,
    date_int INT COMMENT 'yyyymmdd, for range filters',
    topics ARRAY<STRING>,
    places ARRAY<STRING>,
    reuters_id INT,
    source STRING,
    embedding ARRAY<FLOAT>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Serverless runs this file through exec() without __file__, so the bundle passes the root.
    parser.add_argument("--repo-root", required=True, help="where the bundle synced the repo")
    parser.add_argument("--archive", required=True, help="path of reuters21578.tar.gz in a volume")
    parser.add_argument("--table", required=True, help="catalog.schema.table to write")
    parser.add_argument("--secret-scope", default="rag")
    parser.add_argument("--secret-key", default="openai_api_key")
    # Must match the app's EMBEDDING_MODEL, CHUNK_SIZE and CHUNK_OVERLAP (app/config.py).
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="index at most N articles (0: all)")
    parser.add_argument("--batch-size", type=int, default=100, help="articles per embedding request")
    parser.add_argument("--write-every", type=int, default=2000, help="articles per Delta write")
    args = parser.parse_args()

    sys.path.insert(0, args.repo_root)
    from app.services.chunking import chunk_text
    from scripts.ingest_reuters import iter_articles

    spark = SparkSession.builder.getOrCreate()
    client = OpenAI(api_key=dbutils.secrets.get(args.secret_scope, args.secret_key), max_retries=5)

    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {args.table} ({COLUMNS}) "
        "COMMENT 'Reuters-21578 chunks with OpenAI embeddings' "
        "TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    schema = spark.table(args.table).schema

    rows: list[tuple] = []
    pending: list[tuple] = []  # rows still waiting for their embeddings
    n_articles = n_chunks = n_writes = 0

    def embed_pending() -> None:
        if not pending:
            return
        response = client.embeddings.create(model=args.embedding_model, input=[row[3] for row in pending])
        rows.extend((*row, item.embedding) for row, item in zip(pending, response.data, strict=True))
        pending.clear()

    def write_rows() -> None:
        nonlocal n_writes
        if not rows and n_writes:
            return
        # The first write replaces the previous run's rows, later ones append.
        mode = "append" if n_writes else "overwrite"
        spark.createDataFrame(rows, schema).write.mode(mode).saveAsTable(args.table)
        n_writes += 1
        rows.clear()
        print(f"{n_articles} articles, {n_chunks} chunks written", flush=True)

    for document_id, content, metadata in iter_articles(Path(args.archive)):
        if args.limit and n_articles >= args.limit:
            break
        iso_date = metadata.get("date")
        for i, chunk in enumerate(chunk_text(content, args.chunk_size, args.chunk_overlap)):
            pending.append(
                (
                    f"{document_id}:{i}",
                    document_id,
                    i,
                    chunk,
                    metadata["title"],
                    date.fromisoformat(iso_date) if iso_date else None,
                    metadata.get("date_int"),
                    metadata.get("topics", []),
                    metadata.get("places", []),
                    metadata["reuters_id"],
                    metadata["source"],
                )
            )
            n_chunks += 1
        n_articles += 1
        if n_articles % args.batch_size == 0:
            embed_pending()
        if n_articles % args.write_every == 0:
            write_rows()
    embed_pending()
    write_rows()
    print("Done.")


if __name__ == "__main__":
    main()
