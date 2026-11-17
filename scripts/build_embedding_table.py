import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from evaluate_surrogates import METADATA_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-embedding-2"
DIMS = 1024
BATCH = 40  # programs per request
NAME = f"{MODEL}_{DIMS}"
CACHE = ROOT / "data" / f"embeddings_cache_{NAME}.pkl"
OUT = ROOT / "data" / f"embeddings_{NAME}.csv"


def code_key(code):
    return hashlib.sha256(code.encode()).hexdigest()

# starting with gemini api first
# to do: generalize to any embedding API
def main():
    
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        sys.exit("api key is missing from .env")
    client = genai.Client(api_key=api_key)

    table = pd.read_csv(ROOT / "data" / "sbox_table.csv")
    codes = table["code"].drop_duplicates().tolist()
    pilot = len(sys.argv) > 1
    if pilot:
        codes = codes[: int(sys.argv[1])]

    # caching so I don't pay twice
    cache = pd.read_pickle(CACHE) if CACHE.exists() else {}
    todo = [c for c in codes if code_key(c) not in cache]
    print(f"{len(codes)} programs, {len(todo)} still to embed")

    start = time.time()
    for i in range(0, len(todo), BATCH):

        batch = todo[i : i + BATCH]
        # each program is its own Content; a plain list of strings is merged into one input
        contents = [types.Content(parts=[types.Part(text=code)]) for code in batch]
        try:
            result = client.models.embed_content(
                model=MODEL, contents=contents, config=types.EmbedContentConfig(output_dimensionality=DIMS)
            )

        except errors.ClientError as e:
            if e.code != 429:
                raise
            # over the TPM limit: wait for the minute to reset, then retry this batch
            print("Rate limit hit, waiting 60s", flush=True)
            
            time.sleep(60)
            result = client.models.embed_content(
                model=MODEL, contents=contents, config=types.EmbedContentConfig(output_dimensionality=DIMS)
            )

        # one vector per program, not one combined vector for the whole batch
        if len(result.embeddings) != len(batch):
            sys.exit(f"Expected {len(batch)} embeddings, got {len(result.embeddings)}")

        for code, embedding in zip(batch, result.embeddings):
            cache[code_key(code)] = embedding.values
        
        pd.to_pickle(cache, CACHE)  # saved after every batch, so a crash loses nothing
        print(f"{min(i + BATCH, len(todo))}/{len(todo)} embedded", flush=True)

    print(f"Took {time.time() - start:.1f}s")

    sizes = {len(cache[code_key(c)]) for c in codes}
    print("Vector sizes:", sizes)

    if pilot:
        return  # the table is only written for the full dataset

    vectors = pd.DataFrame([cache[code_key(c)] for c in table["code"]], index=table.index)
    vectors.columns = [f"emb_{j}" for j in vectors.columns]
    pd.concat([table[METADATA_COLS], vectors], axis=1).to_csv(OUT, index=False)

    # API models can change or be retired, so record exactly what made this table
    info = {"model": MODEL, "dims": DIMS, "embedded_text": "code only", "task_type": None, "date": str(date.today())}
    OUT.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
