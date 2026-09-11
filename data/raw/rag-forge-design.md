# rag-forge design notes

These notes double as the seed corpus, so `make ingest && make ask` works
immediately after a clone, with nothing to download.

## Hybrid retrieval

Dense retrieval embeds the query and the passages into the same vector space and
compares them by cosine similarity. It handles paraphrase well, and fails on exact
tokens it has never seen — product codes, statute numbers, rare proper nouns.

Sparse retrieval scores exact term overlap. It handles those exact tokens and fails
on paraphrase.

Reciprocal Rank Fusion combines the two ranked lists without requiring their scores
to be comparable. Each document receives the sum of 1 / (k + rank) over every list it
appears in, with k set to 60 by default. Because only rank position is used, no score
normalisation is needed, which is why the method is robust across datasets.

## Reranking

The bi-encoder embeds query and passage separately, so it is fast enough to search
millions of chunks, but it never lets the two texts interact. The cross-encoder reads
query and passage together and produces a single relevance score. It is far too slow
to search the whole corpus and far more accurate on a shortlist. Retrieval is
therefore a recall stage, and reranking is the precision stage.

## Citations

The model is never trusted to report where a fact came from. It is asked to return
verbatim quotes, and every quote is then located inside the passage it claims. Only a
quote that is actually found is converted into a character span of the original
document and shown as a citation. A quote that cannot be located is dropped, and a
dropped quote is itself a hallucination signal.

## The grounding gate

An answer is released only when the retrieved evidence supports it. Support is scored
with the cross-encoder, comparing the generated answer against the retrieved passages,
and the result is squashed into a probability. Below the configured threshold the
system refuses instead of answering.

There are three independent grounds for refusal: the model reporting that the context
did not support an answer, a grounding score under the threshold, and every quote
failing verification. Any one of them is enough. Refusing is treated as a correct
outcome, not a failure — a system that cannot say "I don't know" cannot be trusted
with the answers it does give.
