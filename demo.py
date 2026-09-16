"""Four quotes a model might produce. Only the ones that really exist survive.

    python demo.py

Every citation shown to a user is located character-by-character in the
retrieved source first. A quote that cannot be found is dropped, not
softened -- because a citation that looks right and is not is worse than no
citation at all.

No model, no database, no network: the "model output" is written here so the
verification step can be watched on its own.
"""

import sys

sys.path.insert(0, "src")

from ragforge.generate.citations import verify_quotes
from ragforge.types import Chunk, ScoredChunk

REFUND_POLICY = (
    "Section 4.2 Refunds. A customer may request a refund within 30 days of "
    "delivery. Refunds are issued to the original payment method within 14 "
    "business days of approval. Shipping charges are not refundable."
)
WARRANTY = (
    "Section 7.1 Warranty. Hardware is covered for 24 months from the date of "
    "purchase. The warranty does not cover damage caused by liquid ingress."
)

retrieved = [
    ScoredChunk(
        chunk=Chunk(
            ordinal=0,
            content=REFUND_POLICY,
            char_start=0,
            char_end=len(REFUND_POLICY),
            source="policy.md",
            section="4.2 Refunds",
        ),
        score=0.91,
    ),
    ScoredChunk(
        chunk=Chunk(
            ordinal=1,
            content=WARRANTY,
            char_start=0,
            char_end=len(WARRANTY),
            source="policy.md",
            section="7.1 Warranty",
        ),
        score=0.77,
    ),
]

QUOTES = [
    {"passage": 1, "quote": "within 30 days of delivery"},
    {"passage": 2, "quote": "covered for 24 months from the date of purchase"},
    # Right passage, wrong number -- the text is real, so it should still verify.
    {"passage": 1, "quote": "does not cover damage caused by liquid ingress"},
    # Plausible, policy-shaped, and nowhere in either source.
    {"passage": 1, "quote": "refunds are processed within 3 business days"},
]

print("INPUT")
print(f"   {len(retrieved)} retrieved passages from policy.md")
print(f"   {len(QUOTES)} quotes the model claims to have taken from them:")
for q in QUOTES:
    print(f'      passage {q["passage"]}  "{q["quote"]}"')
print()

citations, unverified = verify_quotes(QUOTES, retrieved)

print("OUTPUT")
print(f"   {len(citations)} verified, {len(unverified)} dropped")
print()
for c in citations:
    print(f"   VERIFIED  {c.source} sec {c.section}  chars {c.char_start}-{c.char_end}")
    print(f'             "{c.quote}"')
for u in unverified:
    print(f'   DROPPED   "{u}"')
print()
print("   The third quote named the wrong passage and still verified: a real")
print("   quote with a bad index is a real quote. The fourth reads exactly")
print("   like the others and appears in neither source, so it never reaches")
print("   the user.")
