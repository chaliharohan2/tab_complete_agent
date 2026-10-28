"""
Deterministic routing for the obvious cases, so the LLM is only asked when
the wording is actually ambiguous.

Layers, first match wins:
  1. The user literally named the type at the slot ("for the client").
  2. High-precision constructions ("sales in" -> location, "units of" ->
     product, "sold by" -> person, "sold to" -> client, "sales of X by" ->
     person).
  3. A type is already in play (leading "In Delhi, ..." or a type keyword
     used as the subject) and a new slot is opening -- return the remaining
     types, not the one already used.
"""
import re

# Longest phrases first so "account manager" wins over "account".
TYPE_KEYWORDS = [
    ("account manager", "person"),
    ("sales person", "person"),
    ("sales people", "person"),
    ("sales rep", "person"),
    ("salesperson", "person"),
    ("salesman", "person"),
    ("saleswoman", "person"),
    ("representative", "person"),
    ("seller", "person"),
    ("rep", "person"),
    ("employee", "person"),
    ("client", "client"),
    ("customer", "client"),
    ("company", "client"),
    ("account", "client"),
    ("buyer", "client"),
    ("product", "product"),
    ("item", "product"),
    ("sku", "product"),
    ("model", "product"),
    ("location", "location"),
    ("city", "location"),
    ("region", "location"),
    ("country", "location"),
    ("state", "location"),
    ("place", "location"),
    ("area", "location"),
    ("territory", "location"),
]

PREPOSITIONS = {
    "for", "in", "of", "by", "to", "from", "at", "with", "on", "about", "among",
    "between", "vs", "versus", "and", "than", "into", "per",
}
ARTICLES = {"the", "a", "an", "this", "that", "our", "my", "each", "every"}
SLOT_PREPS = {"for", "of", "from", "by"}
ALL_TYPES = ["person", "location", "product", "client"]

# Context words that end the committed text rather than start a value. If the
# raw text does not end in whitespace and its last word is NOT one of these,
# the last word is a partially typed value and gets stripped before routing.
CONTEXT_WORDS = PREPOSITIONS | ARTICLES | {kw for phrase, _ in TYPE_KEYWORDS for kw in phrase.split()} | {
    "or", "plural",
}

# "In Delhi, ..." / "In delhi what ..." -- location already set as the scene.
# Negative lookahead skips "In the city of" (that's an explicit type slot).
_LEADING_LOCATION = re.compile(
    r"^\s*in\s+(?!the\s+(?:city|region|country|state|area|territory|place)\b)[\w.-]+",
    re.I,
)

_PRODUCT_OF = {"unit", "units", "quantity", "stock", "inventory", "sku", "skus", "order", "orders"}
_PERSON_BY = {"sold", "closed", "handled", "earned", "signed", "generated", "brought"}
_CLIENT_TO = {"sold", "sell", "billed", "invoiced", "issued"}
_CLIENT_OR_LOC_TO = {"shipped", "sent", "delivered", "exported"}


def split_context(text: str) -> str:
    """Return the committed context, dropping a trailing partially-typed value."""
    if not text or text[-1].isspace():
        return text.strip()
    head, _, last = text.rpartition(" ")
    if not head or last.lower().strip(",.?!") in CONTEXT_WORDS or _plural_keyword(last):
        return text.strip()
    return head.strip()


def _plural_keyword(word: str) -> bool:
    w = word.lower().strip(",.?!")
    return w.endswith("s") and w[:-1] in CONTEXT_WORDS


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _keyword_in(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"s?\b", text) is not None


def explicit_type(context: str) -> str | None:
    words = _words(context)
    # "in the city of" / "in the region of": drop a trailing "of"
    while words and words[-1] == "of":
        words = words[:-1]
    if not words:
        return None
    # tail = words after the last preposition, e.g. "for [the sales person]"
    last_prep = max((i for i, w in enumerate(words) if w in PREPOSITIONS), default=-1)
    tail = [w for w in words[last_prep + 1:] if w not in ARTICLES]
    if not tail or len(tail) > 3:
        return None
    tail_text = " ".join(tail)
    padded = " " + tail_text
    for phrase, etype in TYPE_KEYWORDS:
        if padded.endswith(" " + phrase) or padded.endswith(" " + phrase + "s"):
            return etype
    return None


def _mentioned_types(context: str) -> set[str]:
    text = " ".join(_words(context))
    found: set[str] = set()
    for phrase, etype in TYPE_KEYWORDS:
        if _keyword_in(text, phrase):
            found.add(etype)
    return found


def _used_types(context: str) -> set[str]:
    """Types already in play -- not the type being asked for at the slot."""
    used = set()
    if _LEADING_LOCATION.search(context):
        used.add("location")
    mentioned = _mentioned_types(context)
    slot = explicit_type(context)
    used |= mentioned - ({slot} if slot else set())
    return used


def rule_types(context: str) -> list[str] | None:
    """Return entity types if a deterministic rule applies, else None."""
    if not context.strip():
        return None

    explicit = explicit_type(context)
    if explicit:
        return [explicit]

    words = _words(context)
    if not words:
        return None
    last = words[-1]
    window = words[-6:]

    # Trailing "in" / "in the" / "at" / "at the" -> location.
    if last in {"in", "at"} or (
        len(words) >= 2 and words[-2] in {"in", "at"} and last in ARTICLES
    ):
        return ["location"]

    # "units of" / "stock of" / "inventory of" -> product.
    if last == "of" and any(w in _PRODUCT_OF for w in window[:-1]):
        return ["product"]

    # "sold by" / "closed by" / "earned by" -> person.
    if last == "by" and any(w in _PERSON_BY for w in window[:-1]):
        return ["person"]

    # "sales of <X> by" -> person (the X is already given; next is who sold it).
    # Distinct from generic "split by", which has no "of" in front.
    if last == "by" and "of" in window[:-1]:
        return ["person"]

    # "sold to" / "billed to" -> client; "shipped to" -> client or location.
    if last == "to":
        if any(w in _CLIENT_OR_LOC_TO for w in window[:-1]):
            return ["client", "location"]
        if any(w in _CLIENT_TO for w in window[:-1]):
            return ["client"]
        # "sales of <X> to" -- X is already given, next is the buyer.
        if "of" in window[:-1]:
            return ["client"]

    # A type is already used and a new slot is opening -> the other types.
    if last in SLOT_PREPS:
        used = _used_types(context)
        remaining = [t for t in ALL_TYPES if t not in used]
        if used and remaining:
            return remaining

    # Bare "for" / "of" with no other signal: any type could come next.
    if last in {"for", "of"}:
        return list(ALL_TYPES)

    return None
