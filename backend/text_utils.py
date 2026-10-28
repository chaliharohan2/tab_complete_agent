"""
Rule-based (no model) split of the user's in-progress text into:
  - context: the committed text so far
  - query:   the partially typed value, used as the fuzzy search term

Shares classifier_service.rules.split_context with the router so both sides
agree on whether the last word is a value being typed ("...for del") or part
of the context ("...for", "...the client").
"""
from classifier_service.rules import split_context


def extract_context_and_query(text: str) -> tuple[str, str]:
    if not text.strip():
        return "", ""
    if " " not in text.strip():
        # a single word with nothing before it: treat it as the value being typed
        return "", text.strip()
    context = split_context(text)
    query = text.strip()[len(context):].strip()
    return context, query
