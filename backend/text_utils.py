"""
Rule-based (no model) split of the user's in-progress text into:
  - context: the "committed" text so far, used for entity-type classification
  - query:   the partial word currently being typed, used as the fuzzy search term

This deliberately avoids using the LLM/classifier for extraction -- our bake-off
showed a small instruct model was unreliable at pulling out the right span
(it sometimes grabbed already-typed words), whereas simple whitespace splitting
is both cheaper and more predictable for this job.
"""


def extract_context_and_query(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    if text[-1].isspace():
        # user just finished a word (or is between words) -- nothing partial yet
        return text.strip(), ""

    stripped = text.rstrip()
    if " " not in stripped and "\t" not in stripped:
        # single in-progress word, no committed context yet
        return "", stripped

    # split off the last whitespace-delimited token as the in-progress query
    head, _, tail = stripped.rpartition(" ")
    return head.strip(), tail
