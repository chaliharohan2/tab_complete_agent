"""Local (no GPU) checks for the deterministic rule layer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classifier_service"))
from rules import rule_types, split_context  # noqa: E402

P, L, R, C = "person", "location", "product", "client"


def check(text, expected):
    pred = rule_types(split_context(text))
    exp = expected if expected is None else list(expected) if isinstance(expected, list) else sorted(expected)
    got = None if pred is None else sorted(pred)
    exp_s = None if exp is None else sorted(exp) if isinstance(exp, list) else exp
    ok = got == exp_s
    status = "OK" if ok else "FAIL"
    print(f"{status:4s} {text!r:55s} got {got} exp {exp_s}")
    return ok


cases = [
    # user's reported failures
    ("In delhi what were the sales of ", [P, R, C]),
    ("In delhi what were the sales of", [P, R, C]),
    ("Show me the sales for", [P, L, R, C]),
    ("Can you show me the sales for", [P, L, R, C]),
    ("How many units of", [R]),
    ("Which product sold the most in", [L]),
    # trailing in -> location
    ("Show me the sales in", [L]),
    ("What were the sales in", [L]),
    ("Which salesperson sold the most in", [L]),
    ("Best performing sales people in", [L]),
    ("How many clients do we have in", [L]),
    # units of -> product
    ("How many units of del", [R]),
    ("How much stock of", [R]),
    ("How many orders of", [R]),
    ("What is the inventory level of", [R]),
    # sold by / of X by -> person
    ("Units sold by", [P]),
    ("Deals closed by", [P]),
    ("Commission earned by", [P]),
    ("Show me sales of DEL Laptop by", [P]),
    ("Sales of DEL Laptop by", [P]),
    # sold to -> client; shipped to -> client+location
    ("How much did we sell to", [C]),
    ("Invoices issued to", [C]),
    ("Total revenue billed to", [C]),
    ("Shipped to", [C, L]),
    ("Orders shipped to", [C, L]),
    # explicit type keywords
    ("Show me the sales for the sales person", [P]),
    ("Show me the sales for the client", [C]),
    ("Show me the sales for the product", [R]),
    ("Show me the sales in the city of", [L]),
    ("Average deal size for the rep", [P]),
    # already-used type + new slot
    ("In Mumbai, how much revenue came from", [P, R, C]),
    ("In Bengaluru, which product did best for", [P, C]),
    ("Top selling products for", [P, L, C]),
    ("Which location had the highest sales of", [P, R, C]),
    ("In Paris, sales for", [P, R, C]),
    # should NOT fire (leave to LLM)
    ("Show me total revenue", None),
    ("Show revenue split by", None),  # generic "by" stays with the LLM
    ("What's the growth rate for", [P, L, R, C]),
    ("Compare Delhi and", None),
    ("Filter the results by", None),
    ("Show me sales of DEL Laptop to", [C]),
]

ok = sum(check(t, e) for t, e in cases)
print(f"\n{ok}/{len(cases)} rule checks passed")
sys.exit(0 if ok == len(cases) else 1)
