"""
Held-out test set: written separately from test_cases.py and NOT used while
choosing prompts/models/thresholds. Only used for a final generalization check.
Same (text, required, optional) format.
"""
from test_cases import ALL, NONE, C, L, P, R

CASES = [
    ("Pull up the numbers for", ALL, NONE),
    ("Could you get me revenue for", ALL, NONE),
    ("Break down last month's sales for", ALL, NONE),
    ("Show revenue split by", ALL, NONE),
    ("What's the growth rate for", ALL, NONE),
    ("Sales attributed to the account manager", {P}, NONE),
    ("Revenue from the customer account", {C}, NONE),
    ("Margins on the SKU", {R}, NONE),
    ("Quarterly sales in the state of", {L}, NONE),
    ("Which rep had the best quarter in", {L}, NONE),
    ("What was the most popular item in", {L}, NONE),
    ("Which customers are based in", {L}, NONE),
    ("How many orders of", {R}, NONE),
    ("What is the inventory level of", {R}, NONE),
    ("Which region bought the most", NONE, {R}),
    ("Who bought the most units of", {R}, NONE),
    ("Commission earned by", {P}, NONE),
    ("Deals signed with", {C}, {P}),
    ("Invoices issued to", {C}, NONE),
    ("Total revenue billed to", {C}, NONE),
    ("In Mumbai, who sold the most", NONE, {R}),
    ("In Bengaluru, which product did best for", {C, P}, NONE),
    ("How much of the MacBook Pro was sold in", {L}, NONE),
    ("How much of the MacBook Pro did we sell to", {C}, {L}),
    ("Priya Delacroix's revenue from", {C, R, L}, NONE),
    ("What did Marcus Webb sell in", {L}, NONE),
    ("Contrast Seattle with", {L}, NONE),
    ("Delta Corp versus", {C}, NONE),
    ("Show me Grace Kim vs", {P}, NONE),
    ("Mechanical Keyboard compared to", {R}, NONE),
    ("What is our year over year growth", NONE, NONE),
    ("Give me a summary of this quarter", NONE, NONE),
]
