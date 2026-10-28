"""Ad-hoc probe of obvious phrasings against the live router. Not the formal eval."""
import httpx

URL = "http://spark-19de.local:8800/classify"

CASES = [
    # generic "for" should be all
    ("Show me the sales for", {"person", "location", "product", "client"}),
    ("Can you show me the sales for", {"person", "location", "product", "client"}),
    ("Give me numbers for", {"person", "location", "product", "client"}),
    ("Break down sales for", {"person", "location", "product", "client"}),
    ("What are the figures for", {"person", "location", "product", "client"}),
    ("Sales data for", {"person", "location", "product", "client"}),
    ("Show me sales of", {"person", "location", "product", "client"}),
    # trailing "in" should be location
    ("Show me the sales in", {"location"}),
    ("What were the sales in", {"location"}),
    ("Which product sold the most in", {"location"}),
    ("Which salesperson sold the most in", {"location"}),
    ("Revenue last quarter in", {"location"}),
    ("How did we do in", {"location"}),
    # units/stock of -> product
    ("How many units of", {"product"}),
    ("How much stock of", {"product"}),
    ("What is the inventory of", {"product"}),
    ("Quantity of", {"product"}),
    ("Number of units sold of", {"product"}),
    # explicit type words
    ("Show me the sales for the sales person", {"person"}),
    ("for the client", {"client"}),
    ("for the product", {"product"}),
    ("in the city of", {"location"}),
    ("for the salesperson named", {"person"}),
    # already have a location, next slot is not location
    ("In delhi what were the sales of", {"person", "product"}),
    ("In Delhi what were the sales of", {"person", "product"}),
    ("In mumbai, sales of", {"person", "product"}),
    ("In London, who sold", {"person"}),
    ("In Tokyo, which product", {"product"}),
    ("In Paris, sales for", {"person", "product", "client"}),
    # already have a person
    ("How much did Delilah sell in", {"location"}),
    ("How much did Delilah sell to", {"client"}),
    ("Delilah's numbers in", {"location"}),
    # already have a product
    ("Sales of DEL Laptop in", {"location"}),
    ("Sales of DEL Laptop to", {"client"}),
    ("Sales of DEL Laptop by", {"person"}),
    ("Units of the laptop sold in", {"location"}),
    # "by" after activity -> person
    ("Units sold by", {"person"}),
    ("Deals closed by", {"person"}),
    ("Sales handled by", {"person"}),
    ("Commission earned by", {"person"}),
    # "to" after sell -> client
    ("How much did we sell to", {"client"}),
    ("Orders shipped to", {"client", "location"}),
    # comparisons
    ("Compare Delhi and", {"location"}),
    ("Compare Mumbai with", {"location"}),
    ("Delhi versus", {"location"}),
    # none
    ("Show me total revenue", set()),
    ("How are we doing this year", set()),
    ("What is the monthly trend", set()),
    # leftover from holdout-ish
    ("Show revenue split by", {"person", "location", "product", "client"}),
    ("What's the growth rate for", {"person", "location", "product", "client"}),
    ("In Bengaluru, which product did best for", {"person", "client"}),
    ("Top selling products for", {"person", "location", "client"}),
    ("Best performing sales people in", {"location"}),
]

def fmt(s):
    return "{" + ",".join(sorted(s)) + "}" if s else "{}"

ok = 0
for text, must in CASES:
    r = httpx.post(URL, json={"text": text}, timeout=15).json()
    pred = set(r["entity_types"])
    # loose: must be subset of pred and pred shouldn't include types that
    # are clearly wrong. For this probe we just check must <= pred, and
    # flag extras separately if must is a single type.
    hit = must <= pred if must else pred == set()
    extra = pred - must
    if must and len(must) == 1 and extra:
        hit = False
    status = "OK" if hit else "FAIL"
    if hit:
        ok += 1
    scores = " ".join(f"{k[:3]}={v:.2f}" for k, v in r["scores"].items())
    print(f"{status:4s} {text!r:50s} got {fmt(pred):32s} want {fmt(must):32s} [{r['source']}] {scores}")

print(f"\n{ok}/{len(CASES)} obvious cases passed (loose check)")
