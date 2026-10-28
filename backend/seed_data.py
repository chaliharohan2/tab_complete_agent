"""
Dummy demo data standing in for the real Elasticsearch-backed entity columns.
Includes the example values from the spec ("Delilah", "Delhi", "DEL Laptop")
so the classic "del" prefix demo works out of the box.
"""

PERSONS = [
    "Delilah Nguyen",
    "Derek Alston",
    "Priya Delacroix",
    "Marcus Webb",
    "Sofia Ramirez",
    "Amit Deshpande",
    "Grace Kim",
    "Oliver Bennett",
    "Natasha Volkov",
    "Delano Marsh",
    "Chen Wei",
    "Isabella Rossi",
]

LOCATIONS = [
    "Delhi",
    "New Delhi",
    "Mumbai",
    "Bengaluru",
    "Dallas",
    "Denver",
    "Seattle",
    "Toronto",
    "Berlin",
    "Delft",
    "Singapore",
    "Cape Town",
]

PRODUCTS = [
    "DEL Laptop Pro 15",
    "DEL Wireless Mouse",
    "Dell UltraSharp Monitor",
    "Deluxe Office Chair",
    "Standing Desk Converter",
    "Noise Cancelling Headphones",
    "4K Webcam",
    "Mechanical Keyboard",
    "Portable SSD 1TB",
    "Delta Power Bank",
    "Smart Desk Lamp",
    "Ergonomic Footrest",
]

CLIENTS = [
    "Delta Corp",
    "Deloitte Partners",
    "Delmont Industries",
    "Acme Retail",
    "Globex Inc",
    "Initech Solutions",
    "Umbrella Logistics",
    "Stark Manufacturing",
    "Wayne Enterprises",
    "Hooli Systems",
    "Vandelay Imports",
    "Pied Piper Tech",
]

ALL_ENTITIES = (
    [("person", v) for v in PERSONS]
    + [("location", v) for v in LOCATIONS]
    + [("product", v) for v in PRODUCTS]
    + [("client", v) for v in CLIENTS]
)
