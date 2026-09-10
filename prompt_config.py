import re

INSTRUCTION = "Estimate the fair market retail value in USD based on the product title, condition, category, and technical features."

OUTPUT_FORMAT = "int"

SYSTEM_PROMPT = (
    "You are a precise electronics fair-market valuation model. "
    "Estimate the normal fair retail value, not the current deal price. "
    "Use product identity, generation, brand, condition, category, and specifications. "
    "Do not anchor on a promotional/deal price. "
    "Output ONLY the raw integer estimated USD value without dollar signs or decimals."
)

def format_input(category, condition, title, features):
    if isinstance(features, (list, tuple)):
        feat_str = ", ".join(str(x).strip() for x in features if str(x).strip())
    else:
        feat_str = str(features).strip()
    feat_str = feat_str or "Standard specifications"
    return f"Product: [Category: {str(category).strip()}] [Condition: {str(condition).strip()}] {str(title).strip()} | Features: {feat_str}"

def build_user_turn(input_text):
    return f"{INSTRUCTION}\n{input_text.strip()}"

def build_messages(input_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(input_text)},
    ]

def format_target(raw_price):
    if isinstance(raw_price, str):
        clean = re.sub(r"[^\d.]", "", raw_price.replace(",", ""))
    else:
        clean = str(raw_price)
    try:
        return str(int(round(float(clean))))
    except (ValueError, TypeError):
        return "0"

def parse_price(text):
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d{1,2})?", str(text).replace(",", ""))
    return float(match.group(0)) if match else None
