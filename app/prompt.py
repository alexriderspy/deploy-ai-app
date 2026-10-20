"""Copied unchanged from the fine-tuning project: the model only works with the exact prompt it was trained on."""
CATEGORIES = ["bug", "feature request", "usage", "installation", "performance", "new model", "documentation"]

SYSTEM = ("You triage GitHub issues for vLLM, the open-source LLM inference engine. "
          "Read the issue and reply with exactly one category: " + ", ".join(CATEGORIES) + ".")

def build_prompt(tokenizer, title: str, body: str, max_body_tokens: int = 128) -> str:
    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"][:max_body_tokens]
    body = tokenizer.decode(body_ids)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Title: {title}\n\nBody: {body}"}]
    # a fixed date, otherwise Llama's template stamps today's date into every prompt
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         date_string="20 Oct 2026")
