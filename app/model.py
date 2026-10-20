"""Load the fine-tuned model once, then label a whole batch of issues in one generate() call."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from .prompt import CATEGORIES, build_prompt

hf_logging.disable_progress_bar()             # keep the logs one JSON line per event


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse(text: str) -> str:
    # the same rule as the evaluation: the first category the answer mentions wins
    text = text.lower().replace("*", "")
    found = [(text.find(c), c) for c in CATEGORIES if c in text]
    return min(found)[1] if found else "unknown"


class IssueClassifier:
    def __init__(self, weights_dir: str, device: str = "auto", dtype: str = "bfloat16"):
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"      # every prompt must end right where the answer starts
        self.model = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, issues: list[tuple[str, str]]) -> list[tuple[str, float]]:
        prompts = [build_prompt(self.tokenizer, title, body) for title, body in issues]
        batch = self.tokenizer(prompts, add_special_tokens=False, padding=True,
                               return_tensors="pt").to(self.device)
        out = self.model.generate(**batch, max_new_tokens=4, do_sample=False, output_scores=True,
                                  return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id)
        answers = out.sequences[:, batch["input_ids"].shape[1]:]
        first_token_probs = out.scores[0].float().softmax(dim=-1)
        results = []
        for i, tokens in enumerate(answers):
            label = parse(self.tokenizer.decode(tokens, skip_special_tokens=True))
            results.append((label, first_token_probs[i, tokens[0]].item()))
        return results
