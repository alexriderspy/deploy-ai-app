"""Fold the LoRA adapter into the base weights, so serving needs one folder and no peft."""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
base_id = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]

base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter)
model = model.merge_and_unload()            # W + (alpha / r) * B @ A, written into W itself
model.save_pretrained(out)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)

size_gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
print(f"base model: {base_id}")
print(f"merged {adapter} -> {out}  ({size_gb:.2f} GB of weights)")
