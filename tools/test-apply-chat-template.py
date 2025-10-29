import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

data_path = Path("/Users/yentinglin/projects/slime/tau_airline_hermes_format_with_sp_sft.jsonl")
with data_path.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
output_path = data_path.with_suffix(".chat_template.md")
with output_path.open("w", encoding="utf-8") as f:
    for idx, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools"),
            tokenize=False,
        )
        f.write(f"## Prompt {idx}\n\n```\n{prompt}\n```\n\n")
