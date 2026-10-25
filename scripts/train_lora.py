"""PHASE 2 SCAFFOLD - QLoRA fine-tuning of Qwen3-4B-Instruct-2507 on teacher-edited narratives.

    pip install unsloth trl datasets            # (or: peft transformers bitsandbytes trl datasets)
    python scripts/train_lora.py --config configs/lora.json

NOT a dependency of the app and never imported by it. Heavy libraries are imported inside main() only.
Rows with "needs_teacher_edit": true (template drafts from build_narrative_dataset.py) are rejected unless
--allow-unedited is given (smoke tests only). After training, export to GGUF and serve with Ollama, then
compare against the base model with scripts/eval_narrative.py before changing configs/slm.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
log = logging.getLogger("train_lora")


def load_rows(path: Path, allow_unedited: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unedited = [r for r in rows if r.get("needs_teacher_edit", True)]
    if unedited and not allow_unedited:
        raise SystemExit(f"{len(unedited)} of {len(rows)} rows still need teacher editing (needs_teacher_edit=true); "
                         "edit them (and set the flag to false) or pass --allow-unedited for a smoke test only")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/lora.json")
    ap.add_argument("--allow-unedited", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from cdacdm.config import resolve_path

    cfg = json.loads(resolve_path(a.config).read_text(encoding="utf-8"))
    rows = load_rows(resolve_path(cfg["dataset"]), a.allow_unedited)

    from datasets import Dataset  # heavy imports only here
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows]).train_test_split(test_size=cfg["eval_fraction"], seed=cfg["seed"])
    if cfg["backend"] == "unsloth":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(cfg["base_model"], max_seq_length=cfg["max_seq_length"],
                                                       load_in_4bit=cfg["load_in_4bit"])
        model = FastLanguageModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                                 lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
                                                 random_state=cfg["seed"])
    else:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=q,
                                                                                     device_map="auto"))
        model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                                                 target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    args = SFTConfig(output_dir=str(resolve_path(cfg["output_dir"])), learning_rate=cfg["learning_rate"],
                     num_train_epochs=cfg["num_train_epochs"], per_device_train_batch_size=cfg["per_device_train_batch_size"],
                     gradient_accumulation_steps=cfg["gradient_accumulation_steps"], warmup_ratio=cfg["warmup_ratio"],
                     seed=cfg["seed"], max_length=cfg["max_seq_length"], eval_strategy="epoch", logging_steps=10,
                     assistant_only_loss=cfg.get("train_on_responses_only", True))
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    log.info("adapter saved to %s; export to GGUF and evaluate with scripts/eval_narrative.py", args.output_dir)


if __name__ == "__main__":
    main()
