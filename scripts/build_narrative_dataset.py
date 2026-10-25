"""PHASE 2 SCAFFOLD - build a (facts -> narrative) JSONL for optional LoRA fine-tuning.

    python scripts/build_narrative_dataset.py --artifact-dir artifacts --classes 20 --class-size 30 --students 100

Every target ("assistant") text is a DRAFT produced by the deterministic template (fallback_narrative). It is
NOT gold data: each row carries "needs_teacher_edit": true and must be rewritten/approved by a teacher before
training (scripts/train_lora.py refuses unedited rows by default). Facts are built exactly like the app does
(pseudonymised ids, no raw responses). Not imported by the app; nothing here is run by the pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("build_narrative_dataset")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact-dir", default="artifacts")
    ap.add_argument("--out", default="artifacts/reports/narrative_dataset_DRAFT.jsonl")
    ap.add_argument("--classes", type=int, default=20, help="number of sampled classes (teacher reports)")
    ap.add_argument("--class-size", type=int, default=30)
    ap.add_argument("--students", type=int, default=100, help="number of student reports")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import numpy as np
    import pandas as pd

    from cdacdm.config import resolve_path
    from cdacdm.narrative import build_prompt, fallback_narrative, load_narrative_config
    from cdacdm.service import CDMService

    art = resolve_path(a.artifact_dir)
    svc = CDMService.load(art)
    cfg = load_narrative_config()
    demo_p = next((p for p in [art / "demo/test_split_responses.csv", art / "demo/eedi_test_split_responses.csv"] if p.exists()), None)
    if demo_p is None:
        raise SystemExit(f"no demo responses under {art}/demo")
    demo = pd.read_csv(demo_p, dtype={"student_id": str})
    ids = demo["student_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(a.seed)
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        def write(facts):
            nonlocal n
            msgs = build_prompt(facts, facts.audience, examples=[])
            fh.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": fallback_narrative(facts)}],
                                 "audience": facts.audience, "prompt_version": cfg.prompt_version,
                                 "label_source": "template_draft", "needs_teacher_edit": True}, ensure_ascii=False) + "\n")
            n += 1

        for _ in range(a.classes):
            cls = rng.choice(ids, size=min(a.class_size, len(ids)), replace=False)
            res = svc.diagnose(demo[demo["student_id"].isin(cls)])
            write(svc.narrative_facts(res, "teacher", config=cfg))
        pick = rng.choice(ids, size=min(a.students, len(ids)), replace=False)
        res = svc.diagnose(demo[demo["student_id"].isin(pick)])
        for sid in pick:
            write(svc.narrative_facts(res, "student", str(sid), cfg))
    log.info("wrote %s DRAFT rows to %s - teachers must edit the assistant texts before training", n, out)


if __name__ == "__main__":
    main()
