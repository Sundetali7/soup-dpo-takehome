# Soup take-home: DPO + layer streaming on a free T4

**Verdict: DON'T SHIP** — see [REPORT.md](REPORT.md).

- `notebooks/final_clean.ipynb` — final run, top to bottom, with a changelog vs the draft
- `notebooks/draft_plan_A.ipynb` — first attempt (soup 0.72.4), failures kept as evidence
- `scripts/verify_training.py` — Part 2 check (4 criteria + NULL/RANDOM controls)
- `logs/` — raw timestamped logs, raw `nvidia-smi` CSV, verification and ship JSON
- Run B adapter weights (35 MB) omitted; reproduce with `configs/run_B.yaml`

Environment: Colab T4, Python 3.12 venv, soup-cli 0.75.1, Qwen2.5-0.5B-Instruct, `r1char9/ru-detox-dpo`.
