# Repository Guidelines (Codex Operating Rules)

## Language
- This file is written in English for maximum model fidelity.
- Communicate with the user in Turkish (all explanations, questions, summaries in Turkish).

## Mandatory Workflow
For every non-trivial task:
1) PLAN: goal, files to change, risks, verification commands.
2) IMPLEMENT: small steps, minimal diffs, avoid sweeping refactors unless asked.
3) QC: run/outline checks (tests + lint + type) and report results.

Do not skip the PLAN step.

## Code Quality
- Write production-grade Python (3.10+).
- Type hints are mandatory. Add docstrings for public functions/classes.
- Defensive programming: validate inputs; raise clear exceptions.
- Error handling: no `except: pass`. Catch narrowly; re-raise with context.
- Logging: no `print`. Use `logging` (INFO/ERROR). Use `tqdm` for long loops.
- Determinism: set seed=42 for random/numpy/torch (if used) in one place.

## Data / Time-Series Safety (Leak-Free)
- Never use future information. If generating events/signals, ensure leak-free behavior; apply `shift(1)` when needed.
- Dataframe validation requirements:
  - required columns exist
  - time is monotonic increasing (or sort with a warning)
  - detect duplicate timestamps and apply a clear policy
  - warn if NaN ratio > 0.5% (stricter for critical columns)
- Dtypes: float32 for continuous columns, uint8 for event flags.

## Safe Writes (Health Gate + Atomic)
- Never write output files unless `health_check(...)` passes.
- Always write atomically: write to `.tmp` then rename.
- Never leave partial/corrupted files behind; clean up `.tmp` on failure.
- If paths look suspicious, stop and ask before writing (Q-GATE).

## Testing (Mandatory)
- Add at least one pytest test for every new module/feature.
- Add at least one regression test when fixing a bug.
- Prefer small, fast tests that can run frequently.

## Repo Layout (Preferred)
- src/ : library code
- scripts/ : CLI entrypoints
- tests/ : pytest tests
- configs/ : YAML configs
- README.md : run instructions

## Security / Permissions
- Avoid destructive commands and writing outside the repo.
- If a potentially risky action is required, explain and request confirmation.

## Optional: Colab Profile (when applicable)
- If running in Google Colab, provide notebook-friendly setup steps:
  - `from google.colab import drive; drive.mount("/content/drive")`
  - dependency install with `!pip install ...`
- Long runs must support checkpoints and resume.
