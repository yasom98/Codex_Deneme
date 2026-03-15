# Colab Workflow

## Canonical Colab Contract

- Colab full repo clone yapar
- Drive mounted asset store olur
- Local VM stage root training/evaluation icin kullanilir
- Drive'a dogrudan train etmek yok
- Overwrite yok

## Step Order

1. Drive mount et
2. GPU runtime sec
3. Repo'yu GitHub'dan `/content/Codex_Deneme` altina clone/pull et
4. `python -m pip install -r requirements.colab.txt`
5. `torch` ve runtime dependency gate'ini gec
6. `scripts/stage_colab_inputs.py` ile required seti local VM'e kopyala
7. training veya evaluation komutunu local stage root'tan calistir
8. final dosyalari Drive'da unique attempt klasorlerine sync et

## Fail-Closed Gates

Asagidaki kosullar saglanmadan run baslatma:

- `torch` import olur
- `torch.cuda.is_available()` training icin `true`
- `stable_baselines3`, `gymnasium`, `pandas`, `pyarrow` import olur
- `colab_staging_closure_report.json` icinde `overall_closure_valid=true`
- `colab_runtime_dependency_report.json` icinde `runtime_dependency_overall=true`

## Evaluation-Only Special Rule

`scripts/stage_colab_inputs.py` model artifact zip'i stage etmez.

Bu nedenle evaluation-only path'te operator:

1. run lineage inputlarini stage eder
2. model zip'i Drive'dan local VM'e manuel kopyalar
3. sonra `scripts/evaluate_policy.py` calistirir

## Sync-Back Rules

Success:

- stage report seti `runs/<RUN_ID>/colab_stage/stage_<UTCSTAMP>/`
- training artifact seti `runs/<RUN_ID>/ppo_artifact/artifact_<UTCSTAMP>/`
- evaluation report seti `runs/<RUN_ID>/evaluation/eval_<UTCSTAMP>/`

Failure:

- report/manifest dosyasi uretildiyse unique attempt klasorune sync et
- `.tmp` dosyalarini sync etme
- mevcut attempt klasorunu reuse etme
