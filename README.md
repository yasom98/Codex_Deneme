# Codex_Deneme

WSL + Codex ile geliştiriliyor.

## 4.7 Training Config Examples

- `configs/training_config.launch_smoke.example.json`: bounded launch validation icin kucuk smoke baslangic config'i.
- `configs/training_config.baseline_train.example.json`: PPO icin ilk egitim baslangic noktasi, tuned sonuc degil.

Bu iki config 4.7 kapsaminda yalnizca starter/validation amaclidir.
Gercek hyperparameter optimization kapsamli olarak bilerek Milestone 4.9'a ertelenmistir.

## 4.8 Closure Prep Artifact Production

- `configs/training_config.artifact_production.example.json`: explicit path ile tuketilecek, tek kanonik `PPO` artefakti uretmek icin dar kapsamli production config ornegi.
- `scripts/produce_canonical_ppo_artifact.py`: explicit upstream path'leri dogrular, tek `canonical_ppo_model.zip` uretir, load-back validation yapar ve `artifact_production_manifest.json` ile `artifact_production_report.json` yazar.

## 4.9 Constrained PPO Search

- `configs/ppo_search.study.example.json`: explicit upstream ref'ler, dar ilk-dalga PPO search space'i, objective/guardrail/pruning/promotion kurallari ve output root iceren 4.9 study contract ornegi.
- `scripts/run_ppo_search_study.py`: `StudySpec -> TrialSpec -> Train -> Evaluate -> Score -> Guardrail -> Trial Reports -> Study Summary` zincirini explicit path ve fail-closed davranisla yurutur.
- `src/rl/ppo_search_orchestrator.py`: machine-readable study/trial artifact seti, validation-centered tek scalar objective, conservative pruning ve promotion-readiness durum semantigini uygular.
