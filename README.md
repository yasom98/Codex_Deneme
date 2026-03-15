# Codex_Deneme

WSL + Codex ile geliştiriliyor.

## Canonical GitHub + Drive + Colab Workflow

Bu repo icin canonical isletim modeli sudur:

- GitHub = code / config / docs / tests / spec source of truth
- Google Drive = agir training/evaluation girdileri ve run artifact source of truth
- Colab = disposable execution environment
- Colab her zaman full repo clone yapar
- Agir dosyalar GitHub'a tekrar tekrar tasinmaz
- `tgz` ancak opsiyonel snapshot/arsiv yoludur; gunluk workflow degildir

Operator-facing canonical dokumanlar:

- `docs/github_drive_colab_workflow.md`
- `docs/repo_audit_report.md`
- `Drive_Egitim_Dosyalari/README.md`
- `docs/operations/colab_first_real_ppo_training.md`

Root seviyesindeki asagidaki agir CSV'ler canonical repo asset'i degildir:

- `BTC_USDT_15m_price_data.csv`
- `BTC_USDT_5m_price_data.csv`

Bu dosyalar bu passtta silinmeyecek, ancak `legacy tracked migration debt` olarak degerlendirilmelidir. Yeni agir data/artifact dosyalari GitHub'a eklenmemelidir.

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
