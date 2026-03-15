# Optional Assets Manifest

Bu dosyalar faydalidir ama ilk serious training/evaluation run'i icin zorunlu degildir.

## Optional Drive Assets

- `raw_market_data/legacy_csv/`
- `runs/<RUN_ID>/data_standardized/`
- `runs/<RUN_ID>/data_tail_refresh/`
- `runs/<RUN_ID>/training_launch/`
- `runs/<RUN_ID>/ppo_artifact_smoke/`
- `runs/<RUN_ID>/evaluation_smoke/`
- `runs/<RUN_ID>/checkpoints/`

## Notes

- raw market CSV'leri upstream lineage veya yeniden uretim icin yararlidir
- smoke output klasorleri tarihsel kanit olarak yararlidir
- checkpoint klasorleri resume stratejisi icin yararlidir
- scaler stats mevcutsa faydali bir reference olabilir, ama ilk serious run icin zorunlu degildir
