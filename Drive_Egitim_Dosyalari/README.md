# Drive Egitim Dosyalari

Bu klasor gercek agir data/artifact deposu degildir.

Amac:

- Google Drive'a neyin yuklenecegini canonical olarak tanimlamak
- repo ile Drive arasindaki gorev ayrimini netlestirmek
- Colab operatorunun eksik asset surprizi yasamasini engellemek

Bu klasorde:

- gercek buyuk dataset yok
- fake artifact yok
- fake report yok

Canonical karar:

- GitHub = code / config / docs / tests / spec
- Drive = agir training/evaluation assetleri
- Colab = execution environment

Canonical Drive root:

- `MyDrive/Codex_Deneme_Assets`

Baslica referans dosyalari:

- `DRIVE_FOLDER_TREE.md`
- `REQUIRED_ASSETS_MANIFEST.md`
- `OPTIONAL_ASSETS_MANIFEST.md`
- `COLAB_WORKFLOW.md`

Placeholder klasorleri sadece paketleme/taksonomi contract'i icindir. Gercek agir dosyalar bu repo'ya eklenmemelidir.
