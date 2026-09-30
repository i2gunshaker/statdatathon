# Stat.Datathon 2026

Дескриптивный анализ синтетических микроданных Казахстана за 2021–2024 годы:
236 таблиц, 72 643 893 строки. Первый этап — 30 сентября.

- [Результаты](DATA_AUDIT.md)
- [Ключевые поля](audit/DATA_DICTIONARY.md)
- [Профили таблиц](audit/table_profiles.csv) и [колонок](audit/column_profiles.csv)
- [Расчёты за 2024 год](analysis/results/descriptive_checks.json)

## Воспроизведение

Python 3.11+. Архив организаторов `synthetic_microdata_2021-2024_20260921.zip`
поместить в корень проекта.

```sh
python3 -m pip install -r requirements.txt
python3 analysis/descriptive.py
```

Результат: `analysis/results/descriptive_checks.json`. Исходные данные не изменяются.

Полный аудит дополнительно требует одноимённый TAR, macOS `textutil` и Poppler `pdftotext`:

```sh
python3 audit/audit_data.py prepare
python3 audit/profile_tables.py
python3 audit/check_relationships.py
python3 audit/build_dictionary.py
```
