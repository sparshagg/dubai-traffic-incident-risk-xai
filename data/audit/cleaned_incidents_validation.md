# Cleaned incident dataset validation

- Source file: `traffic_incidents_2026-06-01_10-44-37_1.csv`
- Mapping file: `data/mappings/acci_name_translation_map.csv`
- Output file: `data/processed/traffic_incidents_cleaned.csv`
- Cleaned row count: `720155`
- Mapping rows: `233`
- Normalized mapping keys used for joining: `226`
- Duplicate normalized mapping keys collapsed for joining: `7`
- Rows missing mapping: `0`
- Rows mapped to `needs_review` categories: `0`
- Rows excluded from category-level EDA: `426`
- Rows with blank translation outside `needs_review`: `0`
- Rows with unknown or unspecified severity: `74325`
- Unknown-severity rows with nonzero weight or wrong flag: `0`
- Known-severity rows with wrong known-severity flag: `0`
