# Raw traffic incidents audit

- Source file: `traffic_incidents_2026-06-01_10-44-37_1.csv`
- File size bytes: `104515392`
- SHA-256: `686e7c7b7292dedcd4540c732cb7d46edb6ef2525a94cb569ab12d95339c0694`
- Columns: `acci_id, acci_time, acci_name, acci_x, acci_y, load_timestamp`
- Row count: `720155`
- Unique `acci_id`: `716059`
- Duplicate `acci_id` rows: `4096`
- Unique `acci_name`: `233`
- Date range: `2018-08-13 08:17:21` to `2026-06-01 14:09:45`
- Invalid/blank `acci_time`: `0`
- Missing coordinate rows: `1`
- Invalid coordinate rows: `0`
- Longitude range: `0.0` to `131.26965999`
- Latitude range: `0.0` to `132.17045`
- Rows with `acci_x`/`acci_y` in expected longitude/latitude Dubai bounds: `31934`
- Rows that appear latitude/longitude swapped for Dubai bounds: `685934`
- Rows with zero coordinates: `2191`
- Other out-of-bounds coordinate rows: `95`

## Missing values by column

| Column | Missing rows |
| --- | ---: |
| `acci_id` | 0 |
| `acci_time` | 0 |
| `acci_name` | 0 |
| `acci_x` | 1 |
| `acci_y` | 0 |
| `load_timestamp` | 0 |

## Top 20 incident categories

| Rank | Count | `acci_name` |
| ---: | ---: | --- |
| 1 | 83297 | مركبه عطلانه في الشارع - بسيط |
| 2 | 58031 | الوقوف خلف المركبات (دبل بارك) - بسيط |
| 3 | 43851 | صدم عمود - بسيط |
| 4 | 36356 | حادث اصطدام بين سيارتين- بسيط |
| 5 | 35380 | تعطل مركبة على طريق عام |
| 6 | 31819 | صدم جدار - بسيط |
| 7 | 29010 | صدم حاجز - بسيط |
| 8 | 28475 | اصطدام بين مركبتين - بسيط |
| 9 | 25951 | تعطل مركبة خفيفة - بسيط |
| 10 | 22262 | مركبات مخالفة |
| 11 | 22034 | حادث صدم عمود- بسيط |
| 12 | 17905 | صدم رصيف - بسيط |
| 13 | 16915 | الصدم والهروب - بسيط |
| 14 | 15388 | وجود جسم في الشارع - بسيط |
| 15 | 15302 | حادث صدم جدار- بسيط |
| 16 | 14982 | حادث ضد مجهول- بسيط |
| 17 | 14693 | حادث صدم و هروب- بسيط |
| 18 | 12323 | صدم دراجة نارية - بسيط |
| 19 | 11242 | صدم دراجة نارية - بليغ |
| 20 | 10786 | حادث صدم حاجز- بسيط |
