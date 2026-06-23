# Data Dictionary --- `credo-detections` export

Companion to `credo_export.csv(.gz)` (full index) and `credo_useful.csv(.gz)` (usable events + useful images). 73 columns = 72 mapped ES fields + the derived `partition` column.

**Reading the export:** the first column `partition` separates the two CosmicWatch schema versions and the sources. Nested fields (`location`, `metadata`) and lists are JSON-encoded in their cells; booleans are `true`/`false`; image fields hold base64 PNG text.

## Key caveats (read first)

- **Both** CosmicWatch partitions are usable (~3.36M events): `cosmicwatch_parsed` (582k) keys on `timestamp`+`coincident`; `cosmicwatch_raw_axlab` (2.77M) keys on `wall_time`+`coincidence_flag`.
- Canonical fields: time = `timestamp` else `wall_time`; label = `coincident` else `coincidence_flag` (see `credo_loader.py`).
- `timestamp_ms` is **1-second** resolution; `wall_time`/`pico_timestamp_s` are microsecond.
- `credo.science` is **degenerate**: `latitude`/`longitude` = 0,0, `energy` = 0, `particle_type` constant.
- `visible` is constant `False`; `llm_interpretation` is mostly error strings --- neither is a usable label.
- `adc_value` is 0--4095 (12-bit, saturates at 4095); ~5.8 keV/ADC by approximate MIP calibration.

## Export / identity

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `partition` | derived | label | partition (derived) | DERIVED export column: separates the two CosmicWatch schemas and the sources. | Values: cosmicwatch_parsed / cosmicwatch_raw_axlab / legacy / credo.science / phone-camera / credo-science. |
| `source` | keyword |  | — | Ingest source / dataset the record came from. |  |
| `device_id` | keyword |  | cosmicwatch-v3x, credo-science, credo.science, legacy, phone-camera | Device/detector identifier. | cosmicwatch-v3x has a single id: cosmicwatch-001. |
| `device_model` | text |  | — | Phone/device model string. | image sources; inferred. |
| `detector_name` | text |  | cosmicwatch-v3x | Human-readable detector name. | parsed CosmicWatch partition. |
| `detector` | text |  | — | Detector label in the raw partition. | raw AxLab partition uses 'AxLab'. |
| `doc_id` | long |  | — | Source document id (credo.science). | inferred. |
| `id` | long |  | — | Record id in the raw partition / image sources. | inferred. |
| `team_id` | keyword |  | credo-science, legacy | CREDO team identifier (image sources). |  |
| `user_id` | keyword |  | credo-science, legacy | CREDO user/participant identifier (image sources). |  |
| `provider` | keyword |  | credo-science, legacy, phone-camera | Location provider (e.g. gps/network) for image sources. |  |
| `event` | long |  | cosmicwatch-v3x | Event counter (parsed CosmicWatch). |  |
| `event_num` | long |  | — | Event counter (raw partition). |  |
| `es_synced` | long |  | — | Elasticsearch sync flag/marker. | inferred. |

## Timing

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `timestamp` | date | ISO-8601 UTC | cosmicwatch-v3x, credo-science, credo.science, legacy, phone-camera | Wall-clock event time. | Only on the 582k PARSED CosmicWatch + image sources; 1-second resolution. |
| `timestamp_ms` | long | ms since epoch | cosmicwatch-v3x | Wall-clock time in epoch milliseconds. | Quantized to 1 second (ends in 000); use pico_timestamp_s for fine timing. |
| `timestamp_s` | float | s (boot-relative) | — | Seconds since device power-on (raw partition). | Boot-relative; use `wall_time` for ABSOLUTE time on the raw partition. |
| `pico_timestamp_s` | float | s (boot-relative) | cosmicwatch-v3x | High-resolution (microsecond) device clock. | Boot-relative; DIFFERENCES are valid inter-arrival times (Poisson check). |
| `comp_date` | text |  | — | On-device computed date string. | parsed CosmicWatch. |
| `comp_time` | text |  | — | On-device computed time string. | parsed CosmicWatch. |
| `time_received` | date | date | — | Server-side ingestion time. | image sources. |
| `wall_time` | float | s (epoch) | — | REAL wall-clock time: Unix epoch seconds, microsecond precision. | The absolute-time ANCHOR for the raw AxLab partition (which has no `timestamp`); also on phone-camera. Convert to UTC. |
| `ml_timestamp` | date | date | — | Time an ML/LLM inference was attached. |  |

## CosmicWatch detector (physics)

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `adc_value` | long | ADC counts | cosmicwatch-v3x | Pulse height; PROPORTIONAL TO ENERGY DEPOSITED. | 0-4095 (12-bit); saturates at 4095; ~5.8 keV/ADC (approx MIP calibration). |
| `sipm_mv` | float | mV | cosmicwatch-v3x | SiPM pulse amplitude in millivolts. |  |
| `coincident` | boolean | bool | cosmicwatch-v3x | Both internal panels fired (intra-unit coincidence). | Weak hardware label; ~12% in clean window; parsed partition. |
| `coincidence_flag` | long | 0/1 | cosmicwatch-v3x | Coincidence as an integer (raw partition). | Same meaning as coincident. |
| `temperature_c` | float | deg C | cosmicwatch-v3x | On-board temperature. | Has corrupt outliers (max ~2.5e6); clip to [-50,80]. |
| `pressure_pa` | float | Pa | cosmicwatch-v3x | On-board atmospheric pressure. | Has 0-Pa floor + corrupt highs; clip to [80000,110000]. |
| `deadtime_s` | float | s (cumulative) | cosmicwatch-v3x | Running total of detector dead-time. | Monotonic; per-event = difference; ~0.07% of live time. |
| `cosmicwatch_adc` | long | ADC counts | — | Alias of adc_value in some records. | inferred duplicate. |
| `cosmicwatch_sipm_mv` | float | mV | — | Alias of sipm_mv. | inferred duplicate. |
| `cosmicwatch_coincident` | boolean | bool | — | Alias of coincident. | inferred duplicate. |
| `cosmicwatch_temperature` | float | deg C | — | Alias of temperature_c. | inferred duplicate. |
| `cosmicwatch_pressure` | float | Pa | — | Alias of pressure_pa. | inferred duplicate. |
| `cosmicwatch_deadtime` | float | s | — | Alias of deadtime_s. | inferred duplicate. |

## Motion sensors

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `accel_x_g` | float | g | cosmicwatch-v3x | Accelerometer X in gravitational units. | parsed CosmicWatch (~17% coverage). |
| `accel_y_g` | float | g | cosmicwatch-v3x | Accelerometer Y in gravitational units. |  |
| `accel_z_g` | float | g | cosmicwatch-v3x | Accelerometer Z in g; ~-1 g at rest = gravity vector (tilt). |  |
| `accel_x` | float | raw | — | Raw accelerometer X (raw partition). |  |
| `accel_y` | float | raw | — | Raw accelerometer Y. |  |
| `accel_z` | float | raw | — | Raw accelerometer Z. |  |
| `gyro_x_degs` | float | deg/s | cosmicwatch-v3x | Gyroscope X angular rate. |  |
| `gyro_y_degs` | float | deg/s | — | Gyroscope Y angular rate. |  |
| `gyro_z_degs` | float | deg/s | — | Gyroscope Z angular rate. |  |
| `gyro_x` | float | raw | — | Raw gyroscope X (raw partition). |  |
| `gyro_y` | float | raw | — | Raw gyroscope Y. |  |
| `gyro_z` | float | raw | — | Raw gyroscope Z. |  |

## Geo

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `latitude` | float | deg | credo.science, phone-camera | Latitude. | credo.science values are all 0.0 (DEGENERATE); phone-camera real (~LA). |
| `longitude` | float | deg | credo.science, phone-camera | Longitude. | credo.science all 0.0 (DEGENERATE); phone-camera real. |
| `altitude` | float | m | credo-science, credo.science, legacy, phone-camera | Altitude. | image sources / credo.science. |
| `location` | geo_point | geo_point {lat,lon} | credo-science, legacy | Geo-point coordinate object. | legacy & credo-science: real Poland coords. |
| `accuracy` | float | m | credo-science, legacy | Location accuracy radius. | image sources. |
| `gps_accuracy` | float | m | — | GPS accuracy radius. | inferred. |

## Image / CV

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `frame_content` | binary | base64 PNG | credo-science, legacy | Image hit-crop (~20x20 RGBA PNG), base64-encoded. | legacy (69k) & credo-science (77); decodable PNG. |
| `image_b64` | text | base64 PNG | phone-camera | Image, base64-encoded. | phone-camera: present on 1,569 of 3,095 rows. |
| `frame_width` | long | px | — | Full source frame width. |  |
| `frame_height` | long | px | — | Full source frame height. |  |
| `width` | long | px | credo-science, legacy | Full image width (e.g. 960). | image sources; crop itself is ~20x20. |
| `height` | long | px | credo-science, legacy | Full image height (e.g. 720). | image sources. |
| `brightness` | float | 0-255 | phone-camera | Hit/region brightness. | phone-camera. |
| `hit_x` | long | px | — | Hit x-coordinate within the frame. |  |
| `hit_y` | long | px | — | Hit y-coordinate within the frame. |  |
| `x` | long | px | — | Hit x-coordinate (credo-science). |  |
| `y` | long | px | — | Hit y-coordinate (credo-science). |  |
| `visible` | boolean | bool | credo-science, legacy | Whether the hit is flagged visible. | legacy: constant False -> NOT a usable label. |
| `cluster_size` | long | px | phone-camera | Size of the detected pixel cluster. | phone-camera. |

## Particle / ML

| field | type | unit | populated in | description | caveats |
|---|---|---|---|---|---|
| `energy` | long | eV (nominal) | credo.science | Reconstructed particle energy. | credo.science only and ALL 0 -> DEGENERATE/unusable. |
| `particle_type` | text |  | credo.science | Particle class label. | credo.science: constant 'cosmic_ray' (single-valued). |
| `ml_prediction` | long | class id | — | ML model predicted class. | inferred. |
| `ml_probability` | float | 0-1 | — | ML model confidence. | inferred. |
| `llm_interpretation` | text | text | cosmicwatch-v3x | LLM-generated interpretation string. | Mostly LLM-ERROR strings (e.g. 'LLM unavailable'); not a usable label. |
| `metadata` | object | json object | — | Free-form nested metadata. | JSON-encoded in CSV cells. |

