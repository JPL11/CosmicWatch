#!/usr/bin/env python3
"""Stream canonical CosmicWatch rows from the local useful CSV export."""
import csv
import datetime as dt


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_cosmicwatch(path="credo_useful.csv"):
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("source") != "cosmicwatch-v3x":
                continue
            timestamp = _number(row.get("timestamp"))
            wall_time = _number(row.get("wall_time"))
            epoch = timestamp / 1000.0 if timestamp is not None else wall_time
            coincidence = row.get("coincident")
            if coincidence == "":
                coincidence = row.get("coincidence_flag")
            coincident = str(coincidence).strip().lower() in {"1", "true"}
            yield {
                "time_epoch_s": epoch,
                "partition": "parsed" if timestamp is not None else "raw",
                "adc_value": _number(row.get("adc_value")),
                "sipm_mv": _number(row.get("sipm_mv")),
                "temperature_c": _number(row.get("temperature_c")),
                "pressure_pa": _number(row.get("pressure_pa")),
                "coincident": coincident,
            }


def utc_day(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date().isoformat()
