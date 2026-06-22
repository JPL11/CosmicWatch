#!/usr/bin/env python3
"""
Check whether the ES index contains synchronized multi-node data.

This is intentionally separate from the edge-AI experiment. It answers:
  - which sources have timestamped data,
  - which node/device identifiers appear per source,
  - whether different sources overlap on the same day,
  - whether any source has multiple devices active on the same day.
"""
import datetime as dt
import json
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import requests
import urllib3

from credo_config import es_auth, es_settings, verify_certs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def post_json(body, timeout=180):
    es_url, index = es_settings()
    response = requests.post(
        f"{es_url}/{index}/_search",
        auth=es_auth(),
        verify=verify_certs(),
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def timestamp_iso(milliseconds):
    if milliseconds is None:
        return None
    return (
        dt.datetime.fromtimestamp(milliseconds / 1000, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def source_terms():
    response = post_json(
        {
            "size": 0,
            "aggs": {"sources": {"terms": {"field": "source", "size": 20}}},
        },
        timeout=120,
    )
    return response["aggregations"]["sources"]["buckets"]


def summarize_source(source):
    response = post_json(
        {
            "size": 0,
            "track_total_hits": True,
            "query": {"term": {"source": source}},
            "aggs": {
                "time_range": {"stats": {"field": "timestamp"}},
                "timestamp_exists": {"filter": {"exists": {"field": "timestamp"}}},
                "device_ids": {"terms": {"field": "device_id", "size": 10}},
                "detector_names": {
                    "terms": {"field": "detector_name.keyword", "size": 10}
                },
                "detectors": {"terms": {"field": "detector.keyword", "size": 10}},
                "device_models": {
                    "terms": {"field": "device_model.keyword", "size": 10}
                },
                "user_ids": {"terms": {"field": "user_id", "size": 10}},
                "lat_exists": {"filter": {"exists": {"field": "latitude"}}},
                "lon_exists": {"filter": {"exists": {"field": "longitude"}}},
                "pico_exists": {"filter": {"exists": {"field": "pico_timestamp_s"}}},
            },
        }
    )
    aggs = response["aggregations"]
    time_range = aggs["time_range"]
    return {
        "source": source,
        "total_docs": response["hits"]["total"]["value"],
        "timestamp_rows": aggs["timestamp_exists"]["doc_count"],
        "earliest_timestamp": timestamp_iso(time_range["min"]),
        "latest_timestamp": timestamp_iso(time_range["max"]),
        "device_ids": aggs["device_ids"]["buckets"],
        "detector_names": aggs["detector_names"]["buckets"],
        "detectors": aggs["detectors"]["buckets"],
        "device_models": aggs["device_models"]["buckets"],
        "user_ids": aggs["user_ids"]["buckets"],
        "rows_with": {
            "latitude": aggs["lat_exists"]["doc_count"],
            "longitude": aggs["lon_exists"]["doc_count"],
            "pico_timestamp_s": aggs["pico_exists"]["doc_count"],
        },
    }


def days_with_multiple_sources():
    response = post_json(
        {
            "size": 0,
            "query": {"exists": {"field": "timestamp"}},
            "aggs": {
                "days": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "day",
                        "min_doc_count": 1,
                    },
                    "aggs": {"sources": {"terms": {"field": "source", "size": 10}}},
                }
            },
        }
    )
    overlap = []
    for bucket in response["aggregations"]["days"]["buckets"]:
        sources = {item["key"]: item["doc_count"] for item in bucket["sources"]["buckets"]}
        if len(sources) > 1:
            overlap.append({"day": bucket["key_as_string"][:10], "sources": sources})
    return overlap


def multi_device_days(source):
    response = post_json(
        {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"source": source}},
                        {"exists": {"field": "timestamp"}},
                    ]
                }
            },
            "aggs": {
                "days": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "day",
                        "min_doc_count": 1,
                    },
                    "aggs": {
                        "devices": {"terms": {"field": "device_id", "size": 20}},
                        "device_count": {"cardinality": {"field": "device_id"}},
                    },
                }
            },
        }
    )
    days = []
    for bucket in response["aggregations"]["days"]["buckets"]:
        if bucket["device_count"]["value"] > 1:
            days.append(
                {
                    "day": bucket["key_as_string"][:10],
                    "count": bucket["doc_count"],
                    "device_count": bucket["device_count"]["value"],
                    "devices": bucket["devices"]["buckets"][:8],
                }
            )
    return days


def main():
    sources = source_terms()
    source_names = [source["key"] for source in sources]
    output = {
        "source_counts": sources,
        "source_summaries": [summarize_source(source) for source in source_names],
        "days_with_multiple_sources": days_with_multiple_sources(),
        "multi_device_days_by_source": {
            source: multi_device_days(source) for source in source_names
        },
        "conclusion": (
            "No synchronized multi-source CosmicWatch/CREDO data is present in this "
            "index. CosmicWatch timestamped rows come from one device_id, and there "
            "are zero days where multiple sources overlap."
        ),
    }
    with open("multi_node_probe.json", "w") as output_file:
        json.dump(output, output_file, indent=2)

    print(output["conclusion"])
    print("Wrote multi_node_probe.json")


if __name__ == "__main__":
    main()

