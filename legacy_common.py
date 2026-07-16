#!/usr/bin/env python3
"""Shared, deduplicated access to legacy CREDO records in the local CSV export."""
import base64
import csv
import hashlib
import io
import json

import numpy as np
from PIL import Image


def iter_legacy(path="credo_useful.csv", include_image=True):
    seen = set()
    duplicate_count = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("source") != "legacy":
                continue
            image = row.get("frame_content", "")
            digest = hashlib.sha1(image.encode("ascii")).digest() if image else b""
            key = (row.get("device_id", ""), row.get("timestamp", ""), digest)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            location = None
            if row.get("location"):
                try:
                    value = json.loads(row["location"])
                    location = (float(value["lat"]), float(value["lon"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            yield {
                "device_id": row.get("device_id", ""),
                "timestamp_ms": int(row["timestamp"]),
                "location": location,
                "visible": row.get("visible"),
                "frame_content": image if include_image else None,
                "duplicates_before": duplicate_count,
            }


def decode_gray(encoded, size=20):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
        if image.size != (size, size):
            image = image.resize((size, size))
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return None


def load_images(path="credo_useful.csv"):
    images, devices, times, locations, visible = [], [], [], [], []
    duplicate_count = 0
    for row in iter_legacy(path, include_image=True):
        duplicate_count = row["duplicates_before"]
        image = decode_gray(row["frame_content"])
        if image is None:
            continue
        images.append(image)
        devices.append(row["device_id"])
        times.append(row["timestamp_ms"])
        locations.append(row["location"])
        visible.append(row["visible"])
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "devices": np.asarray(devices),
        "times": np.asarray(times, dtype=np.int64),
        "locations": locations,
        "visible": visible,
        "duplicates_removed": duplicate_count,
    }
