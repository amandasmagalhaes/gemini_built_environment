
import os
import argparse
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from pydantic import BaseModel, Field
from PIL import Image
from haversine import haversine, Unit
from google import genai
from tqdm import tqdm

import json
from google.genai import types

from streetview import (
    search_panoramas,
    get_panorama,
    get_panorama_meta,
    crop_bottom_and_right_black_border,
)


# =========================
# DEFAULT CONFIG
# =========================

DEFAULT_INPUT_CSV = "items_coordinates_belo_horizonte 5more.csv"
DEFAULT_OUTPUT_CSV = "streetview_predictions.csv"
DEFAULT_IMAGE_DIR = "streetview_images"
DEFAULT_PROMPT_FILE = "gemini_prompt.txt"

GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    "AIzaSyAg2gh8tGGsYt3oNP0TegsOqyPC3oxq3qk",
)
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyC8q5N20wGYnxay8bQH7j9CDrDF3jjnAmM",
)

MAX_DISTANCE_METERS = 10.0
MIN_YEAR = 2015
PANORAMA_ZOOM = 4
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.0-flash")


# =========================
# GEMINI OUTPUT SCHEMA
# =========================

Binary = int


class BuiltEnvPrediction(BaseModel):
    Sidewalks: Binary = Field(ge=0, le=1)
    Curbs: Binary = Field(ge=0, le=1)
    Crosswalks: Binary = Field(ge=0, le=1)
    Speed_Bumps: Binary = Field(ge=0, le=1)
    Pothole: Binary = Field(ge=0, le=1)
    Streetlights: Binary = Field(ge=0, le=1)
    Parked_vehicles: Binary = Field(ge=0, le=1)
    Traffic_light: Binary = Field(ge=0, le=1)
    Stop_Sign: Binary = Field(ge=0, le=1)
    Yield_sign: Binary = Field(ge=0, le=1)
    Sidewalk_obstruction: Binary = Field(ge=0, le=1)
    Bike_lane: Binary = Field(ge=0, le=1)
    Pedestrian_signal: Binary = Field(ge=0, le=1)
    Trees: Binary = Field(ge=0, le=1)
    Kiosks: Binary = Field(ge=0, le=1)
    Median: Binary = Field(ge=0, le=1)
    Bollards: Binary = Field(ge=0, le=1)
    Median_barrier: Binary = Field(ge=0, le=1)
    Lane_markings: Binary = Field(ge=0, le=1)
    Traffic_signs: Binary = Field(ge=0, le=1)
    Crossing_sign: Binary = Field(ge=0, le=1)
    School_zone: Binary = Field(ge=0, le=1)
    Bus_lane: Binary = Field(ge=0, le=1)
    Parking_lane: Binary = Field(ge=0, le=1)
    BRT_Station: Binary = Field(ge=0, le=1)
    Bus_stop: Binary = Field(ge=0, le=1)
    Roundabout: Binary = Field(ge=0, le=1)


PREDICTION_FIELDS = list(BuiltEnvPrediction.model_fields.keys())

class DetectionBox(BaseModel):
    box_2d: list[int]
    label: str


DETECTION_LABEL_MAP = {
    "Sidewalks": ["sidewalk", "pavement"],
    "Curbs": ["curb"],
    "Crosswalks": ["crosswalk", "zebra crossing"],
    "Speed_Bumps": ["speed bump"],
    "Pothole": ["pothole"],
    "Streetlights": ["streetlight", "lamp post"],
    "Parked_vehicles": ["parked car", "parked vehicle", "car", "vehicle"],
    "Traffic_light": ["traffic light"],
    "Stop_Sign": ["stop sign"],
    "Yield_sign": ["yield sign"],
    "Sidewalk_obstruction": ["sidewalk obstruction", "obstruction", "blocked sidewalk"],
    "Bike_lane": ["bike lane", "cycle lane"],
    "Pedestrian_signal": ["pedestrian signal", "walk signal"],
    "Trees": ["tree"],
    "Kiosks": ["kiosk", "newsstand"],
    "Median": ["median", "center median"],
    "Bollards": ["bollard"],
    "Median_barrier": ["median barrier", "barrier", "guardrail"],
    "Lane_markings": ["lane marking", "road marking"],
    "Traffic_signs": ["traffic sign", "road sign"],
    "Crossing_sign": ["pedestrian crossing sign", "crossing sign"],
    "School_zone": ["school zone sign", "school zone"],
    "Bus_lane": ["bus lane"],
    "Parking_lane": ["parking lane", "parking strip"],
    "BRT_Station": ["brt station", "bus rapid transit station"],
    "Bus_stop": ["bus stop", "bus stop sign", "bus shelter"],
    "Roundabout": ["roundabout"],
}

def parse_json_response(text: str):
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return json.loads(text)


# =========================
# HELPERS
# =========================

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_detection_prompt() -> str:
    categories = "\n".join(
        f'- "{feature}": {", ".join(labels)}'
        for feature, labels in DETECTION_LABEL_MAP.items()
    )

    return f"""
You are performing object detection on a Google Street View panoramic image.

Important constraints:
1. This is a panoramic image and may contain severe distortion.
2. Only detect objects that belong to the immediate vicinity of the camera location, roughly within 50 to 100 meters.
3. Ignore objects far away, down the block, or belonging to another intersection.
4. Return only a JSON array.
5. For each detected object, return:
   - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0..1000
   - "label": one of the allowed labels below
6. Do not include explanations or markdown.

Study variables and allowed labels:
{categories}

Return an empty JSON array [] if nothing relevant is present.
""".strip()


def parse_year_month(date_str: Optional[str]) -> tuple[int, int]:
    if not date_str or pd.isna(date_str):
        return (0, 0)
    parts = str(date_str).split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        return (year, month)
    except Exception:
        return (0, 0)


def is_recent_enough(date_str: Optional[str], min_year: int = MIN_YEAR) -> bool:
    year, _ = parse_year_month(date_str)
    return year >= min_year


def distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def make_point_id(row: pd.Series, idx: int) -> str:
    """
    Use existing id if present; otherwise derive a stable ID from coordinates.
    """
    if "id" in row and pd.notna(row["id"]):
        return str(row["id"])
    lat = float(row["lat"])
    lon = float(row["lon"])
    return f"lat{lat:.6f}_lon{lon:.6f}"


def empty_record(point_id: str, lat: float, lon: float) -> Dict[str, Any]:
    rec = {
        "id": point_id,
        "lat": lat,
        "lon": lon,
        "status": "pending",
        "reason": None,
        "pano_id": None,
        "pano_lat": None,
        "pano_lon": None,
        "pano_distance_m": None,
        "pano_date": None,
        "image_path": None,
        "analysis_mode_used": None,
        "detection_boxes_json": None,
        "detection_boxes_path": None,
    }
    for field in PREDICTION_FIELDS:
        rec[field] = None
    return rec


def normalized_box_to_pixels(box_2d: list[int], width: int, height: int) -> dict:
    ymin, xmin, ymax, xmax = box_2d

    abs_ymin = int(ymin / 1000 * height)
    abs_xmin = int(xmin / 1000 * width)
    abs_ymax = int(ymax / 1000 * height)
    abs_xmax = int(xmax / 1000 * width)

    return {
        "xmin": abs_xmin,
        "ymin": abs_ymin,
        "xmax": abs_xmax,
        "ymax": abs_ymax,
    }


def match_feature_from_label(label: str) -> Optional[str]:
    label = str(label).strip().lower()
    for feature, aliases in DETECTION_LABEL_MAP.items():
        alias_set = {a.lower() for a in aliases}
        if label in alias_set:
            return feature
    return None


def save_detection_boxes(
    record: dict,
    width: int,
    height: int,
    raw_items: list[dict],
    boxes_dir: str,
) -> tuple[str, str]:
    """
    Saves per-image detection boxes to disk and returns:
    - compact JSON string for CSV
    - path to saved JSON file
    """
    pano_part = record.get("pano_id") or "no_pano"
    out_path = Path(boxes_dir) / f"{record['id']}_{pano_part}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    saved_items = []
    for item in raw_items:
        label = str(item.get("label", "")).strip()
        box_2d = item.get("box_2d")

        if not isinstance(box_2d, list) or len(box_2d) != 4:
            continue

        matched_feature = match_feature_from_label(label)
        pixel_box = normalized_box_to_pixels(box_2d, width, height)

        saved_items.append({
            "label": label,
            "matched_feature": matched_feature,
            "box_2d": box_2d,
            "box_pixels": pixel_box,
        })

    payload = {
        "id": record["id"],
        "pano_id": record.get("pano_id"),
        "image_path": record.get("image_path"),
        "image_width": width,
        "image_height": height,
        "detections": saved_items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    compact_json = json.dumps(saved_items, ensure_ascii=False)
    return compact_json, str(out_path)

def select_best_panorama(lat: float, lon: float):
    panos = search_panoramas(lat=lat, lon=lon)
    if not panos:
        return None

    candidates = []
    for pano in panos:
        dist = distance_meters(lat, lon, pano.lat, pano.lon)
        candidates.append((dist, pano))

    candidates.sort(key=lambda x: x[0])

    for dist, pano in candidates:
        if dist > MAX_DISTANCE_METERS:
            continue

        pano_date = pano.date

        if not pano_date and GOOGLE_MAPS_API_KEY:
            try:
                meta = get_panorama_meta(pano_id=pano.pano_id, api_key=GOOGLE_MAPS_API_KEY)
                pano_date = meta.date
            except Exception:
                pano_date = None

        if is_recent_enough(pano_date, MIN_YEAR):
            return dist, pano, pano_date

    return None


def save_panorama_image(pano_id: str, output_path: str, zoom: int = PANORAMA_ZOOM) -> None:
    output = Path(output_path)
    if output.exists():
        return

    image = get_panorama(pano_id=pano_id, zoom=zoom, multi_threaded=False)
    image = crop_bottom_and_right_black_border(image)

    max_height = 1685
    if image.height > max_height:
        new_width = int(image.width * (max_height / image.height))
        image = image.resize((new_width, max_height), Image.LANCZOS)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "JPEG", quality=92)


def load_existing_results(output_csv: str) -> Dict[str, Dict[str, Any]]:
    path = Path(output_csv)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "id" not in df.columns:
        return {}

    df = df.where(pd.notna(df), None)
    return {str(row["id"]): row.to_dict() for _, row in df.iterrows()}


def record_has_download_cache(record: Dict[str, Any]) -> bool:
    image_path = record.get("image_path")
    pano_id = record.get("pano_id")
    if image_path and pano_id and Path(image_path).exists():
        return True

    # If we previously determined there is no valid panorama, avoid re-calling Street View.
    if record.get("status") == "skipped" and record.get("reason") == "no_pano_within_10m_and_2015plus":
        return True

    return False


def record_has_gemini_cache(record: Dict[str, Any], analysis_mode: Optional[str] = None) -> bool:
    has_preds = all(record.get(field) in (0, 1) for field in PREDICTION_FIELDS)
    if not has_preds:
        return False

    if analysis_mode is None:
        return has_preds

    if record.get("analysis_mode_used") != analysis_mode:
        return False

    if analysis_mode == "detection":
        boxes_path = record.get("detection_boxes_path")
        return bool(boxes_path and Path(boxes_path).exists())

    return True


def merge_cached_data(base: Dict[str, Any], cached: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cached:
        return base

    merged = dict(base)
    for key, value in cached.items():
        if key not in merged:
            continue
        if value is not None:
            merged[key] = value
    return merged


def run_map(func, items: List[Dict[str, Any]], num_cores: int, debug: bool, desc: str):
    if debug or num_cores == 1:
        results = []
        for item in tqdm(items, total=len(items), desc=desc, unit="img"):
            results.append(func(item))
        return results

    with multiprocessing.Pool(num_cores) as pool:
        return list(tqdm(pool.imap(func, items), total=len(items), desc=desc, unit="img"))


# =========================
# WORKER FUNCTIONS
# =========================

def download_worker(record: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """
    Phase 1: downloads image + panorama metadata, but only if not already cached.
    """
    # breakpoint()
    try:
        if record_has_download_cache(record):
            # Preserve prior skip result
            if record.get("status") == "skipped" and record.get("reason") == "no_pano_within_10m_and_2015plus":
                return record

            # Existing local image + metadata
            if record.get("image_path") and Path(record["image_path"]).exists():
                if not record_has_gemini_cache(record):
                    record["status"] = "downloaded"
                    record["reason"] = "download_cache_hit"
                return record

        selected = select_best_panorama(record["lat"], record["lon"])
        if selected is None:
            record["status"] = "skipped"
            record["reason"] = "no_pano_within_10m_and_2015plus"
            return record

        dist, pano, pano_date = selected

        record["pano_id"] = pano.pano_id
        record["pano_lat"] = pano.lat
        record["pano_lon"] = pano.lon
        record["pano_distance_m"] = round(dist, 3)
        record["pano_date"] = pano_date

        image_path = os.path.join(image_dir, f"{record['id']}_{pano.pano_id}.jpg")
        save_panorama_image(pano_id=pano.pano_id, output_path=image_path)

        record["image_path"] = image_path
        if record_has_gemini_cache(record):
            record["status"] = "ok"
            record["reason"] = "fully_cached"
        else:
            record["status"] = "downloaded"
            record["reason"] = "downloaded"

    except Exception as e:
        record["status"] = "error"
        record["reason"] = f"Download failed: {str(e)}"

    return record


def analyze_with_prompt(
    image_path: str,
    prompt_text: str,
    gemini_api_key: str,
    gemini_model: str,
) -> dict:
    client = genai.Client(api_key=gemini_api_key)

    uploaded = client.files.upload(file=image_path)

    response = client.models.generate_content(
        model=gemini_model,
        contents=[uploaded, prompt_text],
        config={
            "response_mime_type": "application/json",
            "response_schema": BuiltEnvPrediction,
        },
    )

    return BuiltEnvPrediction.model_validate_json(response.text).model_dump()


def analyze_with_detection(
    image_path: str,
    gemini_api_key: str,
    gemini_model: str,
    record: dict,
    boxes_dir: str,
) -> dict:
    client = genai.Client(api_key=gemini_api_key)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    prompt = build_detection_prompt()

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        # thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=[image, prompt],
        config=config,
    )

    raw_items = parse_json_response(response.text)

    output = {feature: 0 for feature in PREDICTION_FIELDS}

    for item in raw_items:
        label = str(item.get("label", "")).strip().lower()
        for feature, aliases in DETECTION_LABEL_MAP.items():
            alias_set = {a.lower() for a in aliases}
            if label in alias_set:
                output[feature] = 1

    detection_boxes_json, detection_boxes_path = save_detection_boxes(
        record=record,
        width=width,
        height=height,
        raw_items=raw_items,
        boxes_dir=boxes_dir,
    )

    output["detection_boxes_json"] = detection_boxes_json
    output["detection_boxes_path"] = detection_boxes_path

    return output


def analyze_worker(
    record: dict,
    prompt_text: str,
    gemini_api_key: str,
    gemini_model: str,
    analysis_mode: str,
    boxes_dir: str,
) -> dict:
    if record_has_gemini_cache(record, analysis_mode=analysis_mode):
        record["status"] = "ok"
        record["reason"] = f"gemini_cache_hit_{analysis_mode}"
        return record

    if not record.get("image_path") or not Path(record["image_path"]).exists():
        return record

    if record.get("status") not in ("downloaded", "ok"):
        return record

    try:
        if analysis_mode == "prompt":
            prediction_data = analyze_with_prompt(
                image_path=record["image_path"],
                prompt_text=prompt_text,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
            )
        elif analysis_mode == "detection":
            prediction_data = analyze_with_detection(
                image_path=record["image_path"],
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
                record=record,
                boxes_dir=boxes_dir,
            )
        else:
            raise ValueError(f"Unsupported analysis_mode: {analysis_mode}")

        record.update(prediction_data)
        record["analysis_mode_used"] = analysis_mode
        record["status"] = "ok"
        record["reason"] = f"success_{analysis_mode}"

    except Exception as e:
        record["status"] = "error"
        record["reason"] = f"Gemini {analysis_mode} failed: {str(e)}"

    return record

# =========================
# ARGPARSE
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Google Street View panoramas and annotate them with Gemini."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Path to input CSV.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Path to output CSV.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Directory for downloaded images.")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Path to Gemini prompt file.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
        help="Number of worker processes."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in single-core debug mode without multiprocessing."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of rows to process."
    )
    parser.add_argument(
    "--analysis-mode",
    choices=["prompt", "detection"],
    default="prompt",
    help="Use the original prompt-based classifier or Gemini object detection.",
    )
    parser.add_argument(
        "--boxes-dir",
        default="detection_boxes",
        help="Directory to save per-image bounding box JSON files.",
    )
    return parser.parse_args()


# =========================
# MAIN ORCHESTRATOR
# =========================

def main():
    args = parse_args()

    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    if not Path(args.prompt_file).exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    prompt_text = load_prompt(args.prompt_file)

    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Input CSV must contain 'lat' and 'lon' columns.")

    Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_results(args.output_csv)

    initial_records = []
    for idx, row in df.iterrows():
        point_id = make_point_id(row, idx)
        base = empty_record(
            point_id=point_id,
            lat=float(row["lat"]),
            lon=float(row["lon"]),
        )
        merged = merge_cached_data(base, existing_results.get(point_id))
        initial_records.append(merged)

    total_items = len(initial_records)
    num_cores = 1 if args.debug else max(1, args.num_cores)

    # -------- Phase 1: download only when needed --------
    print(f"\n--- PHASE 1: DOWNLOADING IMAGES ({num_cores} worker(s)) ---")
    download_func = partial(download_worker, image_dir=args.image_dir)
    downloaded_records = run_map(
        func=download_func,
        items=initial_records,
        num_cores=num_cores,
        debug=args.debug,
        desc="Downloading",
    )

    # Write checkpoint after download phase
    checkpoint_df = pd.DataFrame(downloaded_records)
    checkpoint_df.to_csv(args.output_csv, index=False)
    print(f"Checkpoint saved after download phase: {args.output_csv}")

    # -------- Phase 2: Gemini only when needed --------
    print(f"\n--- PHASE 2: INFERENCE WITH GEMINI ({num_cores} worker(s)) ---")
    analyze_func = partial(
        analyze_worker,
        prompt_text=prompt_text,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=args.gemini_model,
        analysis_mode=args.analysis_mode,
        boxes_dir=args.boxes_dir,
    )
    final_records = run_map(
        func=analyze_func,
        items=downloaded_records,
        num_cores=num_cores,
        debug=args.debug,
        desc="Analyzing",
    )

    out_df = pd.DataFrame(final_records)
    out_df.to_csv(args.output_csv, index=False)

    print(f"\nDone! Processed {len(final_records)} records.")
    print(f"Results written to: {args.output_csv}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()