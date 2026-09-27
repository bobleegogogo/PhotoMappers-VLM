import io
import re
import json
import base64
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from openai import OpenAI
from tqdm import tqdm


API_KEY = "EMPTY"

MODEL_NAME = "InternVL3-8B"
BASE_URL = "http://localhost:8000/v1"

PAIR_FILE = Path("./CVGLresults_fromFabian_R@1.xlsx")

VGI_DIR = Path(
    r"./VGI"
)

SVI_DIR = Path(
    r"./SVI"
)

RSI_DIR = Path(
    r"./RSI"
)

OUTPUT_JSON = Path(
    r"json"
)

MAX_WORKERS = 32
MAX_RETRIES = 3
MAX_IMAGE_SIDE = 1024
MAX_OUTPUT_TOKENS = 20000

OUTPUT_JSONL = OUTPUT_JSON.with_suffix(".jsonl")

RESPONSE_JSON_SCHEMA = {
    "type": "array",
    "minItems": 30,
    "maxItems": 30,
    "items": {
        "type": "object",
        "properties": {
            "VGI filename": {"type": "string"},
            "SVI filename": {"type": "string"},
            "RSI filename": {"type": "string"},
            "indicator name": {"type": "string"},
            "reasoning explanation": {"type": "string"},
            "indicator score": {"type": "integer", "enum": [0, 1, 2]},
            "total score": {"type": "integer"},
            "matching decision": {"type": "string", "enum": ["Yes", "No"]}
        },
        "required": [
            "VGI filename",
            "SVI filename",
            "RSI filename",
            "indicator name",
            "reasoning explanation",
            "indicator score",
            "total score",
            "matching decision"
        ],
        "additionalProperties": False
    }
}

FIXED_INDICATORS = [
    "Building type and scale",
    "Overall building appearance",
    "Building position relative to nearby roads or open space",
    "Building position and shape correspondence in SVI--RSI",
    "Recognizable building outline",
    "Distinctive building appearance or visible name",
    "Road type and width",
    "Road markings and boundary features",
    "Road direction and connectivity",
    "Road position relative to surrounding elements",
    "Distinctive intersection, bridge, or road curve",
    "Traffic lights, road signs, or road markers",
    "Vegetation coverage and type",
    "Ground surface characteristics",
    "Vegetation position relative to roads or buildings",
    "Waterbody, shoreline, or open-space position",
    "Prominent trees, forest belts, or distinctive vegetation",
    "Coastline, river, lake, or slope outline",
    "Utility poles, streetlights, and cables",
    "Road signs, fences, and guardrails",
    "Facility position relative to roads or buildings",
    "Local facility grouping and arrangement",
    "Large facilities such as antennas, water towers, or pylons",
    "Recognizable signs such as storefront signs, billboards, or gas-station signs",
    "Scene functional type",
    "Spatial density and openness",
    "Overall building--road--vegetation relationship",
    "Consistency between the VGI visible scene and the SVI--RSI overall scene",
    "Large building complex, community, or commercial area",
    "Coastline, river, bridge, or regional background"
]

PROMPT = """
This is a post-disaster Volunteered Geographic Information (VGI) image with corresponding SVI and RSI reference images. Please determine whether the VGI image and the SVI--RSI image pair represent the same geolocation.

Evaluate the match between the VGI image and the SVI--RSI image pair using the following 30 indicators. For each indicator, provide a concise but sufficiently informative explanation and assign a score. Finally, output the total score and the final matching decision. The final decision must be either "Yes" or "No".

Please generate the result in a long-table format with the following fields: VGI filename, SVI filename, RSI filename, indicator name, reasoning explanation, indicator score, total score, and matching decision. Each image triplet should contain 30 rows, with one row for each indicator. The filenames, total score, and matching decision should remain the same across the 30 rows for the same image triplet. The matching decision can only be "Yes" or "No".

Scoring scheme:
2 = Correct: Clearly supported by the VGI image and at least one reference view, or consistently absent across views, with no contradiction from the other view.
1 = Partially correct: Weakly or partially supported, with insufficient, ambiguous, or incomplete cross-view evidence.
0 = Incorrect: Unsupported, contradicted across views, spatially mismatched, or hallucinated.

Notes:
1)Do not evaluate the match between SVI and RSI, as they are already treated as a matched reference pair.
2)Do not use disaster damage as matching evidence, since the VGI image is post-disaster while SVI and RSI come from other times.
3)Use a containment-based judgment: the local scene visible in VGI only needs to be reasonably supported by or contained within part of the SVI--RSI reference pair. One-to-one correspondence between all elements is not required.


Class: Building and structure
Appearance
• Building type and scale
• Overall building appearance
Spatial Layout
• Building position relative to nearby roads or open space
• Building position and shape correspondence in SVI--RSI
Landmark Cue
• Recognizable building outline
• Distinctive building appearance or visible name

Class: Road and transport
Appearance
• Road type and width
• Road markings and boundary features
Spatial Layout
• Road direction and connectivity
• Road position relative to surrounding elements
Landmark Cue
• Distinctive intersection, bridge, or road curve
• Traffic lights, road signs, or road markers

Class: Vegetation and terrain
Appearance
• Vegetation coverage and type
• Ground surface characteristics
Spatial Layout
• Vegetation position relative to roads or buildings
• Waterbody, shoreline, or open-space position
Landmark Cue
• Prominent trees, forest belts, or distinctive vegetation
• Coastline, river, lake, or slope outline

Class: Objects and facilities
Appearance
• Utility poles, streetlights, and cables
• Road signs, fences, and guardrails
Spatial Layout
• Facility position relative to roads or buildings
• Local facility grouping and arrangement
Landmark Cue
• Large facilities such as antennas, water towers, or pylons
• Recognizable signs such as storefront signs, billboards, or gas-station signs

Class: Global scene
Appearance
• Scene functional type
• Spatial density and openness
Spatial Layout
• Overall building--road--vegetation relationship
• Consistency between the VGI visible scene and the SVI--RSI overall scene
Landmark Cue
• Large building complex, community, or commercial area
• Coastline, river, bridge, or regional background
"""

lock = threading.Lock()


def encode_image(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")

        if MAX_IMAGE_SIDE is not None:
            longest = max(img.size)

            if longest > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / longest

                new_width = max(
                    1,
                    int(img.width * scale)
                )

                new_height = max(
                    1,
                    int(img.height * scale)
                )

                img = img.resize(
                    (new_width, new_height),
                    Image.LANCZOS
                )

        buf = io.BytesIO()

        img.save(
            buf,
            format="JPEG",
            quality=90
        )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")
    )


def find_image(folder: Path, filename) -> Path:
    if pd.isna(filename):
        raise ValueError(
            f"Image filename is missing: {folder}"
        )

    filename = str(filename).strip()

    for p in folder.iterdir():
        if (
            p.is_file()
            and p.name.lower() == filename.lower()
        ):
            return p

    raise FileNotFoundError(
        f"Missing image {filename} in {folder}"
    )


def try_repair_json(candidate: str):
    try:
        from json_repair import repair_json
    except ImportError:
        return None

    try:
        repaired = repair_json(candidate)

        if isinstance(repaired, (list, dict)):
            return repaired

        return json.loads(repaired)

    except Exception:
        return None


def extract_json_array(text: str):
    if text is None:
        raise ValueError(
            "Model returned content=None"
        )

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Model returned empty content"
        )

    original_text = text

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    lower_text = text.lower()

    if "</think>" in lower_text:
        pos = lower_text.rfind("</think>")

        text = text[
            pos + len("</think>"):
        ].strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json|JSON)?\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

    candidates = []

    if text:
        candidates.append(text)

    start = text.find("[")
    end = text.rfind("]")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        array_candidate = text[start:end + 1]

        if array_candidate not in candidates:
            candidates.append(
                array_candidate
            )

    for candidate in candidates:
        try:
            result = json.loads(candidate)

            if isinstance(result, list):
                return result

        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        try:
            result = json.loads(
                candidate,
                strict=False
            )

            if isinstance(result, list):
                return result

        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        repaired = try_repair_json(
            candidate
        )

        if isinstance(repaired, list):
            return repaired

    raise ValueError(
        "Could not parse a valid JSON array from the model response."
        "\nFirst 1500 characters of the raw model output:\n"
        + repr(original_text[:1500])
    )


def call_gemini(
    client: OpenAI,
    vgi: Path,
    svi: Path,
    rsi: Path
):
    content = [
        {
            "type": "text",
            "text": f"""

VGI filename: {vgi.name}

SVI filename: {svi.name}

RSI filename: {rsi.name}



Evaluate the three images according to the system instructions.

"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image(vgi)
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image(svi)
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": encode_image(rsi)
            }
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "indicator_scores",
                    "schema": RESPONSE_JSON_SCHEMA
                }
            }
        },
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    choice = response.choices[0]
    message = choice.message

    raw_text = message.content

    reasoning_text = getattr(
        message,
        "reasoning_content",
        None
    )

    finish_reason = choice.finish_reason

    if raw_text is None or not str(raw_text).strip():
        reasoning_length = (
            len(reasoning_text)
            if reasoning_text
            else 0
        )

        raise ValueError(
            "The model's final response is empty."
            f" finish_reason={finish_reason},"
            f" reasoning_content_length={reasoning_length}"
        )

    result = extract_json_array(
        raw_text
    )

    return result


def process(row, client):
    vgi_name = row[
        "VGI_query_image"
    ]

    reference_name = row[
        "retrieved_image"
    ]

    vgi = find_image(
        VGI_DIR,
        vgi_name
    )

    svi = find_image(
        SVI_DIR,
        reference_name
    )

    rsi = find_image(
        RSI_DIR,
        reference_name
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = call_gemini(
                client,
                vgi,
                svi,
                rsi
            )

            if not isinstance(result, list):
                raise ValueError(
                    "Model output is not a list: "
                    f"{type(result).__name__}"
                )

            if len(result) != 30:
                raise ValueError(
                    f"Model returned "
                    f"{len(result)} indicators "
                    f"instead of 30"
                )

            for i, x in enumerate(result):
                if not isinstance(x, dict):
                    raise ValueError(
                        f"Indicator {i + 1} "
                        f"is not a dict"
                    )

                x["indicator name"] = FIXED_INDICATORS[i]
                x["VGI filename"] = vgi.name
                x["SVI filename"] = svi.name
                x["RSI filename"] = rsi.name

                if "indicator score" not in x:
                    raise ValueError(
                        f"Indicator {i + 1} "
                        f"missing 'indicator score'"
                    )

                try:
                    x["indicator score"] = int(
                        x["indicator score"]
                    )

                except Exception:
                    raise ValueError(
                        f"Invalid indicator score "
                        f"at row {i + 1}: "
                        f"{x.get('indicator score')}"
                    )

                if x["indicator score"] not in (0, 1, 2):
                    raise ValueError(
                        f"Invalid indicator score "
                        f"at row {i + 1}: "
                        f"{x['indicator score']}"
                    )

                if "reasoning explanation" not in x:
                    raise ValueError(
                        f"Indicator {i + 1} "
                        f"missing "
                        f"'reasoning explanation'"
                    )

                if "total score" not in x:
                    raise ValueError(
                        f"Indicator {i + 1} "
                        f"missing 'total score'"
                    )

                if "matching decision" not in x:
                    raise ValueError(
                        f"Indicator {i + 1} "
                        f"missing "
                        f"'matching decision'"
                    )

            calculated_total = sum(
                x["indicator score"]
                for x in result
            )

            for x in result:
                x["total score"] = calculated_total

            decisions = {
                str(x["matching decision"]).strip()
                for x in result
            }

            if len(decisions) != 1:
                raise ValueError(
                    "Inconsistent matching "
                    f"decisions: {decisions}"
                )

            return result

        except Exception as e:
            tqdm.write(
                f"[Retry "
                f"{attempt + 1}/"
                f"{MAX_RETRIES}] "
                f"VGI={vgi.name}, "
                f"Reference="
                f"{reference_name}, "
                f"{type(e).__name__}: "
                f"{e}"
            )

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(
                10 * (attempt + 1)
                + random.random() * 5
            )


def append_to_jsonl(rows):
    with lock:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_pairs():
    completed = set()

    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except Exception:
                    continue

                vgi_name = str(
                    row.get("VGI filename", "")
                ).strip().lower()

                svi_name = str(
                    row.get("SVI filename", "")
                ).strip().lower()

                if vgi_name and svi_name:
                    completed.add((vgi_name, svi_name))

    return completed


def build_json_from_jsonl():
    if not OUTPUT_JSONL.exists():
        return

    rows = []

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    pairs = pd.read_excel(PAIR_FILE)
    total_pairs = len(pairs)

    completed_pairs = load_completed_pairs()

    pending_rows = []

    for idx, row in pairs.iterrows():
        key = (
            str(row["VGI_query_image"]).strip().lower(),
            str(row["retrieved_image"]).strip().lower()
        )

        if key in completed_pairs:
            continue

        pending_rows.append((idx, row))

    random.shuffle(pending_rows)

    skipped_count = total_pairs - len(pending_rows)

    print(f"Total pairs  : {total_pairs}")
    print(f"Already done : {skipped_count} (resume, skipped)")
    print(f"To run       : {len(pending_rows)}")
    print(f"MAX_WORKERS  : {MAX_WORKERS}")
    print(f"MAX_RETRIES  : {MAX_RETRIES}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Base URL     : {BASE_URL}")
    print(f"Output       : {OUTPUT_JSON}")
    print(f"Max tokens   : {MAX_OUTPUT_TOKENS}")
    print("=" * 70)

    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_pair = {}

        for idx, row in pending_rows:
            future = pool.submit(
                process,
                row,
                client
            )

            future_to_pair[future] = (
                idx,
                row["VGI_query_image"],
                row["retrieved_image"]
            )

        with tqdm(
            total=len(pending_rows),
            desc="VGI-SVI-RSI matching",
            unit="pair",
            dynamic_ncols=True
        ) as pbar:
            for future in as_completed(future_to_pair):
                idx, vgi_name, ref_name = future_to_pair[future]

                try:
                    result = future.result()

                    append_to_jsonl(result)

                    success_count += 1

                    if success_count % 50 == 0:
                        with lock:
                            build_json_from_jsonl()

                except Exception as e:
                    failed_count += 1

                    tqdm.write(
                        f"[FAILED] "
                        f"row={idx}, "
                        f"VGI={vgi_name}, "
                        f"Reference="
                        f"{ref_name}, "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

                pbar.set_postfix(
                    success=success_count,
                    failed=failed_count
                )

                pbar.update(1)

    print()
    print("=" * 70)
    print("Finished")
    print(f"Total   : {total_pairs}")
    print(f"Success : {success_count}")
    print(f"Failed  : {failed_count}")
    print(f"Output  : {OUTPUT_JSON}")
    print("=" * 70)

    build_json_from_jsonl()


if __name__ == "__main__":
    main()