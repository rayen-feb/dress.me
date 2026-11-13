"""DressMe DSO 1 web server.

Run from the project root with:
    python -m uvicorn server:app --reload --port 8000
"""
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, time, timedelta
from itertools import product
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
WARDROBE_ROOT = PROJECT_ROOT / "dso1_wardrobe_scan"
sys.path.insert(0, str(WARDROBE_ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import inference

CFG = json.loads((WARDROBE_ROOT / "models" / "config.json").read_text())
DATA = PROJECT_ROOT / "data"
(DATA / "images").mkdir(parents=True, exist_ok=True)
DB = DATA / "wardrobe.json"
SHOPPING_LIST_DB = DATA / "shopping_list.json"
CALENDAR_DB = DATA / "calendar_events.json"
PENDING = {}  # item_id -> (image bytes, ScanResult); scans waiting for user confirmation

app = FastAPI(title="DressMe - Digital Wardrobe Scan")
app.mount("/images", StaticFiles(directory=DATA / "images"), name="images")


@app.on_event("startup")
def warm_up():
    inference._load()  # load fashion-clip once, so the first scan is fast


def read_db():
    return json.loads(DB.read_text()) if DB.exists() else []


def write_db(items):
    DB.write_text(json.dumps(items))


def _field_value(value):
    if isinstance(value, dict):
        return value.get("value")
    return value


def _normalise(item):
    data = item if isinstance(item, dict) else {}
    item_id = data.get("item_id")
    return {
        "item_id": item_id,
        "subcategory": _field_value(data.get("subcategory")),
        "category": _field_value(data.get("category")),
        "colour": _field_value(data.get("colour")),
        "pattern": _field_value(data.get("pattern")),
        "image_url": f"/images/{item_id}.jpg" if item_id else None,
    }


def _vector_for(item):
    emb = item.get("embedding") or {}
    vector = emb.get("vector")
    if not vector:
        return None
    try:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not arr.size or not np.isfinite(arr).all():
        return None
    return arr


def _find_duplicates(vector, wardrobe, exclude_id=None):
    try:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return []
    if not query.size or not np.isfinite(query).all():
        return []

    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return []

    matches = []
    for other in wardrobe:
        if other.get("item_id") == exclude_id:
            continue
        embedding = other.get("embedding") or {}
        if embedding.get("model_id") != CFG["encoder"]["model_id"]:
            continue
        other_vector = _vector_for(other)
        if other_vector is None or other_vector.shape != query.shape:
            continue
        other_norm = float(np.linalg.norm(other_vector))
        if other_norm == 0:
            continue
        similarity = float(np.clip(
            np.dot(query, other_vector) / (query_norm * other_norm),
            -1.0,
            1.0,
        ))
        if similarity < 0.65:
            continue
        matches.append({
            "item_id": other.get("item_id"),
            "similarity": round(similarity, 3),
            "image_url": f"/images/{other['item_id']}.jpg",
            "subcategory": _field_value(other.get("subcategory")),
            "colour": _field_value(other.get("colour")),
            "pattern": _field_value(other.get("pattern")),
        })
    matches.sort(key=lambda match: match["similarity"], reverse=True)
    return matches[:5]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/options")
def options():
    return {"subcategories": CFG["classes"], "colours": CFG["colours"], "patterns": CFG["patterns"]}


@app.post("/api/wardrobe/scan")
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    r = inference.scan_garment(data, include_embedding=True)
    if r.status == "error":
        raise HTTPException(422, r.error)
    PENDING[r.item_id] = (data, r)
    while len(PENDING) > 50:
        PENDING.pop(next(iter(PENDING)))
    response = r.model_dump(exclude={"embedding"})
    wardrobe = read_db()
    response["duplicates"] = _find_duplicates(
        r.embedding.vector if r.embedding else [],
        wardrobe,
    )
    response["compared_items"] = len(wardrobe)
    return response  # embeddings stay on the server


class SaveIn(BaseModel):
    item_id: str
    subcategory: str
    colour: Optional[str] = None   # None = user left it unknown
    pattern: Optional[str] = None


@app.post("/api/wardrobe/items")
def save(b: SaveIn):
    if b.item_id not in PENDING:
        raise HTTPException(404, "This scan expired. Upload the photo again.")
    if b.subcategory not in CFG["classes"]:
        raise HTTPException(422, "Unknown garment type.")
    if b.colour and b.colour not in CFG["colours"]:
        raise HTTPException(422, "Unknown colour.")
    if b.pattern and b.pattern not in CFG["patterns"]:
        raise HTTPException(422, "Unknown pattern.")
    data, r = PENDING.pop(b.item_id)

    def field(attr, value):
        if value is None:
            return {"value": None, "source": "unknown"}
        same = attr.source == "predicted" and attr.value == value
        return {"value": value, "source": "predicted" if same else "user_confirmed"}

    # The user pressed save after reviewing, so a value they kept counts as confirmed.
    def confirmed(f):
        return {**f, "source": "user_confirmed"} if f["value"] else f

    item = {
        "schema_version": r.schema_version,
        "item_id": r.item_id,
        "category": {"value": CFG["subcategory_to_category"][b.subcategory], "source": "user_confirmed"},
        "subcategory": confirmed(field(r.subcategory, b.subcategory)),
        "colour": confirmed(field(r.colour, b.colour)),
        "pattern": confirmed(field(r.pattern, b.pattern)),
        "model_confidence": {"subcategory": r.subcategory.confidence},
        "embedding": r.embedding.model_dump(),
    }
    inference._open(data).save(DATA / "images" / f"{r.item_id}.jpg", quality=90)
    write_db(read_db() + [item])
    return {"item_id": r.item_id}


@app.get("/api/wardrobe/items")
def items():
    out = []
    for it in read_db():
        out.append({**{k: v for k, v in it.items() if k != "embedding"},
                    "image_url": f"/images/{it['item_id']}.jpg"})
    return out


@app.get("/api/duplicates/{item_id}")
def duplicates(item_id: str):
    wardrobe = read_db()
    target = next((it for it in wardrobe if it["item_id"] == item_id), None)
    if target is None:
        raise HTTPException(404, "Item not found in wardrobe.")

    target_vector = _vector_for(target)
    if target_vector is None:
        return []
    return _find_duplicates(target_vector, wardrobe, exclude_id=item_id)


class OutfitRequest(BaseModel):
    description: str = Field(min_length=3, max_length=500)
    location: str = Field(default="Tunis", min_length=2, max_length=100)
    event_date: date = Field(default_factory=date.today)
    event_time: Optional[time] = None
    dress_code: Literal["auto", "casual", "smart_casual", "formal", "black_tie"] = "auto"
    setting: Literal["auto", "indoors", "outdoors"] = "auto"

    @field_validator("description", "location")
    @classmethod
    def require_nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value


class ShoppingListCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    detail: str = Field(default="", max_length=500)

    @field_validator("title")
    @classmethod
    def require_title(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("A shopping item needs a title.")
        return value

    @field_validator("detail")
    @classmethod
    def strip_detail(cls, value):
        return value.strip()


class ShoppingListUpdate(BaseModel):
    done: bool


class CalendarEventIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    event_date: date
    event_time: Optional[time] = None
    location: Optional[str] = Field(default=None, max_length=100)
    dress_code: Literal["auto", "casual", "smart_casual", "formal", "black_tie"] = "auto"
    setting: Literal["auto", "indoors", "outdoors"] = "auto"

    @field_validator("title")
    @classmethod
    def require_title(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("An event needs a title.")
        return value

    @field_validator("description")
    @classmethod
    def strip_description(cls, value):
        return value.strip()

    @field_validator("location")
    @classmethod
    def normalise_location(cls, value):
        if value is None:
            return None
        return value.strip() or None


class WeatherLookupError(Exception):
    pass


def _normalise_text(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )
    return value.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
    }))


def _has_phrase(text, phrases):
    return any(
        re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
        for phrase in phrases
    )


EVENT_PHRASES = {
    "wedding": ("wedding", "marriage", "mariage", "3ers", "3رس", "عرس", "زواج", "zawaj", "khitba", "خطوبه", "fiançailles"),
    "work": ("interview", "entretien", "meeting", "reunion", "réunion", "travail", "work", "service", "5edma", "خدمه", "مقابله", "اجتماع", "rdv pro"),
    "graduation": ("graduation", "graduation party", "diploma", "takharruj", "تخرج", "حفله تخرج"),
    "party": ("party", "birthday", "anniversaire", "soiree", "soirée", "gala", "حفله", "عيد ميلاد", "سهرة", "sahra"),
    "date": ("date night", "rendez-vous", "rendez vous", "موعد", "date"),
    "beach": ("beach", "plage", "b7ar", "بحر"),
    "sports": ("sport", "gym", "رياضه", "تمرين"),
    "casual": ("casual", "sortie", "hangout", "coffee", "cafe", "قهوه", "خرجه", "5arja", "kharja"),
}

EVENT_LABELS = {
    "wedding": "Wedding / mariage / 3ers",
    "work": "Work or interview",
    "graduation": "Graduation",
    "party": "Party or celebration",
    "date": "Date",
    "beach": "Beach outing",
    "sports": "Sports or exercise",
    "casual": "Casual outing",
    "other": "Special occasion",
}


def _classify_event(description):
    text = _normalise_text(description)
    for occasion, phrases in EVENT_PHRASES.items():
        if _has_phrase(text, tuple(_normalise_text(phrase) for phrase in phrases)):
            return occasion
    return "other"


def _infer_setting(description):
    text = _normalise_text(description)
    if _has_phrase(text, ("outdoor", "outside", "dehors", "exterieur", "برا", "في الهواء الطلق")):
        return "outdoors"
    if _has_phrase(text, ("indoor", "inside", "interieur", "داخل", "في القاعة")):
        return "indoors"
    return "not specified"


def _infer_dress_code(description, occasion, requested):
    if requested != "auto":
        return requested
    text = _normalise_text(description)
    if _has_phrase(text, ("black tie", "cravate noire", "smoking", "black-tie")):
        return "black_tie"
    if _has_phrase(text, ("formal", "elegant", "habille", "officiel", "رسمي", "شيك")):
        return "formal"
    if _has_phrase(text, ("smart casual", "business casual", "tenue correcte")):
        return "smart_casual"
    if _has_phrase(text, ("casual", "relaxed", "decontracte", "مريح")):
        return "casual"
    if occasion in ("wedding", "graduation"):
        return "formal"
    if occasion in ("work", "date", "party"):
        return "smart_casual"
    return "casual"


def _weather_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DressMe wardrobe demo/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise WeatherLookupError("Weather service is temporarily unavailable.") from exc


def _get_weather(location, event_date, event_time=None):
    today = date.today()
    if event_date < today or event_date > today + timedelta(days=15):
        return None, "A forecast is available only for today through the next 15 days."

    geo_query = urllib.parse.urlencode({
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    })
    geocoding = _weather_json(
        f"https://geocoding-api.open-meteo.com/v1/search?{geo_query}"
    )
    results = geocoding.get("results") or []
    if not results:
        return None, f"Could not find “{location}”. Check the city name and try again."
    place = results[0]

    forecast_query = urllib.parse.urlencode({
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "hourly": "weather_code,temperature_2m,precipitation_probability",
        "timezone": "auto",
        "start_date": event_date.isoformat(),
        "end_date": event_date.isoformat(),
    })
    forecast = _weather_json(f"https://api.open-meteo.com/v1/forecast?{forecast_query}")
    daily = forecast.get("daily") or {}
    if not daily.get("time"):
        return None, "No forecast is available for that date."
    result = {
        "place": ", ".join(filter(None, (place.get("name"), place.get("admin1"), place.get("country")))),
        "date": event_date.isoformat(),
        "code": daily["weather_code"][0],
        "high_c": daily["temperature_2m_max"][0],
        "low_c": daily["temperature_2m_min"][0],
        "precipitation_chance": (daily.get("precipitation_probability_max") or [None])[0],
        "source": "Open-Meteo",
    }
    if event_time:
        hourly = forecast.get("hourly") or {}
        target = f"{event_date.isoformat()}T{event_time.strftime('%H:00')}"
        try:
            index = hourly.get("time", []).index(target)
            result["event_temperature_c"] = hourly["temperature_2m"][index]
            result["event_precipitation_chance"] = hourly["precipitation_probability"][index]
            result["event_code"] = hourly["weather_code"][index]
        except (ValueError, IndexError, KeyError):
            pass
    return result, None


def _weather_summary(weather):
    if not weather:
        return None
    code = weather.get("event_code", weather["code"])
    if code == 0:
        description = "clear"
    elif code in (1, 2, 3):
        description = "partly cloudy"
    elif code in (45, 48):
        description = "foggy"
    elif code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        description = "rainy"
    elif code in (71, 73, 75, 77, 85, 86):
        description = "snowy"
    elif code in (95, 96, 99):
        description = "thunderstorms"
    else:
        description = "changeable"
    weather["description"] = description
    return weather


def _event_weather_advice(weather, setting):
    if not weather:
        return []
    advice = []
    high = weather.get("event_temperature_c")
    if high is None:
        high = weather.get("high_c")
    low = weather.get("low_c")
    rain = weather.get("event_precipitation_chance")
    if rain is None:
        rain = weather.get("precipitation_chance") or 0
    if high is not None and high >= 28:
        advice.append("It may be hot at event time; choose breathable, lighter pieces.")
    elif high is not None and high >= 22:
        advice.append("Mild-to-warm weather should suit lighter layers.")
    if low is not None and low <= 12:
        advice.append("It may feel cold; add a warm layer if you have one.")
    elif low is not None and low <= 17:
        advice.append("Bring a light layer for cooler moments.")
    if rain >= 40 or weather.get("description") in ("rainy", "thunderstorms", "snowy"):
        if setting == "indoors":
            advice.append("Rain may affect travel to the indoor event; consider a covered layer and practical shoes.")
        else:
            advice.append("Rain or unsettled weather is possible; consider a covered layer and practical shoes.")
    if not advice:
        advice.append(f"Forecast: {weather['description']}, {low}–{high}°C.")
    return advice


def _outfit_score(pieces, occasion, dress_code, setting, weather):
    types = {piece.get("subcategory") for piece in pieces}
    score = 0
    if occasion in ("wedding", "graduation", "party"):
        score += 3 if types & {"dress", "blazer"} else 0
        score += 2 if types & {"shirt", "blouse_top"} else 0
        score -= 2 if types & {"hoodie", "shorts"} else 0
    elif occasion == "work":
        score += 3 if types & {"blazer", "shirt", "blouse_top"} else 0
        score += 2 if "trousers" in types else 0
        score -= 3 if types & {"hoodie", "shorts", "hat"} else 0
    elif occasion == "beach":
        score += 3 if types & {"shorts", "t_shirt", "dress", "skirt"} else 0
        score -= 3 if types & {"blazer", "outerwear"} else 0
    elif occasion in ("casual", "sports"):
        score += 2 if types & {"t_shirt", "hoodie", "shorts"} else 0
    if dress_code in ("formal", "black_tie"):
        score += 3 if types & {"dress", "blazer", "shirt"} else 0
        score -= 3 if types & {"hoodie", "shorts"} else 0
    elif dress_code == "smart_casual":
        score += 2 if types & {"blazer", "shirt", "blouse_top", "trousers"} else 0
    if setting == "outdoors":
        score += 1 if "shoes" in types else 0
    elif setting == "indoors":
        score += 1 if types & {"blazer", "shirt", "blouse_top"} else 0
    if weather:
        event_temperature = weather.get("event_temperature_c")
        if event_temperature is None:
            event_temperature = weather.get("high_c")
        if event_temperature is not None and event_temperature >= 27:
            score += 2 if types & {"t_shirt", "shorts", "skirt"} else 0
            score -= 2 if types & {"hoodie", "outerwear", "blazer"} else 0
        if weather.get("low_c") is not None and weather["low_c"] <= 12:
            score += 2 if types & {"hoodie", "outerwear", "blazer", "longsleeve"} else 0
        rain = weather.get("event_precipitation_chance")
        if rain is None:
            rain = weather.get("precipitation_chance") or 0
        if rain >= 40:
            score += 1 if types & {"outerwear", "blazer"} else 0
    colours = {_field_value(piece.get("colour")) for piece in pieces}
    neutral = {"black", "white", "grey", "beige", "brown"}
    if len(colours & neutral) >= 2:
        score += 1
    return score


def _recommend_event_outfits(items, occasion, dress_code, setting, weather):
    categories = {}
    for item in items:
        value = _normalise(item)
        categories.setdefault(value["category"], []).append(value)

    tops = (categories.get("tops") or [])[:8]
    bottoms = (categories.get("bottoms") or [])[:8]
    shoes = (categories.get("footwear") or [])[:5]
    outerwear = (categories.get("outerwear") or [])[:5]
    one_piece = (categories.get("one_piece") or [])[:8]
    candidates = []
    if tops and bottoms:
        for top, bottom in product(tops, bottoms):
            candidates.append([top, bottom])
    candidates.extend([[piece] for piece in one_piece])

    scored = []
    for base in candidates:
        shoe_options = shoes[:3] if shoes else [None]
        layer_options = outerwear[:3] if outerwear else [None]
        for shoe, layer in product(shoe_options, layer_options):
            pieces = list(base)
            if shoe:
                pieces.append(shoe)
            event_temperature = weather.get("event_temperature_c") if weather else None
            event_rain = weather.get("event_precipitation_chance") if weather else None
            if weather and event_rain is None:
                event_rain = weather.get("precipitation_chance") or 0
            if layer and weather and (
                (event_temperature is not None and event_temperature <= 17)
                or (weather.get("low_c") is not None and weather["low_c"] <= 17)
                or event_rain >= 40
            ):
                pieces.append(layer)
            score = _outfit_score(pieces, occasion, dress_code, setting, weather)
            scored.append((score, pieces))

    scored.sort(key=lambda result: result[0], reverse=True)
    recommendations = []
    seen = set()
    for score, pieces in scored:
        signature = tuple(piece["item_id"] for piece in pieces)
        if signature in seen:
            continue
        seen.add(signature)
        if occasion in ("wedding", "graduation", "party"):
            title = "Celebration look"
        elif occasion == "work":
            title = "Polished look"
        elif occasion == "beach":
            title = "Easy beach look"
        else:
            title = "Everyday look"
        reasons = [f"Styled for {EVENT_LABELS[occasion].lower()}"]
        if weather:
            reasons.append(f"{weather['description']} weather")
        recommendations.append({
            "title": title if len(recommendations) == 0 else f"{title} {len(recommendations) + 1}",
            "pieces": pieces,
            "reason": " · ".join(reasons),
        })
        if len(recommendations) == 3:
            break
    return recommendations


@app.post("/api/outfits/recommendations")
def outfit_recommendations(payload: OutfitRequest):
    occasion = _classify_event(payload.description)
    dress_code = _infer_dress_code(payload.description, occasion, payload.dress_code)
    setting = payload.setting if payload.setting != "auto" else _infer_setting(payload.description)
    try:
        weather, weather_notice = _get_weather(
            payload.location.strip(), payload.event_date, payload.event_time
        )
    except WeatherLookupError as exc:
        weather, weather_notice = None, str(exc)
    weather = _weather_summary(weather)

    recommendations = _recommend_event_outfits(
        read_db(), occasion, dress_code, setting, weather
    )
    inspiration_query = " ".join(filter(None, (
        EVENT_LABELS[occasion],
        dress_code.replace("_", " "),
        weather["description"] if weather else None,
        "outfit ideas",
    )))
    result = {
        "event": {
            "description": payload.description.strip(),
            "occasion": occasion,
            "occasion_label": EVENT_LABELS[occasion],
            "dress_code": dress_code,
            "setting": setting,
            "location": payload.location.strip(),
            "date": payload.event_date.isoformat(),
            "time": payload.event_time.isoformat(timespec="minutes") if payload.event_time else None,
        },
        "weather": weather,
        "weather_notice": weather_notice,
        "weather_advice": _event_weather_advice(weather, setting),
        "recommendations": recommendations,
        "wardrobe_note": None if recommendations else (
            "Add a top and bottom, or a dress, to your wardrobe to get outfit combinations."
        ),
        "inspiration_url": (
            "https://www.pinterest.com/search/pins/?q="
            + urllib.parse.quote(inspiration_query)
        ),
    }
    return result


def _calendar_event_data(event_id, payload):
    return {
        "id": event_id,
        "title": payload.title,
        "description": payload.description,
        "event_date": payload.event_date.isoformat(),
        "event_time": payload.event_time.isoformat(timespec="minutes") if payload.event_time else None,
        "location": payload.location,
        "dress_code": payload.dress_code,
        "setting": payload.setting,
    }


@app.get("/api/calendar/events")
def calendar_events():
    events = json.loads(CALENDAR_DB.read_text()) if CALENDAR_DB.exists() else []
    return sorted(events, key=lambda item: (item["event_date"], item.get("event_time") or ""))


@app.post("/api/calendar/events")
def create_calendar_event(payload: CalendarEventIn):
    events = json.loads(CALENDAR_DB.read_text()) if CALENDAR_DB.exists() else []
    event = _calendar_event_data(str(uuid4()), payload)
    events.append(event)
    CALENDAR_DB.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    return event


@app.put("/api/calendar/events/{event_id}")
def update_calendar_event(event_id: str, payload: CalendarEventIn):
    events = json.loads(CALENDAR_DB.read_text()) if CALENDAR_DB.exists() else []
    index = next((i for i, item in enumerate(events) if item["id"] == event_id), None)
    if index is None:
        raise HTTPException(404, "Calendar event not found.")
    event = _calendar_event_data(event_id, payload)
    events[index] = event
    CALENDAR_DB.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    return event


@app.delete("/api/calendar/events/{event_id}")
def delete_calendar_event(event_id: str):
    events = json.loads(CALENDAR_DB.read_text()) if CALENDAR_DB.exists() else []
    remaining = [item for item in events if item["id"] != event_id]
    if len(remaining) == len(events):
        raise HTTPException(404, "Calendar event not found.")
    CALENDAR_DB.write_text(json.dumps(remaining, ensure_ascii=False, indent=2))
    return {"deleted": event_id}


@app.get("/api/shopping/advice")
def shopping_advice():
    items = read_db()
    if not items:
        return {"advice": [{
            "title": "Start with a capsule",
            "detail": "Add a few versatile tops, trousers and shoes to build a useful base wardrobe.",
            "priority": "medium",
        }]}

    by_category = {}
    for item in items:
        category = _field_value(item.get("category"))
        if category:
            by_category.setdefault(category, []).append(item)

    advice = []
    missing = [
        ("tops", "A light neutral top is easy to pair with everything."),
        ("bottoms", "A pair of straight trousers will make most outfits easier to style."),
        ("footwear", "A simple pair of shoes completes casual and smart looks."),
        ("outerwear", "A fitted layer adds polish and warmth for different outfits."),
    ]
    for category, detail in missing:
        if category not in by_category:
            advice.append({
                "title": f"Add more {category}",
                "detail": detail,
                "priority": "high",
            })

    if not advice:
        frequent_colours = {}
        for item in items:
            colour = (item.get("colour") or {}).get("value")
            if colour:
                frequent_colours[colour] = frequent_colours.get(colour, 0) + 1
        if frequent_colours:
            most_common = max(frequent_colours, key=frequent_colours.get)
            advice.append({
                "title": "Keep to a flexible palette",
                "detail": f"Most of your wardrobe is in {most_common}. A few accent colours would help create more variety.",
                "priority": "medium",
            })

    if len(items) >= 5:
        advice.append({
            "title": "Look for outfit balance",
            "detail": "Your wardrobe is already useful. Search for a missing layer, a second colour family, or a different pattern to improve mix-and-match options.",
            "priority": "low",
        })

    return {"advice": advice[:4]}


@app.get("/api/shopping/list")
def shopping_list():
    return json.loads(SHOPPING_LIST_DB.read_text()) if SHOPPING_LIST_DB.exists() else []


@app.post("/api/shopping/list")
def add_shopping_list_item(payload: ShoppingListCreate):
    items = json.loads(SHOPPING_LIST_DB.read_text()) if SHOPPING_LIST_DB.exists() else []
    item = {
        "id": str(uuid4()),
        "title": payload.title,
        "detail": payload.detail,
        "done": False,
    }
    items.append(item)
    SHOPPING_LIST_DB.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return item


@app.patch("/api/shopping/list/{item_id}")
def update_shopping_list_item(item_id: str, payload: ShoppingListUpdate):
    items = json.loads(SHOPPING_LIST_DB.read_text()) if SHOPPING_LIST_DB.exists() else []
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        raise HTTPException(404, "Shopping list item not found.")
    item["done"] = payload.done
    SHOPPING_LIST_DB.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return item


@app.delete("/api/shopping/list/{item_id}")
def delete_shopping_list_item(item_id: str):
    items = json.loads(SHOPPING_LIST_DB.read_text()) if SHOPPING_LIST_DB.exists() else []
    remaining = [item for item in items if item["id"] != item_id]
    if len(remaining) == len(items):
        raise HTTPException(404, "Shopping list item not found.")
    SHOPPING_LIST_DB.write_text(json.dumps(remaining, ensure_ascii=False, indent=2))
    return {"deleted": item_id}


@app.delete("/api/wardrobe/items/{item_id}")
def delete(item_id: str):
    write_db([i for i in read_db() if i["item_id"] != item_id])
    (DATA / "images" / f"{item_id}.jpg").unlink(missing_ok=True)
    return {"deleted": item_id}
