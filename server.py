"""DressMe DSO 1 web server.

Run from the project root with:
    python -m uvicorn server:app --reload --port 8000
"""
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
WARDROBE_ROOT = PROJECT_ROOT / "dso1_wardrobe_scan"
sys.path.insert(0, str(WARDROBE_ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference

CFG = json.loads((WARDROBE_ROOT / "models" / "config.json").read_text())
DATA = PROJECT_ROOT / "data"
(DATA / "images").mkdir(parents=True, exist_ok=True)
DB = DATA / "wardrobe.json"
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
    return r.model_dump(exclude={"embedding"})  # embeddings stay on the server


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


@app.delete("/api/wardrobe/items/{item_id}")
def delete(item_id: str):
    write_db([i for i in read_db() if i["item_id"] != item_id])
    (DATA / "images" / f"{item_id}.jpg").unlink(missing_ok=True)
    return {"deleted": item_id}
