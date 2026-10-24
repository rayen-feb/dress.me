import io, json, os, uuid
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from schemas import Attr, Embedding, ScanResult

MODELS = Path(__file__).resolve().parent.parent / "models"
MAX_BYTES = 15 * 1024 * 1024
MIN_SIDE = 64
ALLOWED = {"JPEG", "PNG", "WEBP"}
_S = {}


def _load():
    if _S:
        return _S
    cfg = json.load(open(MODELS / "config.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid = cfg["encoder"]["model_id"]
    _S.update(
        cfg=cfg, device=dev,
        model=CLIPModel.from_pretrained(mid).to(dev).eval(),
        proc=CLIPProcessor.from_pretrained(mid),
        probe=joblib.load(MODELS / "probe.joblib"),
        tc=np.load(MODELS / "text_colour.npy"),
        tp=np.load(MODELS / "text_pattern.npy"),
    )
    return _S


def _as_tensor(x):
    return x if torch.is_tensor(x) else x.pooler_output


def _open(source):
    if isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_BYTES:
            raise ValueError("file too large")
        fh = io.BytesIO(source)
    else:
        if os.path.getsize(source) > MAX_BYTES:
            raise ValueError("file too large")
        fh = source
    try:
        img = Image.open(fh)
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("not a decodable image") from e
    if img.format not in ALLOWED:
        raise ValueError(f"unsupported format: {img.format}")
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise ValueError("image too small")
    return img


def embed(img):
    s = _load()
    with torch.no_grad():
        inp = s["proc"](images=img, return_tensors="pt").to(s["device"])
        f = _as_tensor(s["model"].get_image_features(**inp))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().numpy()


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _decide(labels, probs, auto, k=3):
    order = np.argsort(-probs)[:k]
    top, conf = labels[order[0]], float(min(probs[order[0]], 1.0))
    alts = [labels[i] for i in order[1:]]
    if conf >= auto:
        return Attr(value=top, confidence=conf, source="predicted", alternatives=alts)
    return Attr(value=None, confidence=conf, source="unknown", alternatives=[top] + alts)


def scan_garment(source, include_embedding=True):
    item_id = str(uuid.uuid4())
    try:
        img = _open(source)
        e = embed(img)
    except Exception as ex:
        return ScanResult(item_id=item_id, status="error", error=str(ex)[:200])

    s = _load()
    cfg, t = s["cfg"], s["cfg"]["thresholds"]
    classes, sub2cat = cfg["classes"], cfg["subcategory_to_category"]

    p = s["probe"].predict_proba(e[None])[0]
    order = np.argsort(-p)
    top, conf = classes[order[0]], float(min(p[order[0]], 1.0))
    alts = [classes[i] for i in order[1:3]]
    if conf >= t["category_confirm"]:
        sub = Attr(value=top, confidence=conf, source="predicted", alternatives=alts)
        cat = Attr(value=sub2cat[top], confidence=conf, source="predicted")
    else:
        sub = Attr(value=None, confidence=conf, source="unknown", alternatives=[top] + alts)
        cat = Attr(confidence=conf, source="unknown")
    status = ("ok" if conf >= t["category_auto"]
              else "needs_confirmation" if conf >= t["category_confirm"] else "not_recognised")

    sc = cfg["logit_scale"]
    colour = _decide(cfg["colours"], _softmax(sc * (s["tc"] @ e)), t["colour_auto"])
    pattern = _decide(cfg["patterns"], _softmax(sc * (s["tp"] @ e)), t["pattern_auto"])

    emb = (Embedding(model_id=cfg["encoder"]["model_id"], dimension=len(e),
                     vector=[float(v) for v in e]) if include_embedding else None)
    return ScanResult(item_id=item_id, status=status, category=cat, subcategory=sub,
                      colour=colour, pattern=pattern, embedding=emb)
