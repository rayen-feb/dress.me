from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Source = Literal["predicted", "user_confirmed", "unknown", "not_applicable"]
Status = Literal["ok", "needs_confirmation", "not_recognised", "error"]


class Attr(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Source = "unknown"
    alternatives: List[str] = Field(default_factory=list)


class Embedding(BaseModel):
    model_id: str
    dimension: int
    vector: List[float]


class ScanResult(BaseModel):
    schema_version: str = "0.1"
    item_id: str
    status: Status
    category: Attr = Field(default_factory=Attr)
    subcategory: Attr = Field(default_factory=Attr)
    colour: Attr = Field(default_factory=Attr)
    pattern: Attr = Field(default_factory=Attr)
    embedding: Optional[Embedding] = None
    error: Optional[str] = None
