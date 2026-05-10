from fastapi import FastAPI
from pydantic import BaseModel

from yomi.pipeline import Pipeline
from yomi.render import Segment

app = FastAPI(title="yomi", version="0.1.0")
_pipeline: Pipeline | None = None


class FuriganaRequest(BaseModel):
    text: str


class FuriganaResponse(BaseModel):
    segments: list[Segment]


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/furigana", response_model=FuriganaResponse)
def furigana(req: FuriganaRequest) -> FuriganaResponse:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.load_default()
    segments = _pipeline.run(req.text)
    return FuriganaResponse(segments=segments)
