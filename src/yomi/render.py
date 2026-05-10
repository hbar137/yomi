from pydantic import BaseModel


class Segment(BaseModel):
    surface: str
    reading: str
    is_kanji: bool
