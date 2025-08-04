from pydantic import BaseModel

class AudioRes(BaseModel):
    text: str
    items: int

class QReq(BaseModel):
    query: str
    target: str = 'trash'

class QRes(BaseModel):
    answer: str
