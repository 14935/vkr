from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from .schema import AudioRes, QReq, QRes
from .di import get_pipe

core = FastAPI()
core.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@core.post("/audio", response_model=AudioRes)
async def put_audio(file: UploadFile = File(...), group: str = Query('trash')):
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+file.filename.split(".")[-1]) as f:
        f.write(await file.read())
        f.flush()
        p = get_pipe()
        txt = p.asr(f.name)
    n = p.upsert(group, txt)
    return AudioRes(text=txt, items=n)

@core.post("/qa", response_model=QRes)
async def get_qa(req: QReq):
    answer = get_pipe().ask(req.query, req.target)
    return QRes(answer=answer)

@core.get("/groups")
def get_groups():
    return {"groups": get_pipe().db.names()}
