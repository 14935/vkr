import os
import asyncio
import anyio
from anyio import fail_after
from anyio.exceptions import TimeoutError
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schema import AudioRes, QReq, QRes
from .di import get_pipe
from .utils import sanitize_group

core = FastAPI()
core.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ASR_CONCURRENCY = int(os.getenv("ASR_CONCURRENCY", "1"))
GEN_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "1"))
UPSERT_CONCURRENCY = int(os.getenv("UPSERT_CONCURRENCY", "2"))

SEM_ASR = asyncio.Semaphore(ASR_CONCURRENCY)
SEM_GEN = asyncio.Semaphore(GEN_CONCURRENCY)
SEM_UPSERT = asyncio.Semaphore(UPSERT_CONCURRENCY)

@core.post("/audio", response_model=AudioRes)
async def put_audio(file: UploadFile = File(...), group: str = Query('trash')):
    p = get_pipe()
    group = sanitize_group(group)
    data = await file.read()
    async with SEM_ASR:
        try:
            async with fail_after(float(os.getenv("ASR_TIMEOUT", "600"))):
                txt = await anyio.to_thread.run_sync(p.asr_bytes, data)
        except TimeoutError:
            raise HTTPException(status_code=503, detail="ASR busy")
    async with SEM_UPSERT:
        items = await anyio.to_thread.run_sync(p.upsert, group, txt)
    return AudioRes(text=txt, items=items)

@core.post("/qa", response_model=QRes)
async def get_qa(req: QReq):
    p = get_pipe()
    group = sanitize_group(req.target)
    async with SEM_GEN:
        try:
            async with fail_after(float(os.getenv("GEN_TIMEOUT", "180"))):
                answer = await anyio.to_thread.run_sync(p.ask, req.query, group)
        except TimeoutError:
            raise HTTPException(status_code=503, detail="Generator busy")
    return QRes(answer=answer)

@core.get("/groups")
def get_groups():
    return {"groups": get_pipe().db.names()}
