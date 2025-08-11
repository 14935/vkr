import os
import io
import numpy as np
import soundfile as sf
import librosa
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from faster_whisper import WhisperModel
from huggingface_hub import login
from .vectordb import VStore

class Pipe:
    def __init__(self):
        tok = os.getenv("HF_TOKEN")
        if tok:
            try:
                login(tok)
            except Exception:
                pass

        device_pref = os.getenv("DEVICE", "auto").lower()  
        use_gpu = (device_pref != "cpu") and torch.cuda.is_available()
        self.dev = "cuda" if use_gpu else "cpu"

        enc = "intfloat/multilingual-e5-large"
        rer = "cross-encoder/ms-marco-MiniLM-L6-v2"
        gen = "mistralai/Mistral-7B-Instruct-v0.3"

        self.e = SentenceTransformer(enc, device=self.dev)
        self.ek = self.e.tokenizer
        self.r = CrossEncoder(rer, device=self.dev)
        self.d = self.e.get_sentence_embedding_dimension()

        self.gtok = AutoTokenizer.from_pretrained(gen, use_fast=True)
        self.gtok.pad_token = self.gtok.eos_token

        if use_gpu:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.g = AutoModelForCausalLM.from_pretrained(
                gen,
                quantization_config=quant,
                device_map="auto"
            )
            self.w = WhisperModel("large-v3", device="cuda", compute_type="float16")
        else:
            self.g = AutoModelForCausalLM.from_pretrained(
                gen,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )
            self.w = WhisperModel("large-v3", device="cpu", compute_type="int8")

        self.db = VStore(self.d)

    def asr_bytes(self, data: bytes) -> str:
        y, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # в моно
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        segs, _ = self.w.transcribe(y, beam_size=5)
        return " ".join(s.text for s in segs)

    def slice(self, text: str, n: int = 200, o: int = 40):
        ids = self.ek.encode(text, add_special_tokens=False)
        res, s = [], 0
        while s < len(ids):
            e = min(s + n, len(ids))
            res.append(self.ek.decode(ids[s:e], skip_special_tokens=True))
            s += n - o
        return [r for r in res if r.strip()]

    def embed(self, arr):
        return self.e.encode(arr, convert_to_numpy=True, show_progress_bar=False).tolist()

    def upsert(self, group: str, text: str) -> int:
        self.db.ensure(group)
        parts = self.slice(text)
        if not parts:
            return 0
        embs = self.embed(parts)
        self.db.put(group, parts, embs)
        return len(parts)

    def ask(self, q: str, group: str) -> str:
        self.db.ensure(group)
        v = self.e.encode(q, convert_to_numpy=True).tolist()
        arr = self.db.fetch(group, v, 30)
        if not arr:
            return "Нет данных"
        sc = self.r.predict([(q, x) for x in arr])
        qv = float(sum(sc) / len(sc))
        use = [chunk for chunk, s_ in zip(arr, sc) if s_ >= qv][:10]
        return self.reply(q, use)

    def reply(self, prompt: str, parts) -> str:
        prompt2 = f"<|user|>{prompt}\nКонтекст:\n" + "\n".join(parts) + "\n<|assistant|>"
        t_ = self.gtok(prompt2, return_tensors="pt")
        t_ = {k: v.to(self.dev) for k, v in t_.items()}
        with torch.no_grad():
            res = self.g.generate(
                **t_,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.gtok.eos_token_id
            )
        txt = self.gtok.decode(res[0], skip_special_tokens=True)
        if "<|assistant|>" in txt:
            return txt.split("<|assistant|>")[-1].strip()
        return txt.strip()

pipe = Pipe()
