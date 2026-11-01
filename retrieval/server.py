"""Dense retrieval server for OpenResearcher (training) and BrowseComp-Plus (eval).

Encoder: Qwen3-Embedding-8B (last-token pooling, cosine). Both corpora use the
precomputed Tevatron / OpenResearcher qwen3-embedding-8b indexes.

  POST /search {"corpus": "openresearcher"|"bcp", "query": str, "k": 10}
       -> {"results": [{"docid", "url", "score", "snippet"}]}   (snippet = first 512 tokens)
  POST /open   {"corpus": ..., "docid": str}
       -> {"docid", "url", "text"}                               (text capped at 8192 tokens)
  GET  /health
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import glob
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrieval")

HF = HF_HUB_CACHE
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
SNIPPET_TOKENS = 512
DOC_TOKENS = 8192


def snapshot(repo: str) -> str:
    paths = sorted(glob.glob(f"{HF}/{repo}/snapshots/*"))
    if not paths:
        raise FileNotFoundError(f"{repo} not in {HF}; download it first (see README)")
    return paths[-1]


class Corpus:
    """Doc text store + embedding matrix (rows aligned with ``docids``)."""

    def __init__(self, name: str, parquet_files: list[str], shards: list[torch.Tensor], docids: list[str]):
        self.name = name
        self.shards = shards  # row blocks of the normalized index, possibly on different devices
        self.docids = docids
        t0 = time.time()
        self.text, self.url, self.row = [], [], {}
        for f in parquet_files:
            t = pq.read_table(f, columns=["docid", "text", "url"])
            base = len(self.text)
            self.row.update({str(d): base + i for i, d in enumerate(t.column("docid").to_pylist())})
            self.text.extend(t.column("text").to_pylist())
            self.url.extend(t.column("url").to_pylist())
        log.info(
            "[%s] %d docs loaded in %.0fs; index %d rows on %s",
            name,
            len(self.text),
            time.time() - t0,
            sum(len(x) for x in shards),
            [str(x.device) for x in shards],
        )

    def doc(self, docid: str) -> tuple[str, str] | None:
        i = self.row.get(str(docid))
        return None if i is None else (self.text[i] or "", self.url[i] or "")


class Retriever:
    def __init__(self, encoder_device: str, or_devices: list[str], names: set[str]):
        model_dir = snapshot("models--Qwen--Qwen3-Embedding-8B")
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self._local = threading.local()  # HF fast tokenizers are not thread-safe
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(encoder_device).eval()
        self.device = encoder_device
        self.corpora: dict[str, Corpus] = {}
        if "bcp" in names:
            self.corpora["bcp"] = self._load_bcp()
        if "openresearcher" in names:
            self.corpora["openresearcher"] = self._load_openresearcher(or_devices)

    def _load_bcp(self) -> Corpus:
        idx_dir = snapshot("datasets--Tevatron--browsecomp-plus-indexes") + "/qwen3-embedding-8b"
        embs, ids = [], []
        for f in sorted(glob.glob(f"{idx_dir}/corpus.shard*_of_4.pkl")):
            with open(f, "rb") as fh:
                e, lookup = pickle.load(fh)
            embs.append(torch.from_numpy(np.asarray(e, dtype=np.float32)))
            ids.extend(str(x) for x in lookup)
        emb = F.normalize(torch.cat(embs), dim=-1).to(self.device, torch.bfloat16)
        files = sorted(glob.glob(snapshot("datasets--Tevatron--browsecomp-plus-corpus") + "/data/*.parquet"))
        return Corpus("bcp", files, [emb], ids)

    def _load_openresearcher(self, devices: list[str]) -> Corpus:
        t0 = time.time()
        idx = sorted(
            glob.glob(
                snapshot("datasets--OpenResearcher--OpenResearcher-Indexes") + "/qwen3-embedding-8b/**/*.pkl",
                recursive=True,
            )
        )
        with open(idx[0], "rb") as fh:
            e, lookup = pickle.load(fh)
        log.info("[openresearcher] index pickle %s loaded in %.0fs", e.shape, time.time() - t0)
        # fp16 -> normalized bf16 in chunks (CPU bf16 matmul uses AMX; CPU fp16 matmul is slow).
        e = torch.from_numpy(e)
        bounds = np.linspace(0, e.shape[0], len(devices) + 1).astype(int)
        shards, step = [], 1_000_000
        for dev, lo, hi in zip(devices, bounds[:-1], bounds[1:]):
            emb = torch.empty((hi - lo, e.shape[1]), dtype=torch.bfloat16, device=dev)
            for s in range(lo, hi, step):
                t = min(s + step, hi)
                emb[s - lo : t - lo] = F.normalize(e[s:t].float(), dim=-1).to(dev, torch.bfloat16)
            shards.append(emb)
        del e
        files = sorted(
            glob.glob(snapshot("datasets--OpenResearcher--OpenResearcher-Corpus") + "/**/*.parquet", recursive=True)
        )
        return Corpus("openresearcher", files, shards, [str(x) for x in lookup])

    @property
    def thread_tok(self):
        if not hasattr(self._local, "tok"):
            self._local.tok = copy.deepcopy(self.tok)
        return self._local.tok

    @torch.inference_mode()
    def encode(self, queries: list[str]) -> torch.Tensor:
        batch = self.thread_tok(
            [f"{QUERY_PREFIX}{q}" for q in queries], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state[:, -1]  # left padding -> last token
        return F.normalize(hidden.float(), dim=-1)

    @torch.inference_mode()
    def search(self, corpus: str, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        c = self.corpora[corpus]
        q = self.encode(queries)
        vals, idxs, offset = [], [], 0
        for shard in c.shards:
            v, i = torch.topk(q.to(shard.device, shard.dtype) @ shard.T, k, dim=-1)
            vals.append(v.float().cpu())
            idxs.append(i.cpu() + offset)
            offset += len(shard)
        v, j = torch.topk(torch.cat(vals, dim=1), k, dim=-1)
        i = torch.gather(torch.cat(idxs, dim=1), 1, j)
        return [[(c.docids[d], sc) for sc, d in zip(srow, irow)] for srow, irow in zip(v.tolist(), i.tolist())]

    def truncate(self, text: str, n_tokens: int) -> str:
        head = text[: n_tokens * 8]
        ids = self.thread_tok(head, add_special_tokens=False)["input_ids"]
        if len(ids) <= n_tokens and len(head) == len(text):
            return text
        return self.thread_tok.decode(ids[:n_tokens])


class Batcher:
    """Collect concurrent /search calls per corpus into one encode + matmul."""

    def __init__(
        self, retriever: Retriever, corpus: str, executor: ThreadPoolExecutor, max_batch: int = 64, wait_ms: float = 5
    ):
        self.r, self.corpus, self.ex = retriever, corpus, executor
        self.max_batch, self.wait = max_batch, wait_ms / 1000
        self.q: asyncio.Queue = asyncio.Queue()

    async def submit(self, query: str, k: int):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((query, k, fut))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.q.get()]
            await asyncio.sleep(self.wait)
            while not self.q.empty() and len(items) < self.max_batch:
                items.append(self.q.get_nowait())
            k = max(i[1] for i in items)
            try:
                res = await loop.run_in_executor(self.ex, self.r.search, self.corpus, [i[0] for i in items], k)
                for (_, ki, fut), hits in zip(items, res):
                    fut.set_result(hits[:ki])
            except Exception as e:  # noqa: BLE001
                log.exception("search batch failed")
                for *_, fut in items:
                    fut.set_exception(e)


class SearchReq(BaseModel):
    corpus: str
    query: str
    k: int = 10


class OpenReq(BaseModel):
    corpus: str
    docid: str


def build_app(r: Retriever) -> FastAPI:
    app = FastAPI()
    compute_pool = ThreadPoolExecutor(1)  # one encode+search batch at a time
    cpu_pool = ThreadPoolExecutor(16)  # snippet/doc tokenization
    batchers = {c: Batcher(r, c, compute_pool) for c in r.corpora}

    @app.on_event("startup")
    async def _start():
        for b in batchers.values():
            asyncio.create_task(b.run())

    def corpus_or_404(name: str) -> Corpus:
        if name not in r.corpora:
            raise HTTPException(404, f"unknown corpus {name!r}; loaded: {list(r.corpora)}")
        return r.corpora[name]

    @app.get("/health")
    async def health():
        return {"corpora": {k: len(v.text) for k, v in r.corpora.items()}}

    @app.post("/search")
    async def search(req: SearchReq):
        c = corpus_or_404(req.corpus)
        hits = await batchers[req.corpus].submit(req.query, max(1, min(req.k, 50)))

        def fmt():
            out = []
            for docid, score in hits:
                text, url = c.doc(docid) or ("", "")
                out.append(
                    {"docid": docid, "url": url, "score": round(score, 4), "snippet": r.truncate(text, SNIPPET_TOKENS)}
                )
            return out

        return {"results": await asyncio.get_running_loop().run_in_executor(cpu_pool, fmt)}

    @app.post("/open")
    async def open_doc(req: OpenReq):
        c = corpus_or_404(req.corpus)
        d = c.doc(req.docid)
        if d is None:
            raise HTTPException(404, f"docid {req.docid!r} not found")
        text = await asyncio.get_running_loop().run_in_executor(cpu_pool, r.truncate, d[0], DOC_TOKENS)
        return {"docid": req.docid, "url": d[1], "text": text}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--encoder-device", default="cuda:0")
    ap.add_argument(
        "--or-devices",
        default="cuda:1,cuda:2",
        help="comma-separated devices to shard the 14.9M-doc OpenResearcher index over (122 GB bf16); 'cpu' works but is ~50x slower",
    )
    ap.add_argument("--corpora", default="openresearcher,bcp")
    args = ap.parse_args()
    torch.set_num_threads(min(64, os.cpu_count() or 8))
    retriever = Retriever(args.encoder_device, args.or_devices.split(","), set(args.corpora.split(",")))
    uvicorn.run(build_app(retriever), host=args.host, port=args.port, log_level="warning")
