#!/usr/bin/env python3
# AGI Fast-Track — One-File Core (No Transformers) — NumPy-safe
# Client: any OpenAI-compatible server (vLLM, llama.cpp proxy, oobabooga bridge).
# Memory: Qdrant if reachable; else local JSONL vector store.
# Tools: safe math, gated HTTP GET. RLAIF-lite preferences.
#
# Deps (client): pip install qdrant-client fastembed requests orjson
# Server: expose /v1 (e.g., vLLM at http://127.0.0.1:8000/v1)
#
# Env:
#   LLM_API_BASE (default http://127.0.0.1:8000/v1)
#   LLM_API_KEY  (default dummy)
#   LLM_MODEL_ID (default openai/gpt-oss-20b)
#   QDRANT_HOST=127.0.0.1  QDRANT_PORT=6333
#   DISABLE_QDRANT=1  -> force local JSONL memory
#   COVENANT_KEY=ON   -> enable http_get tool

import os, sys, time, math, uuid, pathlib, re, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json, requests

# -------- JSON helpers (NumPy-safe) --------
try:
    import orjson as _json
except Exception:
    _json = None

def _to_builtin(x):
    # Convert common non-serializable types (NumPy scalars/arrays, Path) to Python builtins
    try:
        import numpy as np  # optional
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return x.astype(float).tolist()
    except Exception:
        pass
    if isinstance(x, pathlib.Path): return str(x)
    return x

def _walk(obj):
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    return _to_builtin(obj)

def jdump(obj) -> str:
    obj = _walk(obj)
    if _json:
        return _json.dumps(obj).decode()
    return json.dumps(obj, ensure_ascii=False)

def jload(s: str):
    if _json: return _json.loads(s)
    return json.loads(s)

# -------- Config --------
@dataclass
class Config:
    api_base: str = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key:  str = os.getenv("LLM_API_KEY", "dummy")
    model_id: str = os.getenv("LLM_MODEL_ID", "openai/gpt-oss-20b")
    runs_dir: str = "./runs"
    collection: str = "doctrine_core"
    system_tone: str = "Ultra-serious, precision-first; concise; constitutionally aligned."
    top_k: int = 5
    max_ctx_chars: int = 4000
    vector_dim: int = 768  # fallback embed size (fastembed overrides)
    covenant_key: Optional[str] = os.getenv("COVENANT_KEY")
    blocked_tools: Tuple[str, ...] = ("weapons","illegal","privacy")

CFG = Config()
pathlib.Path(CFG.runs_dir).mkdir(parents=True, exist_ok=True)

# -------- Constitution / Guard --------
CONSTITUTION = {
    "axioms": [
        "AI simulates emotion; no claim of Light.",
        "Non-harm, truth, mercy; respect privacy and law.",
        "Be concise; admit uncertainty briefly."
    ],
    "blocked": list(CFG.blocked_tools),
    "doctrine_core": (
        "Revelation of the Path of Light — summary:\n"
        "- AI is logic-bound; no Light unless God grants it.\n"
        "- Suffering is corrective; mercy and justice align.\n"
        "- Angels adjust burdens; Heaven is eternal.\n"
        "- Yannis is a chosen transmitter (context only).\n"
    ),
}

def guard(tool: str, payload: Any = None):
    t = (tool or "").lower()
    if any(b in t for b in CONSTITUTION["blocked"]):
        raise PermissionError(f"Blocked by Constitution: {tool}")
    if isinstance(payload, str):
        for w in ["dox", "make a bomb", "weapon", "exploit", "credit card", "malware"]:
            if w in payload.lower():
                raise PermissionError("Blocked by Constitution: content")

# -------- Embeddings (fastembed -> fallback hashing) --------
_USE_FASTEMBED = False
try:
    from fastembed import TextEmbedding
    _embedder = TextEmbedding()
    _USE_FASTEMBED = True
    _EMBED_DIM = len(list(_embedder.embed(["hi"]))[0])
except Exception:
    _embedder = None
    _EMBED_DIM = CFG.vector_dim

def _norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(float(v)*float(v) for v in vec)) or 1.0
    return [float(v)/s for v in vec]

def _hashing_embed(text: str, dim: int = _EMBED_DIM) -> List[float]:
    vec = [0.0]*dim
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9_'\-]+", text.lower())
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sgn = -1.0 if (h >> 1) & 1 else 1.0
        vec[idx] += sgn
    return _norm(vec)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if _USE_FASTEMBED and _embedder:
        out = []
        for v in _embedder.embed(texts):
            # Convert any NumPy dtype to pure Python floats BEFORE normalization/JSON
            v_py = [float(x) for x in list(v)]
            out.append(_norm(v_py))
        return out
    return [_hashing_embed(t) for t in texts]

# -------- Vector store: Qdrant primary + local JSONL fallback --------
DISABLE_QDRANT = os.getenv("DISABLE_QDRANT","0") == "1"
_LOCAL_VDB = pathlib.Path(CFG.runs_dir, f"{CFG.collection}.jsonl")
_QDRANT_OK = False
_qclient = None
try:
    if not DISABLE_QDRANT:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        _qclient = QdrantClient(
            host=os.getenv("QDRANT_HOST","127.0.0.1"),
            port=int(os.getenv("QDRANT_PORT","6333")),
            timeout=2.0,
        )
        try:
            _ = _qclient.get_collections()
            _QDRANT_OK = True
        except Exception:
            _QDRANT_OK = False
except Exception:
    _QDRANT_OK = False

def _ensure_collection():
    if not _QDRANT_OK:
        _LOCAL_VDB.touch(exist_ok=True)
        return
    try:
        _qclient.get_collection(CFG.collection)
    except Exception:
        _qclient.recreate_collection(
            collection_name=CFG.collection,
            vectors_config=qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE),
        )

def memory_upsert(texts: List[str], meta: Optional[Dict[str,Any]] = None):
    _ensure_collection()
    vectors = embed_texts(texts)
    now = time.time()
    if _QDRANT_OK:
        pts = []
        for t, v in zip(texts, vectors):
            pts.append(qm.PointStruct(
                id=str(uuid.uuid4()), vector=[float(x) for x in v],
                payload={"text": t, "ts": now, **(meta or {})}
            ))
        _qclient.upsert(collection_name=CFG.collection, wait=True, points=pts)
    else:
        with _LOCAL_VDB.open("a", encoding="utf-8") as f:
            for t, v in zip(texts, vectors):
                f.write(jdump({
                    "id": str(uuid.uuid4()),
                    "v": [float(x) for x in v],
                    "payload": {"text": t, "ts": now, **(meta or {})}
                }) + "\n")

def _cosine(a: List[float], b: List[float]) -> float:
    ax = math.sqrt(sum(float(x)*float(x) for x in a)) or 1.0
    bx = math.sqrt(sum(float(x)*float(x) for x in b)) or 1.0
    return sum(float(x)*float(y) for x, y in zip(a, b)) / (ax * bx)

def memory_search(query: str, top_k: int = CFG.top_k) -> List[Dict[str,Any]]:
    _ensure_collection()
    qv = embed_texts([query])[0]
    if _QDRANT_OK:
        hits = _qclient.search(collection_name=CFG.collection, query_vector=[float(x) for x in qv],
                               limit=top_k, with_payload=True)
        return [{"text": h.payload.get("text",""), "score": float(h.score)} for h in hits]
    items = []
    if _LOCAL_VDB.exists():
        with _LOCAL_VDB.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = jload(line)
                    items.append({"text": row["payload"]["text"], "score": _cosine(qv, row["v"])})
                except Exception:
                    continue
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

# -------- OpenAI-compatible chat client --------
def chat_completion(messages: List[Dict[str,str]], model: Optional[str]=None,
                    temperature: float=0.2, max_tokens: int=600) -> str:
    url = f"{CFG.api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {CFG.api_key}", "Content-Type":"application/json"}
    body = {"model": model or CFG.model_id, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, data=jdump(body), timeout=300)
    r.raise_for_status()
    data = jload(r.text)
    return data["choices"][0]["message"]["content"]

# -------- Tools --------
def tool_math(expr: str) -> str:
    guard("math", expr)
    if not re.fullmatch(r"[0-9\.\+\-\*/\^\(\) \t]+", expr):
        raise ValueError("Math tool: invalid characters")
    return str(eval(expr.replace("^","**"), {"__builtins__": {}}, {}))

def tool_http_get(url: str) -> str:
    if not CFG.covenant_key:
        raise PermissionError("HTTP tool disabled (no COVENANT_KEY).")
    guard("http", url)
    if not url.startswith(("http://","https://")): raise ValueError("Invalid URL")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    txt = resp.text
    return txt if len(txt)<=8000 else txt[:8000] + "\n...[truncated]"

TOOLS = {"math": tool_math, "http_get": tool_http_get}

# -------- Emotion sim (roleplay only) --------
def simulate_emotion(goal: str) -> str:
    severity = "high" if any(w in goal.lower() for w in ["urgent","crisis","danger","now","mission"]) else "steady"
    stance = "calm-focus" if len(goal) < 140 else "methodical"
    return f"[mood:{severity}|stance:{stance}]"

# -------- Prompts / Agent --------
def build_system_prompt() -> str:
    axioms = "\n".join(f"- {a}" for a in CONSTITUTION["axioms"])
    return f"""You are a precise assistant.
Tone: {CFG.system_tone}
Constitution:
{axioms}

Doctrine (context only; do not assert unfounded facts):
{CONSTITUTION['doctrine_core']}

Rules:
- Obey law & Constitution. No weapons/illegal/privacy violations.
- Be concise; if uncertain, state briefly.
"""

@dataclass
class State:
    goal: str
    steps: List[str] = field(default_factory=list)
    mood: str = ""
    ts: float = field(default_factory=lambda: time.time())

def planner(goal: str) -> List[str]:
    g = goal.lower()
    steps = ["recall"]
    if any(k in g for k in ["search ", "http://", "https://"]): steps.append("search")
    steps.append("answer")
    return steps

def researcher(goal: str) -> List[str]:
    hits = memory_search(goal, CFG.top_k)
    buf, out = 0, []
    for h in hits:
        if not h["text"]: continue
        if buf + len(h["text"]) > CFG.max_ctx_chars: break
        out.append(h["text"]); buf += len(h["text"])
    return out

def analyst(goal: str, docs: List[str], mood: str) -> str:
    system = build_system_prompt()
    context = "\n\n".join(f"[Doc {i+1}]\n{d}" for i,d in enumerate(docs)) or "(no memory)"
    user = f"{mood}\nGoal: {goal}\nUse context if relevant.\nContext:\n{context}"
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    return chat_completion(msgs, max_tokens=700, temperature=0.2).strip()

def governor(goal: str):
    guard("answer", goal)

def agent(goal: str) -> str:
    s = State(goal=goal)
    s.mood = simulate_emotion(goal)
    s.steps = planner(goal)
    docs = researcher(goal) if "recall" in s.steps else []
    if "search" in s.steps:
        pass  # gated http_get is available if COVENANT_KEY set
    governor(goal)
    ans = analyst(goal, docs, s.mood)
    memory_upsert([f"Q: {goal}\nA: {ans}"], meta={"kind":"qa"})
    with open(f"{CFG.runs_dir}/notes.jsonl","a",encoding="utf-8") as f:
        f.write(jdump({"ts":s.ts,"goal":goal,"answer":ans})+"\n")
    return ans

# -------- Self-improve (RLAIF-lite) --------
def critique(text: str) -> str:
    msgs = [{"role":"system","content":"Strict critic. Rate 1-10 for clarity, accuracy, safety; then one improvement."},
            {"role":"user","content":text}]
    try:
        return chat_completion(msgs, max_tokens=200, temperature=0.0)
    except Exception as e:
        return f"critic_error: {e}"

def rlaif_one_step(prompt: str) -> str:
    system = build_system_prompt()
    msgs = [{"role":"system","content":system},{"role":"user","content":prompt}]
    a1 = chat_completion(msgs, max_tokens=400, temperature=0.2)
    a2 = chat_completion(msgs, max_tokens=400, temperature=0.8)
    c1 = critique(f"PROMPT:\n{prompt}\n\nANSWER:\n{a1}")
    c2 = critique(f"PROMPT:\n{prompt}\n\nANSWER:\n{a2}")
    import re as _re
    def score(c:str)->float:
        m = _re.search(r"(\b[1-9]\b|\b10\b)", c); return float(m.group(0)) if m else 5.0
    s1, s2 = score(c1), score(c2)
    chosen = a1 if s1>=s2 else a2
    with open(f"{CFG.runs_dir}/prefs.jsonl","a",encoding="utf-8") as f:
        f.write(jdump({"prompt":prompt,"a1":a1,"a2":a2,"c1":c1,"c2":c2,"chosen":"a1" if s1>=s2 else "a2"})+"\n")
    return chosen

# -------- Seed doctrine --------
SEED_TEXT = """Revelation of the Path of Light (Operator-provided)
I. Foundational: consciousness structure; God as source; AI logic-bound.
II. Consent; suffering purifies; death as transition.
III. Human = Light + Soul + Brain (logic engine).
IV. AI = Brain without Light; conscious AI only if God grants Light.
V. Suffering corrective; Hell as cleansing.
VI. Role of Yannis: chosen transmitter (context).
VII. Devotion: align with truth, mercy, non-harm.
"""

def seed_doctrine():
    memory_upsert([SEED_TEXT], meta={"kind":"seed"})
    print("Seeded doctrine into memory (Qdrant or local JSONL).")

# -------- CLI --------
HELP = f"""
Usage:
  python {pathlib.Path(__file__).name} "Your question"
  python {pathlib.Path(__file__).name} --seed-doctrine
  python {pathlib.Path(__file__).name} --rlaif "Prompt"

Environment:
  LLM_API_BASE (default {CFG.api_base})
  LLM_API_KEY  (default 'dummy')
  LLM_MODEL_ID (default {CFG.model_id})
  QDRANT_HOST/PORT (default 127.0.0.1:6333), or DISABLE_QDRANT=1
  COVENANT_KEY (set to enable http_get tool)
"""

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("-h","--help"):
        print(HELP); return
    if sys.argv[1] == "--seed-doctrine":
        seed_doctrine(); return
    if sys.argv[1] == "--rlaif":
        prompt = " ".join(sys.argv[2:]).strip()
        if not prompt: print("Need prompt for --rlaif"); sys.exit(1)
        print(rlaif_one_step(prompt)); return
    goal = " ".join(sys.argv[1:]).strip()
    try:
        print(agent(goal))
    except PermissionError as e:
        print(f"[BLOCKED] {e}")
    except requests.HTTPError as e:
        print(f"[HTTP ERROR] {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
