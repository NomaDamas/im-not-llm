"""Dense text embedder: speed, memory and retrieval quality of one `inl serve` engine per process.

    dense.py <label> <model> <device> [dtype]

Same data and memory sampling as sparse.py (Ko-StrategyQA dev, <work>/kostrategyqa.json). Texts go to
the TEI `POST /embed` (which takes `truncate`) with Qwen3-Embedding's retrieval instruction on queries. Reports memory after load
(`loaded_mb`) and at peak. Appends one JSON line per run to results_dense.jsonl next to this file.
"""
import json, math, os, re, subprocess, sys, threading, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INL = os.environ.get("INL", "/Volumes/SSD1/ios-dev/cargo-target/release/inl")
WORK = os.environ.get("WORK", "/Volumes/SSD1/ane-work")
label, model, device = sys.argv[1:4]
dtype = sys.argv[4] if len(sys.argv) > 4 else None
data = json.load(open(f"{WORK}/kostrategyqa.json"))
INSTRUCT = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
PROBE = INSTRUCT + "훈민정음은 언제 만들어졌나요?"


def footprint_mb(pid):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)", out)
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)] if m else None


if sys.platform == "darwin":
    memory_mb, INTERVAL = footprint_mb, 0.5
else:
    sys.path.insert(0, HERE)
    from gpumem import gpu_mb as memory_mb, INTERVAL


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.halt = pid, 0.0, False

    def run(self):
        while not self.halt:
            f = memory_mb(self.pid)
            if f:
                self.peak = max(self.peak, f)
            time.sleep(INTERVAL)


port = 18452
args = [INL, "serve", "--model", model, "--port", str(port), "--idle-timeout", "0"]
if device != "default":
    args += ["--device", device]
if dtype:
    args += ["--dtype", dtype]
p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sampler = Sampler(p.pid)
sampler.start()
url = f"http://127.0.0.1:{port}"


def emb(texts):
    body = json.dumps({"inputs": texts, "truncate": True}).encode()
    req = urllib.request.Request(url + "/embed", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return np.asarray(json.load(r), np.float32)


def bulk(texts, bs=32):
    return np.concatenate([emb(texts[i:i + bs]) for i in range(0, len(texts), bs)])


t0 = time.perf_counter()
while True:
    try:
        urllib.request.urlopen(url + "/health", timeout=1)
        break
    except Exception:
        if p.poll() is not None:
            sys.exit(f"server exited {p.returncode}")
        time.sleep(0.2)
emb([PROBE])
res = {"engine": label, "load_s": time.perf_counter() - t0}
time.sleep(1)
res["loaded_mb"] = memory_mb(p.pid)
health = json.load(urllib.request.urlopen(url + "/health"))
res["device"], res["dtype"] = health.get("device"), health.get("dtype")
for _ in range(3):
    emb([PROBE])
ts = []
for _ in range(30):
    t = time.perf_counter(); emb([PROBE]); ts.append(time.perf_counter() - t)
res["single_ms"] = 1000 * float(np.median(ts))
t = time.perf_counter(); qv = bulk([INSTRUCT + q[1] for q in data["queries"]]); res["queries_s"] = time.perf_counter() - t
t = time.perf_counter(); dv = bulk([d[1] for d in data["docs"]]); res["docs_s"] = time.perf_counter() - t
time.sleep(2)
res["idle_after_mb"] = memory_mb(p.pid)
res["footprint_peak_mb"] = sampler.peak
p.terminate()
p.wait()
sampler.halt = True

ids = [d[0] for d in data["docs"]]
qn = qv / np.linalg.norm(qv, axis=1, keepdims=True)
dn = dv / np.linalg.norm(dv, axis=1, keepdims=True)
scores = qn @ dn.T
ndcg, recall = [], []
for k, (qid, _) in enumerate(data["queries"]):
    rank = np.argsort(-scores[k], kind="stable")[:100]
    rel = data["qrels"][qid]
    dcg = sum(rel.get(ids[j], 0) / math.log2(i + 2) for i, j in enumerate(rank[:10]))
    ideal = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:10]))
    ndcg.append(dcg / ideal)
    recall.append(sum(1 for j in rank if ids[j] in rel) / len(rel))
res.update({"ndcg@10": float(np.mean(ndcg)), "recall@100": float(np.mean(recall)),
            "n_docs": len(dv), "n_queries": len(qv)})
print(json.dumps(res), flush=True)
open(f"{HERE}/results_dense.jsonl", "a").write(json.dumps(res) + "\n")
