"""KURE-Sparse-nano: speed, memory and retrieval quality of one `inl serve` engine per process.

    sparse.py <label> <model or package dir> <device>

Needs <work>/kostrategyqa.json ({"docs": [[id, text]], "queries": [[id, text]], "qrels": {qid: {cid: rel}}},
Ko-StrategyQA dev from taeminlee/Ko-StrategyQA) and `inl` built in release. Queries are sent with
input_type "query" ("[Q] "), passages with "passage" ("[D] "). Appends one JSON line per run to
results_sparse.jsonl next to this file.
"""
import json, math, os, re, subprocess, sys, threading, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INL = os.environ.get("INL", "/Volumes/SSD1/ios-dev/cargo-target/release/inl")
WORK = os.environ.get("WORK", "/Volumes/SSD1/ane-work")
label, model, device = sys.argv[1:4]
data = json.load(open(f"{WORK}/kostrategyqa.json"))
PROBE = "훈민정음은 언제 만들어졌나요?"


def footprint_mb(pid):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)", out)
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)] if m else None


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.halt = pid, 0.0, False

    def run(self):
        while not self.halt:
            f = footprint_mb(self.pid)
            if f:
                self.peak = max(self.peak, f)
            time.sleep(0.5)


port = 18451
args = [INL, "serve", "--model", model, "--port", str(port), "--idle-timeout", "0", "--sparse"]
if device != "default":
    args += ["--device", device]
p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sampler = Sampler(p.pid)
sampler.start()
url = f"http://127.0.0.1:{port}"


def encode(texts, kind):
    body = json.dumps({"inputs": texts, "input_type": kind, "truncate": True}).encode()
    req = urllib.request.Request(url + "/embed_sparse", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return [{e["index"]: e["value"] for e in row} for row in json.load(r)]


def bulk(texts, kind, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        out += encode(texts[i:i + bs], kind)
    return out


t0 = time.perf_counter()
while True:
    try:
        urllib.request.urlopen(url + "/health", timeout=1)
        break
    except Exception:
        if p.poll() is not None:
            sys.exit(f"server exited {p.returncode}")
        time.sleep(0.2)
encode([PROBE], "query")
res = {"engine": label, "load_s": time.perf_counter() - t0}
health = json.load(urllib.request.urlopen(url + "/health"))
res["device"], res["dtype"] = health.get("device"), health.get("dtype")
for _ in range(3):
    encode([PROBE], "query")
ts = []
for _ in range(30):
    t = time.perf_counter(); encode([PROBE], "query"); ts.append(time.perf_counter() - t)
res["single_ms"] = 1000 * float(np.median(ts))
t = time.perf_counter(); qv = bulk([q[1] for q in data["queries"]], "query"); res["queries_s"] = time.perf_counter() - t
t = time.perf_counter(); dv = bulk([d[1] for d in data["docs"]], "passage"); res["docs_s"] = time.perf_counter() - t
res["footprint_peak_mb"] = sampler.peak
p.terminate()
p.wait()
sampler.halt = True

# Inverted index scoring, then nDCG@10 / Recall@100 over the qrels.
postings = {}
for j, v in enumerate(dv):
    for term, w in v.items():
        postings.setdefault(term, []).append((j, w))
postings = {t: (np.array([j for j, _ in ps]), np.array([w for _, w in ps], np.float32)) for t, ps in postings.items()}
ids = [d[0] for d in data["docs"]]
ndcg, recall = [], []
for (qid, _), v in zip(data["queries"], qv):
    s = np.zeros(len(ids), np.float32)
    for term, w in v.items():
        if term in postings:
            js, ws = postings[term]
            s[js] += w * ws
    rank = np.argsort(-s, kind="stable")[:100]
    rel = data["qrels"][qid]
    dcg = sum(rel.get(ids[j], 0) / math.log2(i + 2) for i, j in enumerate(rank[:10]))
    ideal = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:10]))
    ndcg.append(dcg / ideal)
    recall.append(sum(1 for j in rank if ids[j] in rel) / len(rel))
res.update({"ndcg@10": float(np.mean(ndcg)), "recall@100": float(np.mean(recall)),
            "doc_nnz": float(np.mean([len(v) for v in dv])), "query_nnz": float(np.mean([len(v) for v in qv])),
            "n_docs": len(dv), "n_queries": len(qv)})
print(json.dumps(res), flush=True)
open(f"{HERE}/results_sparse.jsonl", "a").write(json.dumps(res) + "\n")
