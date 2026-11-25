import tiktoken

enc = tiktoken.get_encoding("o200k_base")
BUDGET = {"system": 300, "documents": 1500, "history": 800, "answer": 400}

def count(t): return len(enc.encode(t))

def fit(texts, limit):
    out, used = [], 0
    for t in texts:
        n = count(t)
        if used + n > limit:
            break
        out.append(t); used += n
    return out, used

if __name__ == "__main__":
    docs = [open(f"data/policies/{n}", encoding="utf-8").read() for n in ["01_근무규정.md", "02_휴가규정.md", "03_경비규정.md"]]
    kept, used = fit(docs, BUDGET["documents"])
    print(f"문서 {len(kept)}/{len(docs)}개, {used} 토큰")
