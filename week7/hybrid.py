# (선택 실습) 조항 번호 질문은 Vector 검색이 약합니다. BM25를 섞어 봅니다.
from rank_bm25 import BM25Okapi
from pipeline import DOCS, vector_search

def bigrams(t):
    t = t.replace(" ", "")
    return [t[i:i + 2] for i in range(len(t) - 1)]

bm25 = BM25Okapi([bigrams(d["text"]) for d in DOCS])
TEXT = {d["id"]: d["text"] for d in DOCS}

def hybrid_search(q, k=3, rrf_k=60):
    s = bm25.get_scores(bigrams(q))
    bm = [DOCS[i]["id"] for i in sorted(range(len(DOCS)), key=lambda i: -s[i])[:10]]
    vec = [h[0] for h in vector_search(q, 10)]
    score = {}
    for ranking in (bm, vec):
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0) + 1 / (rrf_k + r)
    top = sorted(score, key=lambda d: -score[d])[:k]
    return [(d, 0.0, TEXT[d]) for d in top]     # 거리 대신 0.0 (거절 판단은 vector 기준으로 따로)
