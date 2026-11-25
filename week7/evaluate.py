import json, sys
from pathlib import Path
from pipeline import answer, vector_search, NOT_FOUND

ROOT = Path(__file__).resolve().parent.parent
QS = json.loads((ROOT / "eval/questions.json").read_text(encoding="utf-8"))
use_hybrid = "--hybrid" in sys.argv
if use_hybrid:
    from hybrid import hybrid_search

retr_ok = ans_ok = 0
for q in QS:
    hits = (hybrid_search if use_hybrid else vector_search)(q["question"])
    exp = q["expected_source"]
    r_ok = exp is None or any(h[0].startswith(exp) for h in hits)
    a, _ = answer(q["question"])
    a_ok = q["answer_contains"] in a
    retr_ok += r_ok; ans_ok += a_ok
    print(f"{q['id']} 검색 {'O' if r_ok else 'X'} 답변 {'O' if a_ok else 'X'}  {q['question']}")
print(f"\n검색 {retr_ok}/{len(QS)}  답변 {ans_ok}/{len(QS)}")
