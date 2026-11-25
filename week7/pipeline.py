import json
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI()
ROOT = Path(__file__).resolve().parent.parent
NOT_FOUND = "규정에서 찾을 수 없습니다."
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40, separators=["\n## ", "\n\n", "\n"])

def embed(t): return [d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=t).data]

def load_docs():
    return [{"id": f"{p.stem}-{i}", "text": c, "source": p.stem}
            for p in sorted((ROOT / "data/policies").glob("*.md"))
            for i, c in enumerate(splitter.split_text(p.read_text(encoding="utf-8")))]

DOCS = load_docs()
col = chromadb.PersistentClient(str(ROOT / "chroma_db")).get_or_create_collection("h1", metadata={"hnsw:space": "cosine"})
if col.count() != len(DOCS):
    col.upsert(ids=[d["id"] for d in DOCS], documents=[d["text"] for d in DOCS], embeddings=embed([d["text"] for d in DOCS]))

def vector_search(q, k=3):
    r = col.query(query_embeddings=embed([q]), n_results=k)
    return list(zip(r["ids"][0], r["distances"][0], r["documents"][0]))

def answer(q, search=vector_search, max_dist=0.55):
    hits = [h for h in search(q) if h[1] <= max_dist]
    if not hits:
        return NOT_FOUND, []
    ctx = "\n".join(f"[{i}] {h[2]}" for i, h in enumerate(hits, 1))
    a = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
        {"role": "system", "content": f"참고 문서만 근거로 답하고 [번호]를 단다. 없으면 '{NOT_FOUND}'"},
        {"role": "user", "content": f"<문서>\n{ctx}\n</문서>\n질문: {q}"}]).choices[0].message.content
    return a, [h[0] for h in hits]

if __name__ == "__main__":
    print(*answer("경비규정 제4조가 뭐야?"), sep="\n")
