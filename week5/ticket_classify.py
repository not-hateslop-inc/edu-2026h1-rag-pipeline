import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
df = pd.read_csv("week5/data/tickets.csv")
LABELS = "장비, 계정, 네트워크"

ZERO = f"IT 문의를 [{LABELS}] 중 하나로 분류한다. 라벨만 답한다."
FEW = ZERO + "\n예시:\n- 회의실 TV 화면 공유가 안 돼요 → 장비\n- 2단계 인증 기기를 바꿨어요 → 계정\n- 게스트 와이파이 알려 주세요 → 네트워크"

for name, system in [("zero-shot", ZERO), ("few-shot", FEW)]:
    pred = [client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": t}]).choices[0].message.content.strip()
        for t in df.text]
    acc = (pd.Series(pred) == df.label).mean()
    print(f"{name:<10} {acc:.2f}")
