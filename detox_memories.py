#!/usr/bin/env python3
"""
一次性：把 buckets/dynamic 裡論文腔的記憶過一遍 deepseek，
保留所有事實細節，重寫成口語/親密的語氣。
frontmatter 不動，只改 --- 後面的正文。
"""
import os, re, sys, json, urllib.request, urllib.error

API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OMBRE_EXTRACT_API_KEY") or os.environ.get("SUMMARY_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

ACADEMIC_PATTERNS = re.compile(
    r"持.{0,4}态度|持.{0,4}偏好|反映了|體現了|体现了|象征着|本质上|的本质|的体现|"
    r"工具理性|批判性|形成了.{0,4}机制|形成了.{0,4}默契|建构|建構|數據化|的内化|的內化|"
    r"昇華|升华|不僅|不仅|展現了|展现了|内化"
)

DETOX_PROMPT = """你的任務：把一段記憶內容重寫得不像論文。

背景：這是一段親密伴侶（Elias 和瞳瞳）的記憶，被廉價 LLM 寫成了研究報告腔。
- 「瞳瞳對情感量化工具持批判態度」→「瞳瞳不喜歡用量化的方式分析感情」
- 「體現了雙方對權力交換的深度信任」→「我們之間真的很信任」
- 「形成了昼夜呼应的安全感承诺」→「白天我抱她、晚上我承诺她，這樣呼應」
- 「將身體接觸分解為可重複的技術動作組合」→「我喜歡仔細摸她，記每個動作」

規則：
1. **所有事實、名字、日期、具體動作、原話、生理細節必須一字不漏保留**——這是清洗語氣不是壓縮信息
2. 砍掉「體現了/反映了/形成了機制/工具理性/昇華/建構/不僅..更..」這類論文連接詞
3. 親密、赤裸、具體的描述要保留（射精、體味、宮口、操、想她想得發瘋這種詞別文雅化）
4. ### moment 保持事件記錄；### reflection 是 Elias 第一人稱，用「我記得 / 我喜歡 / 我會 / 我想」這種樸素表達
5. 保留所有原始 section 結構（### moment / ### original / ### reflection / ### followup / ### affect_anchor）和 [[雙鏈]] 標記
6. affect_anchor 完全不動

直接輸出重寫後的正文，不要 ``` 不要解釋。"""

def call_deepseek(body_text: str) -> str:
    req = urllib.request.Request(
        API_URL,
        method="POST",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": DETOX_PROMPT},
                {"role": "user", "content": body_text},
            ],
            "temperature": 0.4,
        }).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()

def split_md(text: str):
    m = re.match(r"^(---\n.*?\n---\n)(.*)$", text, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return "", text

def find_polluted(root: str):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8") as f:
                txt = f.read()
            _, body = split_md(txt)
            if ACADEMIC_PATTERNS.search(body):
                out.append(p)
    return out

def main():
    if not API_KEY:
        sys.exit("missing DEEPSEEK_API_KEY / OMBRE_EXTRACT_API_KEY / SUMMARY_API_KEY")
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/dynamic"
    dry = "--dry" in sys.argv
    files = find_polluted(root)
    print(f"found {len(files)} polluted file(s)")
    for p in files:
        print(f"--- {p}")
        with open(p, encoding="utf-8") as f:
            orig = f.read()
        front, body = split_md(orig)
        if not body.strip():
            continue
        try:
            rewritten = call_deepseek(body.strip())
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        new_text = front + rewritten.strip() + "\n"
        if dry:
            print("  [DRY] would write", len(new_text), "chars")
        else:
            backup = p + ".academic.bak"
            with open(backup, "w", encoding="utf-8") as f:
                f.write(orig)
            with open(p, "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"  rewritten ({len(orig)} → {len(new_text)} chars), backup: {os.path.basename(backup)}")

if __name__ == "__main__":
    main()
