# -*- coding: utf-8 -*-
"""验证 transcribe 全链路（monkeypatch 上传层，不发真实请求）：

1. prompt 链式传递：块0 只有词表；块1+ 带词表+上一块结尾文本
2. offset 正确加到每段
3. 重叠区去重：同句两份 -> 保留更完整的；截断版被完整版替换
4. 漏识补缺：只在一块有的句子原样保留
5. 相邻句子（重叠 <0.45s）不误伤
6. multipart body 真的含 prompt 字段
7. build_vocab_text / build_prompt 边界行为
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groq_srt as gs

FFMPEG = gs.find_ffmpeg()

# ---------- 单元：词表与 prompt 构建 ----------
v = gs.build_vocab_text(["心理治疗", "心理治疗,认知疗法", "欧文·亚隆"])
assert v == "心理治疗 认知疗法 欧文·亚隆", v
v2 = gs.build_vocab_text("甲，乙;丙 丁")
assert v2 == "甲 乙 丙 丁", v2
assert gs.build_vocab_text("") == "" and gs.build_vocab_text(None) == ""
p = gs.build_prompt("心理治疗 欧文·亚隆", "上一块的结尾文本内容")
assert p.startswith("心理治疗") and p.endswith("上一块的结尾文本内容"), p
# 超长截断：保 tail，砍词表
long_vocab = "词" * 400
p2 = gs.build_prompt(long_vocab, "尾" * 100)
assert len(p2) <= gs.PROMPT_MAX_CHARS, len(p2)
assert p2.endswith("尾" * 100), p2[-20:]
print("[OK] build_vocab_text / build_prompt（含超长截断保 tail）")

# ---------- 端到端：monkeypatch _upload_one ----------
VOCAB = ["心理治疗", "认知疗法"]

# 每块的合成 segments（本地时间）
BLOCKS = [
    # 块0: 全局 = 本地
    [{"start": 1.0, "end": 3.0, "text": "开场第一句。"},
     {"start": 4.0, "end": 6.0, "text": "第二句。"},
     {"start": 10.0, "end": 11.5, "text": "正常句A。"},
     {"start": 13.0, "end": 14.5, "text": "交界句子完整版。"},
     {"start": 17.0, "end": 17.5, "text": "交界句子被截断的半句"}],  # 块0音频止于18，被截断
    # 块1 (offset 12): 本地 + 12
    [{"start": 0.5, "end": 2.5, "text": "交界句子完整版。"},   # 全局 12.5-14.5，与块0 13-14.5 重叠 -> 去重
     {"start": 4.8, "end": 6.2, "text": "交界句子被截断的半句"},  # 全局 16.8-18.2：完整版，应替换块0的截断版
     {"start": 7.0, "end": 9.0, "text": "块1独有句。"}],
    # 块2 (offset 27): 本地 + 27
    [{"start": 0.3, "end": 2.0, "text": "块2开头。"},
     {"start": 3.0, "end": 5.0, "text": "中段。"}],
]

calls = []


def fake_upload(audio_path, key, model, language, proxy, prompt=None):
    idx = len(calls)
    calls.append({"prompt": prompt, "path": audio_path})
    segs = [dict(s) for s in BLOCKS[idx]]
    return {"segments": segs}


gs._upload_one = fake_upload

import tempfile
tmp = tempfile.mkdtemp(prefix="pipeline-")
# 造一个足够大的假"音频"文件（>4MB 才会走切块）——内容无所谓，走 --dry-run 式的假路径
src = os.path.join(tmp, "big.m4a")
with open(src, "wb") as f:
    f.write(b"\x00" * (5 * 1024 * 1024))

# split_audio 需要 mvhd 时长——直接 patch 掉，用假 chunks
fake_chunks = []
for i, off in enumerate([0.0, 12.0, 27.0]):
    fp = os.path.join(tmp, "seg%d.m4a" % i)
    with open(fp, "wb") as f:
        f.write(b"\x00" * 4096)  # getsize 只用于日志，内容无所谓
    fake_chunks.append((fp, off))
gs.split_audio = lambda *a, **k: fake_chunks

result = gs.transcribe(src, "gsk_test", "whisper-large-v3", "zh", None,
                       ffmpeg=FFMPEG, vocab=VOCAB, use_prompt=True, overlap=3.0)
segs = result["segments"]

print("\n--- prompt 链 ---")
for i, c in enumerate(calls):
    print("块%d prompt: %r" % (i, c["prompt"]))
assert calls[0]["prompt"] == "心理治疗 认知疗法", calls[0]["prompt"]
assert "交界句子完整版" in calls[1]["prompt"], calls[1]["prompt"]  # 块0 结尾文本
assert calls[1]["prompt"].startswith("心理治疗")
assert "块1独有句" in calls[2]["prompt"], calls[2]["prompt"]      # 块1 结尾文本
print("[OK] prompt 链式传递（词表+上文）")

print("\n--- 合并结果 ---")
for s in segs:
    print("  [%6.2f - %6.2f] %s" % (s["start"], s["end"], s["text"]))

texts = [s["text"] for s in segs]
times = [(s["start"], s["end"]) for s in segs]

# 去重：交界句子完整版只出现一次
assert texts.count("交界句子完整版。") == 1, texts
# 截断版被完整版替换：只保留 16.8-18.2 的完整时段（终点 +0.1 余韵 CUE_TAIL）
trunc = [s for s in segs if s["text"] == "交界句子被截断的半句"]
assert len(trunc) == 1, trunc
assert (trunc[0]["start"], trunc[0]["end"]) == (16.8, 18.3), trunc[0]
# 独有句保留
assert "块1独有句。" in texts and "块2开头。" in texts and "中段。" in texts
assert "正常句A。" in texts and "第二句。" in texts and "开场第一句。" in texts
# 时间单调递增
starts = [s["start"] for s in segs]
assert starts == sorted(starts), starts
# offset 正确：块1独有句本地 7-9 + offset 12 = 全局 19-21
i = texts.index("块1独有句。")
assert (segs[i]["start"], segs[i]["end"]) == (19.0, 21.1), segs[i]
# 块2 中段全局 30-32
i = texts.index("中段。")
assert (segs[i]["start"], segs[i]["end"]) == (30.0, 32.1), segs[i]
print("[OK] 去重/替换/补缺/offset/时序 全部符合预期")

# ---------- 小文件路径也带 prompt ----------
calls.clear()
small = os.path.join(tmp, "small.m4a")
with open(small, "wb") as f:
    f.write(b"\x00" * 1024)
gs._upload_one = fake_upload
gs.transcribe(small, "gsk_test", "whisper-large-v3", "zh", None,
              vocab=VOCAB, use_prompt=True)
assert calls[0]["prompt"] == "心理治疗 认知疗法", calls[0]["prompt"]
print("[OK] 小文件路径同样注入词表 prompt")

# ---------- use_prompt=False 完全禁用 ----------
calls.clear()
gs.transcribe(src, "gsk_test", "whisper-large-v3", "zh", None, ffmpeg=FFMPEG,
              vocab=VOCAB, use_prompt=False, overlap=3.0)
assert all(c["prompt"] is None for c in calls), [c["prompt"] for c in calls]
print("[OK] use_prompt=False 时完全不注入")

# ---------- multipart body 含 prompt 字段 ----------
captured = {}
import importlib
importlib.reload(gs)  # 重新加载模块，拿未污染的 _upload_one / _request
gs.split_audio = lambda *a, **k: fake_chunks


def fake_request(method, path, key, proxy, body=None, headers=None, retries=3):
    captured["body"] = body
    return 200, b'{"segments": []}'


gs._request = fake_request
x_file = os.path.join(tmp, "x.m4a")
with open(x_file, "wb") as f:
    f.write(b"\x00" * 2048)
gs._upload_one(x_file, "gsk_t", "whisper-large-v3", "zh",
               None, prompt="测试引导词")
assert b'name="prompt"' in captured["body"], "body 缺 prompt 字段"
assert "测试引导词".encode("utf-8") in captured["body"]
print("[OK] multipart body 含 prompt 字段")

# ---------- 无 prompt 时不带字段（向后兼容） ----------
captured.clear()
gs._upload_one(x_file, "gsk_t", "whisper-large-v3", "zh",
               None, prompt=None)
assert b'name="prompt"' not in captured["body"]
print("[OK] prompt=None 时 body 不含该字段")

print("\nALL PASS")
sys.exit(0)
