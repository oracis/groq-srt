#!/usr/bin/env python3
"""groq_srt.py - 音频转 SRT，走 Groq Whisper API（免费层 8 小时音频/天）。

用法:
    python groq_srt.py <音频> [-o out.srt] [--model whisper-large-v3]
                          [--language zh] [--ffmpeg <path>] [--keep] [--dry-run]
                          [--proxy socks5h://127.0.0.1:10808] [--key gsk_...]
    key 来源: --key 参数 > GROQ_API_KEY 环境变量
    代理优先级: --proxy > GROQ_PROXY > 默认 socks5h://127.0.0.1:10808

说明:
    Groq 封中国大陆 IP（返回 403 Forbidden），必须走海外出口。
    默认代理 = 本机 Shadowsocks SOCKS5(10808)，实测出口 US；
    也支持 http://host:port（CONNECT 隧道）与 socks5h://host:port，
    传 "none" 表示直连。脚本不读 http_proxy/https_proxy 环境变量
    （本机该环境变量指向一个国内出口的坏代理，勿用）。
    免费层单文件上限 25MB，但实测大 body（>5MB）走 SS 代理上传会被
    Groq 在读 body 过程中断连（响应也可能中途被掐断），故超过 4MB 的
    音频自动用 ffmpeg 切块、逐块上传、时间戳偏移合并（Groq 官方推荐的
    chunking 方案）。切块默认重叠 3 秒：每块头部自带上文，交界句子由相邻
    两块分别转写后按时间重叠择优（--overlap 调整，0=不重叠）。
    ffmpeg 探测顺序: --ffmpeg > PATH > 剪映安装包内置。
    上传前先 GET /openai/v1/models 预检 key 与出口区域（便宜且秒失败）。

    识别质量增强（默认开启，--no-prompt 关闭）：
    - prompt 引导词表：--vocab 传入正确写法（如 "心理治疗 欧文·亚隆"），
      注入 Whisper prompt，从解码期就倾向正确词，不等出事后再正则替换
    - 跨切块上文：第 i 块的 prompt 带上第 i-1 块结尾文本，消除切块上下文
      断片（交界处不断思路）

    字幕分段（词级时间戳 + 智能切分）：
    - 请求 timestamp_granularities=word，拿每个词的 start/end
    - _split_cues 按 句末标点 > 气口(词间静音>=0.7s) > 逗号(过半才断)
      > 宽度/时长兜底 的优先级切条，单条默认 <=20 个汉字宽、<=7s
    - 切块重叠区的词按 文本+时间双重重叠 判重去重（漏识的由另一块补上）
"""

import argparse
import glob
import json
import os
import re
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time

API_HOST = "api.groq.com"
API_PORT = 443
API_PATH = "/openai/v1/audio/transcriptions"
MODELS_PATH = "/openai/v1/models"
SIZE_LIMIT = 25 * 1024 * 1024
CHUNK_THRESHOLD = 4 * 1024 * 1024   # 超过 4MB 走切块（实测 5.6MB 会被断连）
OVERLAP_SEC = 3.0                   # 切块重叠秒数：每块头部带 3s 上文，交界句子由两块分别转写后择优
PROMPT_MAX_CHARS = 220              # Whisper prompt 上限约 448 token，中文 1 字≈1-1.5 token，保守截断
PROMPT_TAIL_CHARS = 120             # 上一块结尾带入 prompt 的字符数
DEFAULT_PROXY = "socks5h://127.0.0.1:10808"

# ---- 字幕分段参数（依据剪映官方"按语音停顿断句"选项 + 行业字幕规范） ----
# 剪映识别字幕官方提供「智能断句 / 按语音停顿断句」两种模式，后者专治
# 断句错位——即本脚本的"气口断开"。行业规范（多家剪辑指南一致）：
# 每行 15-20 个汉字（横屏取上限）、每条 1-3 秒、话音停字幕即停。
# 中文安全阅读速度约 4-6 字/秒，故 20 字配 6 秒上限。
CUE_MAX_CHARS = 10                  # 单条字幕最大显示宽度（汉字计 1，英文计 0.55）
# 实测剪映原生识别：老版平均 10.1 字/条（684 条样本，中位 9），新版开头
# 更是切成 6+6+8 字。真正的断句主力是气口（见 _realign_words 的 brk 标记），
# 这里的宽度上限只做兜底，10 字留给语义断句余量
CUE_MAX_DUR = 6.0                   # 单条最长秒数（超了观众会丢失同步感）
CUE_MIN_DUR = 1.0                   # 单条最短秒数（太短会闪跳）
BREAK_GAP = 0.6                     # 词间静音达到该值视为气口，断开（剪映停顿断句同思路）
STRONG_MIN_CHARS = 6                # 句末标点后断开的最小累计宽度
WEAK_BREAK_RATIO = 0.55             # 逗号级断点：累计宽度达到上限该比例才断

# ---- 音频对齐参数（用真实语音区间校正 Whisper 漂移的时间戳） ----
# Whisper 的词级时间戳是"推断"的而非"测量"的：时间 token 由模型像普通 token
# 一样预测，segment 内的长停顿会把它整体拉偏（社区公认的 "drift after
# silences"）。stable-ts 为此提供 adjust_by_silence()：先用静音检测拿到音频
# 里真实的语音区间，再把词钉回去。这里用 ffmpeg silencedetect 实现，零依赖。
ALIGN_TO_SPEECH = True              # 用音频静音区校正词时间戳（关掉=完全信 Whisper）
SILENCE_MIN_DUR = 0.35              # 短于此值的停顿不算语音区间边界
SPEECH_MIN_DUR = 0.12               # 短于此值的"语音"碎片视为噪声丢弃
CUE_TAIL = 0.10                     # 字幕结束比末词多留的秒数（余韵，勿越入长停顿）
ALIGN_BREAK_GAP = 0.40              # 对齐态的断句气口门槛：区间间隔达到该值才断
MIN_CUE_WIDTH = 4                   # 单条最小显示宽度：低于此值的碎条并入上一条，避免闪读
# 口播水词：剪映的断句引擎会忽略"嗯/啊"这类填充词周围的短停顿，不让它单独成条、
# 也不在这里断句。同理：上一条以水词结尾时，跨越这个气口继续接下文。
_FILLERS = set("嗯啊呃哦唉诶哎噢呀嘛吧呢哈")
SOFT_GAP = 0.25                     # 软断点：短停顿 + 标点同时命中才切（剪映加权思路）

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def die(msg, code=1):
    print("[error] " + msg, file=sys.stderr)
    sys.exit(code)


def find_ffmpeg(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append("ffmpeg")
    cands += glob.glob(r"D:\Jianying6.01\*\*\ffmpeg.exe")
    cands += glob.glob(r"D:\Jianying*\*\ffmpeg.exe")
    cands += glob.glob(r"C:\Program Files*\ffmpeg*\bin\ffmpeg.exe")
    for c in cands:
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True)
            return c
        except Exception:
            continue
    die("找不到可用的 ffmpeg。装一个，或用 --ffmpeg 指定路径。")


def transcode(src, dst, ffmpeg, bitrate="64k"):
    cmd = [ffmpeg, "-y", "-v", "error", "-i", src,
           "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", bitrate, dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die("ffmpeg 转码失败:\n" + (r.stderr or "")[:500])
    return os.path.getsize(dst)


def m4a_end_time(path):
    """解析 m4a/mp4 的 mvhd box。

    注意语义：ffmpeg segment muxer 切出的块，其 mvhd duration 不是"块自身
    时长"，而是"该块内容在原文件时间轴上的结束时刻"（如 480s 均分切块：
    第1块 480、第2块 960、第3块 1440……）。因此第 i 块的全局起始 offset
    正好等于第 i-1 块的 mvhd 值。失败返回 None。
    """
    try:
        with open(path, "rb") as f:
            fsize = os.path.getsize(path)

            def read_at(pos, ln):
                f.seek(pos)
                return f.read(ln)

            def walk(start, end):
                pos = start
                while pos + 8 <= end:
                    h = read_at(pos, 16)
                    if len(h) < 8:
                        return None
                    size = struct.unpack(">I", h[0:4])[0]
                    btype = h[4:8]
                    hs = 8
                    if size == 1:
                        if len(h) < 16:
                            return None
                        size = struct.unpack(">Q", h[8:16])[0]
                        hs = 16
                    elif size == 0:
                        size = end - pos
                    if size < hs or pos + size > end:
                        return None
                    if btype == b"moov":
                        r = walk(pos + hs, pos + size)
                        if r is not None:
                            return r
                    elif btype == b"mvhd":
                        v = read_at(pos + hs, 100)
                        ver = v[0]
                        if ver == 1:
                            ts = struct.unpack(">I", v[20:24])[0]
                            dur = struct.unpack(">Q", v[24:32])[0]
                        else:
                            ts = struct.unpack(">I", v[12:16])[0]
                            dur = struct.unpack(">I", v[16:20])[0]
                        if ts:
                            return dur / ts
                        return None
                    pos += size
                return None

            return walk(0, fsize)
    except Exception:
        return None


def split_audio(src, out_dir, n_chunks, ffmpeg, overlap=OVERLAP_SEC):
    """把 src 切成 n_chunks 块，相邻块重叠 overlap 秒。

    块 i 覆盖 [i*T-overlap, (i+1)*T+overlap)（首/尾块按文件边界裁剪），
    即每块头部自带 overlap 秒的上文音频——交界处的句子会被相邻两块
    分别转写，合并时按时间重叠度择优（见 _merge_segments）。

    切块用 -ss 输入定位 + 流拷贝。实测（PCM 互相关，白噪声源，4 个采样点
    含非网格对齐位置）：实际解码起点与 -ss 请求值误差 0.0ms——ffmpeg 定位到
    前一个音频包后用 elst 编辑列表精确修剪包内偏移。故 offset 即请求值，
    无需网格取整或 mvhd 反推。

    返回 [(块路径, 该块起始 offset 秒), ...]。
    """
    total = m4a_end_time(src)
    if not total:
        die("无法解析音频时长，切块失败。")
    seg_time = total / n_chunks
    ov = overlap if n_chunks > 1 else 0.0
    if ov > seg_time * 0.4:
        ov = seg_time * 0.4
    chunks = []
    for i in range(n_chunks):
        start = max(0.0, i * seg_time - ov)
        end = min(total, (i + 1) * seg_time + ov)
        if end - start < 1.0:
            continue  # 尾部残片（重叠已覆盖其内容）
        fp = os.path.join(out_dir, "seg%03d.m4a" % i)
        cmd = [ffmpeg, "-y", "-v", "error",
               "-ss", "%.3f" % start, "-i", src,
               "-c", "copy", "-t", "%.3f" % (end - start), fp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            die("ffmpeg 切块失败:\n" + (r.stderr or "")[:500])
        if os.path.getsize(fp) < 10240:
            continue
        d = m4a_end_time(fp)
        if d and abs(d - (end - start)) > 0.5:
            print("[warn] 块 %d 时长异常: 期望 %.1fs 实得 %.1fs"
                  % (i + 1, end - start, d))
        chunks.append((fp, start))
    if not chunks:
        die("切块后没有有效文件。")
    return chunks


def srt_ts(sec):
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


# ---------- 代理隧道（纯标准库实现 SOCKS5 / HTTP-CONNECT + TLS） ----------

def make_client_context():
    """客户端 TLS context。生产用默认（校验证书）；测试可 monkeypatch 成不校验。"""
    return ssl.create_default_context()


def _socks5_handshake(sock, host, port):
    sock.sendall(b"\x05\x01\x00")  # VER=5, 1 method, no-auth
    resp = sock.recv(2)
    if len(resp) < 2 or resp[0] != 5:
        raise RuntimeError("SOCKS5 代理响应异常")
    if resp[1] == 0xFF:
        raise RuntimeError("SOCKS5 代理要求认证（本脚本仅支持无认证的本地代理）")
    host_b = host.encode("ascii")
    # ATYP=3 域名：由代理解析（socks5h 语义，防 DNS 污染）
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host_b)]) + host_b
                 + port.to_bytes(2, "big"))
    head = sock.recv(4)
    if len(head) < 4:
        raise RuntimeError("SOCKS5 CONNECT 响应不完整")
    if head[1] != 0:
        raise RuntimeError("SOCKS5 CONNECT 被拒 (code=%d)" % head[1])
    atyp = head[3]
    if atyp == 1:
        sock.recv(4)
    elif atyp == 3:
        ln = sock.recv(1)
        if ln:
            sock.recv(ln[0])
    elif atyp == 4:
        sock.recv(16)
    sock.recv(2)  # BND.PORT


def _http_connect(sock, host, port):
    req = ("CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n"
           "Proxy-Connection: keep-alive\r\n\r\n" % (host, port, host, port))
    sock.sendall(req.encode("ascii"))
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    status = buf.split(b"\r\n", 1)[0]
    if b" 200" not in status:
        raise RuntimeError("HTTP 代理 CONNECT 失败: %s"
                           % status.decode("latin1", "replace")[:80])


def make_tls_tunnel(proxy):
    """返回 make(host, port) -> TLS socket；proxy 为 None/none/direct 时返回 None。"""
    if not proxy or proxy.lower() in ("none", "direct", "off"):
        return None
    m = re.match(r"^(socks5h|socks5|socks4|http)://([^/:]+):(\d+)/?$",
                 proxy.strip(), re.I)
    if not m:
        die("无法解析代理地址: %s（示例 socks5h://127.0.0.1:10808 或 http://127.0.0.1:8080）"
            % proxy)
    scheme = m.group(1).lower()
    phost, pport = m.group(2), int(m.group(3))

    def make(host, port):
        s = socket.create_connection((phost, pport), timeout=30)
        s.settimeout(600)
        try:
            if scheme.startswith("socks"):
                _socks5_handshake(s, host, port)
            else:
                _http_connect(s, host, port)
            ctx = make_client_context()
            return ctx.wrap_socket(s, server_hostname=host)
        except Exception:
            try:
                s.close()
            except OSError:
                pass
            raise
    return make


# ---------- HTTP 请求层（chunked 解码 + 提前断连的响应抢救 + 重试） ----------

class _NetError(Exception):
    pass


def _friendly_net_error(e, proxy):
    msg = str(e)
    low = msg.lower()
    if "10061" in msg or "refused" in low:
        return ("连不上代理 %s（连接被拒）。启动 Shadowsocks/代理软件，或用 --proxy 换一个。"
                % proxy)
    if any(k in msg for k in ("10053", "10054", "10057")) \
            or "aborted" in low or "reset" in low:
        return ("连接被中止（%s）。代理链路不稳：脚本会自动重试，持续失败就换节点。" % msg[:80])
    return msg[:200]


def _read_body(sock, hdrs, initial):
    """按 header 读完整 body：有 Content-Length 按长度读，否则按 chunked 解。"""
    if b"content-length" in hdrs:
        clen = int(hdrs[b"content-length"])
        body = initial
        while len(body) < clen:
            chunk = sock.recv(65536)
            if not chunk:
                break
            body += chunk
        if len(body) < clen:
            raise _NetError("响应体被截断（Content-Length=%d，实收 %d）。"
                            "代理链路在响应传输中途断连，重试或换节点。" % (clen, len(body)))
        return body[:clen]
    if hdrs.get(b"transfer-encoding", b"").lower().find(b"chunked") >= 0:
        return _read_chunked(sock, initial)
    # 无长度无 chunked：读到 EOF
    body = initial
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return body


def _read_chunked(sock, initial):
    """解码 chunked transfer-encoding。中途断连抛 _NetError。"""
    buf = bytearray(initial)
    body = bytearray()
    while True:
        while b"\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise _NetError("响应被截断（chunked 流的 chunk 头都没收全）")
            buf += chunk
        line, _, rest = bytes(buf).partition(b"\r\n")
        buf = bytearray(rest)
        try:
            size = int(line.split(b";")[0].strip(), 16)
        except ValueError:
            raise _NetError("chunked 响应格式异常: %r" % line[:40])
        if size == 0:
            while b"\r\n\r\n" not in buf and b"\r\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
            break
        while len(buf) < size + 2:
            chunk = sock.recv(65536)
            if not chunk:
                raise _NetError("响应被截断（chunk 数据不完整，缺 %d 字节）"
                                % (size + 2 - len(buf)))
            buf += chunk
        body += buf[:size]
        del buf[:size + 2]
    return bytes(body)


def _read_response(sock):
    """从 socket 读一个完整的 HTTP 响应，返回 (status, body_bytes)。"""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    parts = lines[0].split(b" ", 2)
    if len(parts) < 2 or not parts[0].startswith(b"HTTP/"):
        raise RuntimeError("响应格式异常: %s"
                           % head[:80].decode("latin1", "replace"))
    status = int(parts[1])
    hdrs = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(b":")
        hdrs[k.strip().lower()] = v.strip()
    return status, _read_body(sock, hdrs, rest)


def _send_body_async(sock, body):
    """后台线程发送 body。服务器可能在读完全部 body 前就提前响应（如无效 key
    的 401、限流 429），那时主线程已读到响应并关闭 socket，本线程的 send 会
    报错退出——静默忽略即可。"""
    def worker():
        try:
            view = memoryview(body)
            sent = 0
            while sent < len(view):
                n = min(65536, len(view) - sent)
                sock.sendall(view[sent:sent + n])
                sent += n
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def _request_once(method, path, key, proxy, body=None, headers=None):
    """发一次请求（不重试）。成功返回 (status, raw_bytes)；网络层失败抛 _NetError。"""
    req_lines = ["%s %s HTTP/1.1" % (method, path),
                 "Host: %s:%d" % (API_HOST, API_PORT),
                 "Accept: application/json",
                 "Authorization: Bearer " + key,
                 "Connection: close"]
    for k, v in (headers or {}).items():
        req_lines.append("%s: %s" % (k, v))
    head = ("\r\n".join(req_lines) + "\r\n\r\n").encode("latin1")

    try:
        make = make_tls_tunnel(proxy) if proxy else None
        if make is not None:
            sock = make(API_HOST, API_PORT)
        else:
            raw = socket.create_connection((API_HOST, API_PORT), timeout=600)
            sock = make_client_context().wrap_socket(raw, server_hostname=API_HOST)
    except Exception as e:
        raise _NetError(_friendly_net_error(e, proxy))
    try:
        try:
            sock.sendall(head)
            if body:
                _send_body_async(sock, body)
            # 主线程阻塞读响应：SSL 层会自动消化 TLS 1.3 的 session ticket 等
            # 握手机制消息，不会把"可读"误判成响应到达。
            return _read_response(sock)
        except Exception as send_err:
            raise _NetError(_friendly_net_error(send_err, proxy))
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _request(method, path, key, proxy, body=None, headers=None, retries=3):
    """带重试的请求。网络层失败重试；HTTP 状态码原样返回由调用方处理。"""
    last = None
    for attempt in range(retries):
        try:
            return _request_once(method, path, key, proxy, body, headers)
        except _NetError as e:
            last = e
            if attempt < retries - 1:
                print("[warn] 第 %d 次请求失败（%s），%.1f 秒后重试..."
                      % (attempt + 1, str(e)[:70], 1.5 * (attempt + 1)))
                time.sleep(1.5 * (attempt + 1))
    raise last


def _die_for_status(status, detail):
    if status == 403:
        die("Groq 返回 403 Forbidden - 出口 IP 仍被 Groq 封锁。\n"
            "  自检: curl -x socks5h://127.0.0.1:10808 "
            "https://www.cloudflare.com/cdn-cgi/trace 看 loc= 是否非 CN；\n"
            "  若在 CN，换代理节点（美/日/新/欧）。")
    if status == 401:
        die("Groq 返回 401 - key 无效或已被撤销。\n"
            "  去 console.groq.com → API Keys 确认密钥还在（可能被删/重建/风控），\n"
            "  必要时重新创建一个（gsk_ 开头；xai- 开头的是 xAI 的，不通用）。")
    if status == 429:
        die("Groq 返回 429 - 触发限流（免费层 20 请求/分钟、8 小时音频/天）。稍后再试。")
    die("Groq API 错误 %d: %s" % (status, (detail or "")[:300]))


def preflight(key, proxy):
    """上传前便宜地验证 key 与出口区域（几 KB 的 GET，秒失败）。"""
    print("[..] 预检 key 与出口区域（GET /models）...")
    try:
        status, raw = _request("GET", MODELS_PATH, key, proxy)
    except _NetError as e:
        die(str(e))
    if status != 200:
        _die_for_status(status, raw.decode("utf-8", "replace"))
    try:
        n = len(json.loads(raw.decode("utf-8")).get("data", []))
        print("[ok] key 有效，出口区域放行（可用模型 %d 个）" % n)
    except Exception:
        print("[ok] key 有效，出口区域放行")


# ---------- Groq 转录（大文件自动切块 + 时间戳偏移合并） ----------

def build_vocab_text(vocab):
    """vocab: 字符串列表或逗号/空格分隔的字符串 -> 去重后空格连接。"""
    if not vocab:
        return ""
    if isinstance(vocab, str):
        items = re.split(r"[,，;；\s]+", vocab)
    else:
        items = []
        for v in vocab:
            items += re.split(r"[,，;；\s]+", str(v))
    seen, out = set(), []
    for it in items:
        it = it.strip()
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return " ".join(out)


def build_prompt(vocab_text, prev_tail):
    """组 Whisper prompt：引导词 + 上一块结尾文本。

    prompt 对 Whisper 是"续写引导"：词表让解码器倾向正确写法，上文尾句
    恢复跨切块的上下文（交界处不断片）。上文放最后（紧邻解码起点）。
    总长受 448 token 上限约束，超长时优先保上文、砍词表。
    """
    vocab_text = (vocab_text or "").strip()
    prev_tail = (prev_tail or "").strip()
    tail = prev_tail[-PROMPT_TAIL_CHARS:] if prev_tail else ""
    p = (vocab_text + " " + tail).strip() if vocab_text else tail
    if len(p) > PROMPT_MAX_CHARS:
        room = PROMPT_MAX_CHARS - len(tail) - 1
        p = (vocab_text[:room] + " " + tail).strip() if room > 20 else tail
    return p


def _merge_segments(all_segs):
    """合并各块 segments（已加全局 offset），按时间重叠贪心去重。

    重叠切块会在交界处产出同一句子的两份转写：按时间重叠度合并——
    重叠超过阈值时保留时长更长的那份（通常就是未被截断的完整版），
    仅一块有的句子（另一块漏识）原样保留，顺带完成补缺。
    """
    segs = [s for s in all_segs if (s.get("text") or "").strip()]
    segs.sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))
    kept = []
    for seg in segs:
        ss = seg.get("start", 0)
        se = seg.get("end", 0)
        dup_of = None
        for j in range(len(kept) - 1, max(-1, len(kept) - 4), -1):
            ps, pe = kept[j].get("start", 0), kept[j].get("end", 0)
            ov = min(pe, se) - max(ps, ss)
            shorter = min(pe - ps, se - ss)
            if ov > 0.45 and shorter > 0 and ov > shorter * 0.45:
                dup_of = j
                break
        if dup_of is None:
            kept.append(seg)
        elif (se - ss) > (kept[dup_of].get("end", 0)
                          - kept[dup_of].get("start", 0)) * 1.1:
            kept[dup_of] = seg  # 新版本明显更完整，替换
    return kept


# ---------- 词级时间戳 -> 字幕分段 ----------

_STRONG_PUNCT = "。！？；…"
_WEAK_PUNCT = "，、,;：:）)"
_PUNCT_RE = re.compile(r"[\s，。！？；、,.!?;:：…（）()\"'“”‘’\-—]")


def _display_width(text):
    """字幕显示宽度：CJK/全角标点计 1，其余（英文数字）计 0.55。"""
    w = 0.0
    for ch in text:
        w += 1.0 if ord(ch) > 0x2E80 else 0.55
    return w


def _norm_word(text):
    """词文本归一化：去空白与标点，用于跨块同一词的判重。"""
    return _PUNCT_RE.sub("", text or "")


def _result_words(result):
    """从 API 响应提取词级时间戳。

    verbose_json + timestamp_granularities=word 时返回
    words: [{word, start, end}, ...]；个别情况下没有 words 字段，
    退化为把 segments 当整词（一条字幕=一段，分段器会原样保留）。
    """
    words = result.get("words")
    if isinstance(words, list) and words:
        return [{"word": w.get("word") or "", "start": float(w.get("start") or 0),
                 "end": float(w.get("end") or 0)} for w in words]
    segs = result.get("segments") or []
    return [{"word": s.get("text") or "", "start": float(s.get("start") or 0),
             "end": float(s.get("end") or 0)} for s in segs]


def _merge_words(all_words):
    """合并各块 words（已加全局 offset）：按时间排序，文本+时间双重重叠判重。

    重叠切块会在交界处把同一句子转写两遍，词序列几乎一致：归一化文本相同
    且时间区间显著重叠（>较短者一半）即判为同一词，保留一个并展宽时间范围
    （两侧时间戳取并集，边界更稳）。只在一块出现的词原样保留=漏识补缺。
    """
    ws = [w for w in all_words if _norm_word(w.get("word"))]
    ws.sort(key=lambda w: (w.get("start", 0), w.get("end", 0)))
    kept = []
    for w in ws:
        dup_of = None
        for j in range(len(kept) - 1, max(-1, len(kept) - 10), -1):
            k = kept[j]
            if _norm_word(k["word"]) != _norm_word(w["word"]):
                continue
            ov = min(k["end"], w["end"]) - max(k["start"], w["start"])
            shorter = min(k["end"] - k["start"], w["end"] - w["start"])
            if ov > 0.25 and shorter > 0 and ov > shorter * 0.5:
                dup_of = j
                break
        if dup_of is None:
            kept.append(dict(w))
        else:  # 同一词的两份识别：时间范围取并集
            k = kept[dup_of]
            k["start"] = min(k["start"], w["start"])
            k["end"] = max(k["end"], w["end"])
    kept.sort(key=lambda w: (w.get("start", 0), w.get("end", 0)))
    return kept


def _ends_with_filler(text):
    """末尾是口播水词（嗯/啊/呃…）？用于跳过水词处的气口断句。"""
    t = _PUNCT_RE.sub("", text or "")
    return bool(t) and t[-1] in _FILLERS


def _explode_long_words(words, max_chars):
    """把超宽的单个"词"二次切开，时间按显示宽度比例分配。

    Groq 偶尔只返回 segments 不给 words（或个别 chunk 的粒度异常粗），
    此时一个"词"就是整段几十上百字——不切开会生成一条几十秒的巨型字幕
    （实测旧草稿里出现过 223 字 / 59 秒的单条）。切分优先落在标点上，
    其次按宽度硬切，时间按字数比例摊到各片。
    """
    out = []
    for w in words:
        t = (w.get("word") or "").strip()
        if _display_width(t) <= max_chars * 1.6:
            out.append(w)
            continue
        # 先按标点切成小块
        parts, buf = [], ""
        for ch in t:
            buf += ch
            if ch in _STRONG_PUNCT or ch in _WEAK_PUNCT:
                parts.append(buf)
                buf = ""
        if buf:
            parts.append(buf)
        # 过宽的块再按显示宽度硬切
        pieces = []
        for pt in parts:
            while _display_width(pt) > max_chars:
                acc, cut = 0.0, len(pt)
                for idx, ch in enumerate(pt):
                    acc += _display_width(ch)
                    if acc > max_chars:
                        cut = max(1, idx)
                        break
                pieces.append(pt[:cut])
                pt = pt[cut:]
            if pt:
                pieces.append(pt)
        total = sum(_display_width(p) for p in pieces) or 1.0
        s0, e0 = w.get("start", 0), w.get("end", 0)
        acc = 0.0
        for k, p in enumerate(pieces):
            pw = _display_width(p)
            nw = {"word": p,
                  "start": round(s0 + (e0 - s0) * acc / total, 3),
                  "end": round(s0 + (e0 - s0) * (acc + pw) / total, 3)}
            if k == 0 and w.get("brk"):
                nw["brk"] = True
            out.append(nw)
            acc += pw
    return out


def _split_cues(words, max_chars=CUE_MAX_CHARS, max_dur=CUE_MAX_DUR,
                min_dur=CUE_MIN_DUR, gap=BREAK_GAP):
    """把词级时间戳切成适合屏幕阅读的字幕条。

    断开优先级（从高到低）：
    0. 真实气口——词带 brk 标记（音频对齐测出的语音区间起点），见之即断
    1. 句末标点（。！？；…）——句子说完就断
    2. 气口——词间静音 >= gap 秒（说话人换气/停顿；无对齐时的退化判据）
    3. 逗号级标点（，、：等）——累计宽度过半才断，避免碎句
    4. 兜底——累计宽度/时长将超上限时，在最近的逗号处切；没有就硬切
    每条 cue: start=首词 start，end=末词 end + 余韵 CUE_TAIL，text=拼接。
    """
    # 防御：把超宽的巨型词先切开（Groq 未返回 words 时的退化路径）
    words = _explode_long_words(words, max_chars)
    words = [w for w in words if (w.get("word") or "").strip()]
    cues = []
    cur = []  # [(word, start, end), ...]

    def text_of(items):
        return "".join(t for t, _, _ in items)

    def flush(items=None):
        nonlocal cur
        if items is None:
            items = cur
            cur = []
        if not items:
            return
        txt = text_of(items).strip()
        if txt:
            cues.append({"start": round(items[0][1], 3),
                         "end": round(items[-1][2] + CUE_TAIL, 3), "text": txt})

    for w in words:
        t = (w.get("word") or "").strip()
        ws, we = float(w.get("start") or 0), float(w.get("end") or 0)
        if we < ws:  # 防御：个别词 end 缺失/倒挂
            we = ws
        if cur:
            prev_end = cur[-1][2]
            silence = ws - prev_end
            cur_w = _display_width(text_of(cur))
            would_w = cur_w + _display_width(t)
            # 0) 真实气口：音频对齐判定此处有停顿，且已有足够内容才断；
            #    但上一条停在"嗯/啊"这类水词上时不切（口播换气不该单独成条）
            if w.get("brk") and cur_w >= 4 and not _ends_with_filler(text_of(cur)):
                flush()
            # 1) 气口断开（无 brk 标记时的退化判据）
            elif silence >= gap and cur_w >= 4:
                flush()
            # 2) 软断点：短停顿(0.25~气口阈值) + 标点 + 已积够半数宽度，三条
            #    信号叠加到阈值以上才切——单看任何一条都不够（剪映的加权思路）
            elif (silence >= SOFT_GAP
                  and any(c in t for c in _STRONG_PUNCT + _WEAK_PUNCT)
                  and cur_w >= max_chars * 0.5):
                flush()
            # 4a) 宽度/时长兜底：先找 cue 内最近的逗号切点
            elif would_w > max_chars or (we - cur[0][1]) > max_dur:
                cut = -1
                for j in range(len(cur) - 1, -1, -1):
                    ct = cur[j][0]
                    if any(c in ct for c in _WEAK_PUNCT) and j >= 2:
                        cut = j
                        break
                if cut >= 0:
                    head, cur = cur[:cut + 1], cur[cut + 1:]
                    flush(head)
                elif (any(c in t for c in _WEAK_PUNCT)
                      and would_w <= max_chars * 1.3):
                    # 断点正好落在标点上：允许小幅超宽，别把词劈开
                    cur.append((t, ws, we))
                    flush()
                    continue
                else:
                    flush()
        cur.append((t, ws, we))
        # 2) 句末标点：本词收尾整句
        if any(c in t for c in _STRONG_PUNCT):
            if _display_width(text_of(cur)) >= STRONG_MIN_CHARS or len(cur) > 1:
                flush()
        # 3) 逗号级：累计过半才断
        elif any(c in t for c in _WEAK_PUNCT) \
                and _display_width(text_of(cur)) >= max_chars * WEAK_BREAK_RATIO:
            flush()
    flush()

    # 孤儿条合并：宽度不足 MIN_CUE_WIDTH 的碎条并入上一条（避免 1-3 字闪读），
    # 前提是合并后不超宽。宁可上一条多停 0.2s，也别闪一屏 2 个字。
    if len(cues) > 1:
        kept = [cues[0]]
        for c in cues[1:]:
            p = kept[-1]
            # 只在"又短又快"且前面没有大停顿时才合并：
            # 文本窄 + 停留不足 min_dur = 闪读；若前面已有 >= BREAK_GAP 的
            # 大停顿则保持独立（宁可短一点，也不要让上一条跨过气口）。
            too_short = (_display_width(c["text"]) < MIN_CUE_WIDTH
                         and (c["end"] - c["start"]) < min_dur)
            small_gap = (c["start"] - p["end"]) < BREAK_GAP
            if (too_short and small_gap
                    and _display_width(p["text"]) + _display_width(c["text"])
                    <= max_chars):
                p["text"] += c["text"]
                p["end"] = max(p["end"], c["end"])
            else:
                kept.append(c)
        cues = kept

    # 时长下限 + 防重叠：太短的条拉长到 min_dur；任何情况下都不许压住下一条
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if c["end"] - c["start"] < min_dur:
            c["end"] = c["start"] + min_dur
        if nxt is not None and c["end"] > nxt - 0.02:
            # 这里必须是 min 而不是 max(start+0.3, ...)：下一条紧邻时，
            # "至少 0.3 秒"的兜底会重新越过它，让两条字幕叠在一起
            # （实测《众神》464.4s 处三条 0.2s 碎条互压）。宁可这一条短，
            # 也不能压住下一条——重叠在剪映里是肉眼可见的糊字。
            c["end"] = min(c["end"], nxt - 0.02)
    # 被压到几乎没有时长的碎条：文字并进上一条，避免闪读又丢字
    merged = []
    for c in cues:
        if merged and (c["end"] - c["start"]) < 0.05:
            p = merged[-1]
            if _display_width(p["text"]) + _display_width(c["text"]) <= max_chars:
                p["text"] += c["text"]
                p["end"] = max(p["end"], c["end"])
                continue
        merged.append(c)
    return merged


def _parse_db(text, key):
    m = re.search(key + r":\s*(-?\d+(?:\.\d+)?)\s*dB", text or "")
    return float(m.group(1)) if m else None


def _parse_silences(text):
    """从 silencedetect 的 stderr 里解析静音区间 [(start, end), ...]。"""
    out, cur = [], None
    for line in (text or "").splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            cur = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and cur is not None:
            out.append((cur, float(m.group(1))))
            cur = None
    if cur is not None:          # 一直静到文件结尾
        out.append((cur, float("inf")))
    return out


def detect_speech_regions(audio_path, ffmpeg, log=print,
                          min_sil=SILENCE_MIN_DUR, min_speech=SPEECH_MIN_DUR):
    """从波形直接测出真实说话区间，返回 [(start, end), ...]（秒）。

    先 volumedetect 拿整体响度，据此自适应定噪声门限（均值 -8dB，夹在
    -45..-26dB），再 silencedetect 找静音段，取补集即语音段。探测不到任何
    静音（底噪大/门限过高）时降 6dB 重试一次。

    这份区间是从音频量出来的地面真值——Whisper 的时间戳是模型"推断"的，
    遇到 segment 内长停顿会整体偏移，必须用它来纠正。
    """
    def run_detect(noise_db):
        af = "silencedetect=noise=%ddB:d=%.2f" % (int(round(noise_db)), min_sil)
        r = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", audio_path,
                            "-af", af, "-f", "null", "-"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="ignore")
        return r.stderr or ""

    def probe_mean():
        r = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", audio_path,
                            "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="ignore")
        return _parse_db(r.stderr or "", "mean_volume")

    mean_db = probe_mean()
    noise_db = -35.0 if mean_db is None else max(-45.0, min(-26.0, mean_db - 8.0))
    silences = _parse_silences(run_detect(noise_db))
    if not silences:
        alt = _parse_silences(run_detect(noise_db - 6.0))
        if alt:
            silences, noise_db = alt, noise_db - 6.0
    if not silences:
        return []

    total = m4a_end_time(audio_path) or 0.0
    regions, pos = [], 0.0
    for s, e in silences:
        e = min(e, total) if total else e
        if s - pos >= min_speech:
            regions.append((pos, s))
        pos = max(pos, e)
    if total and total - pos >= min_speech:
        regions.append((pos, total))
    if log:
        log("[..] 静音检测: 门限 %ddB，%d 段停顿 -> %d 段语音"
            % (int(round(noise_db)), len(silences), len(regions)))
    return regions


def _realign_words(words, regions):
    """把词时间戳钉到真实语音区间上，并在每个气口打上断句标记。

    1) 每个词归到与之重叠最多（无重叠则中心最近）的语音区间；
    2) 强制区间序号单调不减（词序不可逆，避免交叉）；
    3) 同区间的词按 Whisper 给的相对位置线性重映射到该区间的真实起止；
    4) 区间首词打 brk=True —— 这就是"气口"，_split_cues 见之即断。

    结果：词不会落在静音里，字幕起止天然对上气口。
    """
    if not words or not regions:
        return words
    ws = [dict(w) for w in words]
    ws.sort(key=lambda w: (w.get("start", 0), w.get("end", 0)))
    n = len(regions)
    idx = []
    for w in ws:
        s, e = w["start"], w["end"]
        best, best_ov = -1, 0.0
        for k, (rs, re_) in enumerate(regions):
            ov = min(e, re_) - max(s, rs)
            if ov > best_ov:
                best_ov, best = ov, k
        if best < 0:
            c = (s + e) / 2.0
            best = min(range(n), key=lambda k: abs(
                (regions[k][0] + regions[k][1]) / 2.0 - c))
        idx.append(best)
    for i in range(1, len(idx)):
        if idx[i] < idx[i - 1]:
            idx[i] = idx[i - 1]

    out, i, prev_k = [], 0, None
    while i < len(ws):
        k = idx[i]
        j = i
        while j < len(ws) and idx[j] == k:
            j += 1
        grp = ws[i:j]
        rs, re_ = regions[k]
        span = max(re_ - rs, 0.02)
        src_s = min(g["start"] for g in grp)
        src_e = max(g["end"] for g in grp)
        src = src_e - src_s
        if src <= 0.02:
            # 退化（所有词同一时刻）：按显示宽度比例铺开
            tot = sum(max(_display_width((g.get("word") or "").strip()), 1)
                      for g in grp) or 1
            acc = 0.0
            for g in grp:
                gw = max(_display_width((g.get("word") or "").strip()), 1)
                ns = rs + span * acc / tot
                acc += gw
                out.append({"word": g["word"], "start": round(ns, 3),
                            "end": round(rs + span * acc / tot, 3)})
        else:
            for g in grp:
                ns = rs + (g["start"] - src_s) / src * span
                ne = rs + (g["end"] - src_s) / src * span
                # 线性重映射可能溢出区间边界，钳回来
                ns = min(max(ns, rs), re_)
                ne = min(max(ne, ns), re_)
                out.append({"word": g["word"], "start": round(ns, 3),
                            "end": round(ne, 3)})
        # 区间首词 = 气口断点；但只有真正拉开距离（>= ALIGN_BREAK_GAP）才断，
        # 0.35~0.4s 的句中小换气不算气口，硬断会把句子切碎
        gap_before = regions[k][0] - regions[prev_k][1] if prev_k is not None else 0.0
        out[-len(grp)]["brk"] = prev_k is not None and gap_before >= ALIGN_BREAK_GAP
        prev_k = k
        i = j
    return out


def _align_words(words, audio_path, ffmpeg, log=print, enabled=ALIGN_TO_SPEECH):
    """按音频真实语音区间校正词时间戳；失败时原样返回（不阻断转录）。"""
    if not enabled or not ffmpeg or not words:
        return words
    try:
        regions = detect_speech_regions(audio_path, ffmpeg, log=log)
    except Exception as e:                       # noqa: BLE001 探测失败不影响主流程
        if log:
            log("[warn] 音频对齐失败（%s），沿用 Whisper 时间戳" % e)
        return words
    if not regions:
        if log:
            log("[warn] 未探测到语音区间，沿用 Whisper 时间戳")
        return words
    base = sorted(words, key=lambda w: (w.get("start", 0), w.get("end", 0)))
    out = _realign_words(words, regions)
    moved = sum(1 for a, b in zip(base, out)
                if abs(a.get("start", 0) - b.get("start", 0)) > 0.15)
    if log:
        log("[ok] 音频对齐: %d 段语音，%d/%d 个词时间被校正(>0.15s)，"
            "气口 %d 处" % (len(regions), moved, len(out),
                            sum(1 for w in out if w.get("brk"))))
    return out


def _upload_one(audio_path, key, model, language, proxy, prompt=None):
    """上传单个文件（调用方确保 <4MB），返回解析后的 JSON dict。"""
    boundary = "----groqsrt" + os.urandom(16).hex()
    with open(audio_path, "rb") as f:
        file_bytes = f.read()
    filename = os.path.basename(audio_path)

    def field(name, value):
        return ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
                % (boundary, name, value)).encode("utf-8")

    body = b""
    body += field("model", model)
    body += field("response_format", "verbose_json")
    # 词级时间戳：segment 级一段可达几十秒，字幕会堆成一大块；
    # 词级才能按标点/气口智能分段（_split_cues）
    body += field("timestamp_granularities[]", "word")
    if language:
        body += field("language", language)
    if prompt:
        body += field("prompt", prompt)
    body += ('--%s\r\nContent-Disposition: form-data; name="file"; filename="%s"\r\n'
             "Content-Type: audio/mp4\r\n\r\n" % (boundary, filename)).encode("utf-8")
    body += file_bytes
    body += ("\r\n--%s--\r\n" % boundary).encode("utf-8")

    try:
        status, raw = _request(
            "POST", API_PATH, key, proxy, body=body,
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                     "Content-Length": str(len(body))})
    except _NetError as e:
        die(str(e))
    if status != 200:
        _die_for_status(status, raw.decode("utf-8", "replace"))
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        die("响应解析失败（%s）。响应体损坏或不完整，重跑一次；"
            "持续失败可能是代理链路不稳，换节点。" % type(e).__name__)


def transcribe(audio_path, key, model, language, proxy=None, ffmpeg=None,
               vocab=None, use_prompt=True, overlap=OVERLAP_SEC,
               max_chars=CUE_MAX_CHARS, max_dur=CUE_MAX_DUR,
               align=ALIGN_TO_SPEECH):
    """转录入口：小文件直接传；>4MB 自动切块（块间重叠 overlap 秒）逐块传。

    流程：词级时间戳 -> 跨块合并去重（_merge_words）-> 音频对齐
    （_align_words：用真实语音区间校正 Whisper 漂移，并在气口打标记）->
    智能分段（_split_cues）-> 返回 segments 形式的 cues（start/end/text）。

    vocab: 引导词表（正确写法，list 或分隔字符串）——注入 Whisper prompt，
            从事后正则热词变成解码期引导，专治同音词漂移。
    use_prompt=False 时完全禁用 prompt（含上文续接）。
    align=False 时完全信任 Whisper 时间戳（旧行为）。
    """
    vocab_text = build_vocab_text(vocab)
    size = os.path.getsize(audio_path)
    if size <= CHUNK_THRESHOLD:
        print("[..] 上传 %.1f MB 到 Groq（代理 %s）..."
              % (size / 1048576.0, proxy or "直连"))
        prompt = build_prompt(vocab_text, None) if use_prompt else None
        if prompt:
            print("[..] 注入识别引导: %s" % prompt[:60])
        result = _upload_one(audio_path, key, model, language, proxy, prompt=prompt)
        words = _result_words(result)
        if not words:
            die("API 未返回词级时间戳: " + json.dumps(result, ensure_ascii=False)[:300])
        words = _align_words(words, audio_path, ffmpeg, enabled=align)
        return {"segments": _split_cues(words, max_chars, max_dur)}

    # ---- 切块路径 ----
    n_chunks = -(-size // CHUNK_THRESHOLD)  # ceil
    print("[..] 文件 %.1f MB 超过 %d MB 单传阈值，切 %d 块（重叠 %.1fs）逐块上传..."
          % (size / 1048576.0, CHUNK_THRESHOLD // 1048576, n_chunks, overlap))
    tmpdir = tempfile.mkdtemp(prefix="groq-srt-chunks-")
    try:
        chunks = split_audio(audio_path, tmpdir, n_chunks, ffmpeg, overlap=overlap)
        all_words = []
        prev_tail = ""
        for i, (fp, offset) in enumerate(chunks, 1):
            csz = os.path.getsize(fp)
            print("[..] 块 %d/%d: %.1f MB（时间偏移 +%.1fs，代理 %s）..."
                  % (i, len(chunks), csz / 1048576.0, offset, proxy or "直连"))
            prompt = build_prompt(vocab_text, prev_tail) if use_prompt else None
            if prompt:
                print("[..] 注入识别引导: %s" % prompt[:60])
            result = _upload_one(fp, key, model, language, proxy, prompt=prompt)
            words = _result_words(result)
            if not words:
                die("块 %d 未返回词级时间戳: %s"
                    % (i, json.dumps(result, ensure_ascii=False)[:200]))
            for w in words:
                w["start"] = w.get("start", 0) + offset
                w["end"] = w.get("end", 0) + offset
                all_words.append(w)
            # 本块全部文本的结尾，作为下一块的 prompt 上文（跨切块不断片）
            prev_tail = "".join((w.get("word") or "").strip()
                                for w in words)[-PROMPT_TAIL_CHARS:]
            print("[ok] 块 %d/%d 完成，累计 %d 个词" % (i, len(chunks), len(all_words)))
        merged = _merge_words(all_words)
        if len(merged) < len(all_words):
            print("[ok] 重叠区去重: %d -> %d 个词" % (len(all_words), len(merged)))
        merged = _align_words(merged, audio_path, ffmpeg, enabled=align)
        cues = _split_cues(merged, max_chars, max_dur)
        print("[ok] 智能分段: %d 个词 -> %d 条字幕" % (len(merged), len(cues)))
        return {"segments": cues}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="音频 -> SRT via Groq Whisper API")
    ap.add_argument("audio", help="输入音频/视频文件")
    ap.add_argument("-o", "--output", default=None, help="输出 SRT 路径（默认与音频同名）")
    ap.add_argument("--model", default="whisper-large-v3",
                    choices=["whisper-large-v3", "whisper-large-v3-turbo"])
    ap.add_argument("--language", default="zh", help="语种提示，中文内容填 zh")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 路径（默认自动探测）")
    ap.add_argument("--vocab", default=None,
                    help="识别引导词表（填正确写法，逗号/空格分隔）。识别时注入 "
                         "Whisper prompt 做解码期引导，与事后正则热词互补，"
                         "专治同音词漂移")
    ap.add_argument("--no-prompt", action="store_true",
                    help="禁用 prompt 引导（含跨切块上文续接，默认开启）")
    ap.add_argument("--overlap", type=float, default=OVERLAP_SEC,
                    help="切块重叠秒数（默认 %.1f，0=不重叠）" % OVERLAP_SEC)
    ap.add_argument("--max-chars", type=int, default=CUE_MAX_CHARS,
                    help="单条字幕最大显示宽度，汉字计 1 英文计 0.55"
                         "（默认 %d，约 16:9 屏幕一行）" % CUE_MAX_CHARS)
    ap.add_argument("--max-dur", type=float, default=CUE_MAX_DUR,
                    help="单条字幕最长秒数（默认 %.1f）" % CUE_MAX_DUR)
    ap.add_argument("--no-align", action="store_true",
                    help="不做音频对齐（完全信任 Whisper 时间戳；"
                         "默认会测真实语音区间校正停顿漂移）")
    ap.add_argument("--keep", action="store_true", help="保留中间转码文件")
    ap.add_argument("--dry-run", action="store_true", help="只转码不上传（验证 ffmpeg 与体积）")
    ap.add_argument("--key", default=None, help="Groq API key（默认读 GROQ_API_KEY）")
    ap.add_argument("--proxy", default=None,
                    help="代理地址。默认 socks5h://127.0.0.1:10808（本机 Shadowsocks）；"
                         "支持 http://host:port；传 none 直连")
    args = ap.parse_args()

    src = os.path.abspath(args.audio)
    if not os.path.isfile(src):
        die("文件不存在: " + src)
    out = os.path.abspath(args.output) if args.output else os.path.splitext(src)[0] + ".srt"
    tmp = os.path.splitext(out)[0] + ".groq.m4a"

    ffmpeg = find_ffmpeg(args.ffmpeg)
    size = transcode(src, tmp, ffmpeg)
    print("[ok] 转码完成: %s (%.1f MB)  ffmpeg=%s" % (tmp, size / 1048576.0, ffmpeg))
    if size > SIZE_LIMIT:
        die("仍然超过 25MB 上限（%.1f MB）。" % (size / 1048576.0))

    if args.dry_run:
        print("[dry-run] 不上传。SRT 目标: " + out)
        return

    key = args.key or os.environ.get("GROQ_API_KEY")
    if not key:
        die("没有 API key。用 --key gsk_... 传入，或设环境变量 GROQ_API_KEY。")
    if key.startswith("xai-"):
        die("这是 xAI(Grok) 的 key。Groq 的 key 是 gsk_ 开头，去 console.groq.com 创建。")

    proxy = (args.proxy or os.environ.get("GROQ_PROXY") or DEFAULT_PROXY)
    if proxy and proxy.lower() in ("none", "direct", "off"):
        proxy = None

    preflight(key, proxy)
    result = transcribe(tmp, key, args.model, args.language, proxy, ffmpeg=ffmpeg,
                        vocab=args.vocab, use_prompt=not args.no_prompt,
                        overlap=args.overlap, max_chars=args.max_chars,
                        max_dur=args.max_dur, align=not args.no_align)
    segs = result.get("segments") or []
    if not segs:
        die("API 未返回段落: " + json.dumps(result, ensure_ascii=False)[:300])

    blocks = []
    for i, seg in enumerate(segs, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        blocks.append("%d\n%s --> %s\n%s\n"
                      % (i, srt_ts(seg.get("start", 0)), srt_ts(seg.get("end", 0)), text))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    if not args.keep:
        try:
            os.remove(tmp)
        except OSError:
            pass
    print("[ok] SRT 已生成: %s（%d 条字幕）" % (out, len(blocks)))
    print("[下一步] capcut --jianying quickstart <草稿名> --audio \"%s\" --srt \"%s\" "
          "--drafts \"%%LOCALAPPDATA%%\\JianyingPro\\User Data\\Projects\\com.lveditor.draft\""
          % (src, out))


if __name__ == "__main__":
    main()
