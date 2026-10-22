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
    body += field("timestamp_granularities[]", "segment")
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
               vocab=None, use_prompt=True, overlap=OVERLAP_SEC):
    """转录入口：小文件直接传；>4MB 自动切块（块间重叠 overlap 秒）逐块传。

    vocab: 引导词表（正确写法，list 或分隔字符串）——注入 Whisper prompt，
            从事后正则热词变成解码期引导，专治同音词漂移。
    use_prompt=False 时完全禁用 prompt（含上文续接）。
    """
    vocab_text = build_vocab_text(vocab)
    size = os.path.getsize(audio_path)
    if size <= CHUNK_THRESHOLD:
        print("[..] 上传 %.1f MB 到 Groq（代理 %s）..."
              % (size / 1048576.0, proxy or "直连"))
        prompt = build_prompt(vocab_text, None) if use_prompt else None
        if prompt:
            print("[..] 注入识别引导: %s" % prompt[:60])
        return _upload_one(audio_path, key, model, language, proxy, prompt=prompt)

    # ---- 切块路径 ----
    n_chunks = -(-size // CHUNK_THRESHOLD)  # ceil
    print("[..] 文件 %.1f MB 超过 %d MB 单传阈值，切 %d 块（重叠 %.1fs）逐块上传..."
          % (size / 1048576.0, CHUNK_THRESHOLD // 1048576, n_chunks, overlap))
    tmpdir = tempfile.mkdtemp(prefix="groq-srt-chunks-")
    try:
        chunks = split_audio(audio_path, tmpdir, n_chunks, ffmpeg, overlap=overlap)
        all_segs = []
        prev_tail = ""
        for i, (fp, offset) in enumerate(chunks, 1):
            csz = os.path.getsize(fp)
            print("[..] 块 %d/%d: %.1f MB（时间偏移 +%.1fs，代理 %s）..."
                  % (i, len(chunks), csz / 1048576.0, offset, proxy or "直连"))
            prompt = build_prompt(vocab_text, prev_tail) if use_prompt else None
            if prompt:
                print("[..] 注入识别引导: %s" % prompt[:60])
            result = _upload_one(fp, key, model, language, proxy, prompt=prompt)
            segs = result.get("segments") or []
            if not segs:
                die("块 %d 未返回段落: %s"
                    % (i, json.dumps(result, ensure_ascii=False)[:200]))
            for seg in segs:
                seg["start"] = seg.get("start", 0) + offset
                seg["end"] = seg.get("end", 0) + offset
                all_segs.append(seg)
            # 本块全部文本的结尾，作为下一块的 prompt 上文（跨切块不断片）
            prev_tail = "".join((s.get("text") or "").strip()
                                for s in segs)[-PROMPT_TAIL_CHARS:]
            print("[ok] 块 %d/%d 完成，累计 %d 条段落" % (i, len(chunks), len(all_segs)))
        merged = _merge_segments(all_segs)
        if len(merged) < len(all_segs):
            print("[ok] 重叠区去重: %d -> %d 条" % (len(all_segs), len(merged)))
        return {"segments": merged}
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
                        overlap=args.overlap)
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
