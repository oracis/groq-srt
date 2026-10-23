#!/usr/bin/env python3
"""test_groq_srt.py - groq_srt.py 本地端到端测试（不联网、不需要真 key）。

起本地 HTTPS 服务器冒充 Groq API：
  GET  /openai/v1/models               -> 200 {"data":[...]}
  POST /openai/v1/audio/transcriptions  -> 200 verbose_json（canned segments）
  POST（坏 key，drain 模式）            -> 401（读完 body 才回，模拟正常拒绝）
  POST（坏 key，abort 模式）            -> 读 64KB 就回 401 并断开（模拟提前拒绝）
monkeypatch groq_srt.API_HOST/API_PORT 指向本地并关闭 TLS 校验，跑真实代码路径。
"""
import io
import json
import math
import os
import shutil
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groq_srt

PORT = 8443
GOOD_KEY = "gsk_test_good"
CANNED = {
    "task": "transcribe",
    "language": "zh",
    "duration": 29.47,
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "大家好，欢迎收看本期视频"},
        {"start": 2.5, "end": 5.0, "text": "今天我们来聊聊命令行剪辑"},
    ],
    "text": "大家好，欢迎收看本期视频 今天我们来聊聊命令行剪辑",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    abort_early = False  # True: 读 64KB 就回 401 并断开（模拟 Groq 提前拒绝）

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == groq_srt.MODELS_PATH:
            body = json.dumps({"data": [{"id": "whisper-large-v3"},
                                        {"id": "whisper-large-v3-turbo"}]}).encode()
            self._json(200, body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != groq_srt.API_PATH:
            self.send_error(404)
            return
        if self.headers.get("Authorization", "") != "Bearer " + GOOD_KEY:
            if self.abort_early:
                self.rfile.read(65536)  # 只读一小段，制造"提前断连"
            else:
                self._drain_body()
            body = json.dumps({"error": {"message": "Invalid API Key",
                                         "code": "invalid_api_key"}}).encode()
            self._json(401, body)
            if self.abort_early:
                self.close_connection = True
            return
        data = self._drain_body()
        assert b'name="model"' in data and b"whisper-large-v3" in data, "multipart 缺 model"
        assert b'name="response_format"' in data and b"verbose_json" in data, "缺 response_format"
        assert b'name="language"' in data and b"zh" in data, "缺 language"
        assert b'name="file"' in data and b"audio/mp4" in data, "缺 file 字段"
        assert b'name="timestamp_granularities[]"' in data, "缺时间戳粒度字段"
        assert data.rstrip().endswith(b"--"), "multipart 未正确结束"
        self._json(200, json.dumps(CANNED).encode())

    def _drain_body(self):
        clen = int(self.headers.get("Content-Length", 0))
        chunks, remaining = [], clen
        while remaining > 0:
            c = self.rfile.read(min(65536, remaining))
            if not c:
                break
            chunks.append(c)
            remaining -= len(c)
        return b"".join(chunks)

    def _json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_cert(tmpdir):
    key = os.path.join(tmpdir, "key.pem")
    cert = os.path.join(tmpdir, "cert.pem")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", key, "-out", cert, "-days", "2", "-nodes",
                    "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, capture_output=True)
    return key, cert


def start_server(key, cert):
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def patch_tls():
    _orig = ssl.create_default_context

    def _no_verify(*a, **k):
        c = _orig(*a, **k)
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c
    # 注意：Python 3.15 的 HTTPSConnection 不走 ssl.create_default_context，
    # 必须 patch groq_srt.make_client_context（_request 显式传 context=）。
    groq_srt.make_client_context = _no_verify
    groq_srt.API_HOST = "127.0.0.1"
    groq_srt.API_PORT = PORT


def make_wav(path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000)))
            for i in range(16000)))


def make_dummy(path, size):
    with open(path, "wb") as f:
        f.write(os.urandom(size))


def run_expect_exit(fn, any_of=(), all_of=()):
    """跑 fn()，期望 SystemExit(1) 且 stderr 是 [error] 开头的干净报错。
    all_of 必须全部出现；any_of 至少出现一个。"""
    err = io.StringIO()
    old = sys.stderr
    sys.stderr = err
    try:
        fn()
        raise AssertionError("应当以 SystemExit 退出，但没有")
    except SystemExit as e:
        assert e.code == 1, "退出码 %r" % (e.code,)
    finally:
        sys.stderr = old
    msg = err.getvalue()
    assert msg.startswith("[error]"), "报错格式异常: %s" % msg
    for n in all_of:
        assert n in msg, "报错信息缺少 %r，实际: %s" % (n, msg)
    if any_of:
        assert any(n in msg for n in any_of), \
            "报错信息缺少任一 %r，实际: %s" % (any_of, msg)
    return msg


def test_full_flow(tmpdir):
    wav = os.path.join(tmpdir, "tone.wav")
    make_wav(wav)
    srt = os.path.join(tmpdir, "out.srt")
    argv, sys.argv = sys.argv, ["groq_srt.py", wav, "-o", srt,
                                "--key", GOOD_KEY, "--proxy", "none"]
    try:
        groq_srt.main()
    finally:
        sys.argv = argv
    text = open(srt, encoding="utf-8").read()
    assert "1\n00:00:00,000 --> 00:00:02,500\n大家好，欢迎收看本期视频\n" in text, text
    assert "2\n00:00:02,500 --> 00:00:05,000\n今天我们来聊聊命令行剪辑\n" in text, text
    assert not os.path.exists(os.path.join(tmpdir, "out.groq.m4a")), "临时 m4a 未清理"


def test_bad_key_full_body(tmpdir):
    Handler.abort_early = False
    dummy = os.path.join(tmpdir, "dummy.m4a")
    make_dummy(dummy, 2 * 1024 * 1024)
    msg = run_expect_exit(lambda: groq_srt.transcribe(
        dummy, "gsk_test_bad", "whisper-large-v3", "zh", None),
        all_of=("401", "key"))
    print("    -> " + msg.strip().splitlines()[0])


def test_bad_key_early_abort(tmpdir):
    Handler.abort_early = True
    dummy = os.path.join(tmpdir, "dummy2.m4a")
    make_dummy(dummy, 2 * 1024 * 1024)
    # 服务器读 64KB 就回 401 并带着未读数据关闭（RST）。
    # 后台线程发 body + 主线程专职读响应，应当能抢回 401；万一被 RST 抢先，
    # 也必须是"连接被中止"这种干净报错，不能是 traceback。
    msg = run_expect_exit(lambda: groq_srt.transcribe(
        dummy, "gsk_test_bad", "whisper-large-v3", "zh", None),
        any_of=("401", "中止"))
    print("    -> " + msg.strip().splitlines()[0])


def main():
    tmp = tempfile.mkdtemp(prefix="groqsrt_test_")
    try:
        key, cert = make_cert(tmp)
        httpd = start_server(key, cert)
        try:
            patch_tls()
            test_full_flow(tmp)
            print("PASS 1/3 完整流程（转码 -> 预检 -> 上传 -> SRT）")
            test_bad_key_full_body(tmp)
            print("PASS 2/3 坏 key 完整体拒绝（401 干净退出）")
            test_bad_key_early_abort(tmp)
            print("PASS 3/3 坏 key 提前断连（拿到 401 或友好中止，无 traceback）")
        finally:
            httpd.shutdown()
        print("ALL PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
