"""
Live, steerable Causal Forcing++ stream in the browser.

    python live.py --prompt "A first-person drone flight over a misty pine forest at sunrise"
    then open http://localhost:8000 and type prompts to steer.

A generation thread keeps a frame buffer topped up; a playback thread releases
frames at 16 fps as an MJPEG stream (shown by a plain <img>). New prompts are
applied between frames, so the video keeps flowing while it changes direction.
"""
import argparse
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import torch

import weights
from stream import LAUNCH_DIR, LORA_PRESETS, PLAYBACK_FPS, CausalStream

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>InfiniteStream</title>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:#0b0d14;color:#e6e8ef;font:15px/1.45 system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:0 16px 24px}
header,form,#info,#frame{width:min(100%,1040px)}
header{display:flex;align-items:center;justify-content:space-between;padding:16px 0 12px}
h1{margin:0;font-size:18px;font-weight:650}
#live{display:flex;align-items:center;gap:7px;color:#ff6b6b;font-size:12px;font-weight:700;letter-spacing:.08em}
#live i{width:8px;height:8px;border-radius:50%;background:#ff5252;animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.3}}
#frame{border-radius:12px;overflow:hidden;background:#000;box-shadow:0 12px 48px #0009;border:1px solid #232838}
img{display:block;width:100%;aspect-ratio:832/480}
form{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 10px}
#p{flex:1 1 320px;padding:12px 14px;font:inherit;background:#151926;color:#eef0f6;border:1px solid #2b3145;border-radius:10px;outline:0}
#p:focus{border-color:#6c8cff;box-shadow:0 0 0 3px #6c8cff33}
button{padding:12px 20px;font:inherit;font-weight:600;border-radius:10px;border:0;background:#5b7cff;color:#fff;cursor:pointer}
button:hover{background:#7290ff}
label{display:flex;align-items:center;gap:6px;color:#aab1c4;white-space:nowrap;cursor:pointer}
#info{color:#8a92a8;font-size:13px;white-space:pre-wrap}
</style></head><body>
<header><h1>InfiniteStream</h1><div id="live"><i></i>LIVE</div></header>
<div id="frame"><img src="/stream" alt="live stream"></div>
<form id="f"><input id="p" autocomplete="off" autofocus placeholder="Describe what should happen next, then press Enter">
<label title="Start over from this prompt like a new video (hard cut, nothing carried over), then continue steering as usual"><input type="checkbox" id="reset">Reset canvas</label>
<button>Steer</button></form>
<div id="info"></div>
<script>
const form = document.getElementById("f"), box = document.getElementById("p"), reset = document.getElementById("reset");
form.onsubmit = async e => { e.preventDefault(); if (!box.value.trim()) return;
  await fetch('/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: box.value, reset: reset.checked})}); box.value = ''; };
setInterval(async () => { try { const s = await (await fetch('/status')).json();
  document.getElementById("info").textContent = `generating ${s.gen_fps.toFixed(1)} fps (need ${s.playback_fps}) | buffer ${s.buffer_s.toFixed(2)} s | ` +
    `stalls ${s.stalls} | model ${s.model} | ${s.pending ? 'switching…' : ''}\\nprompt: ${s.prompt}`; } catch (e) {} }, 500);
</script></body></html>"""


class Live:
    def __init__(self, stream, prompt, seed, start_buffer_s, max_buffer_s, model_name="vanilla"):
        self.stream, self.prompt, self.seed, self.model_name = stream, prompt, seed, model_name
        self.start_frames = round(start_buffer_s * PLAYBACK_FPS)
        self.max_frames = round(max_buffer_s * PLAYBACK_FPS)
        self.buffer = collections.deque()
        self.pending = None
        self.jpeg, self.jpeg_ready = None, threading.Condition()
        self.stalls, self.gen_fps, self.resets = 0, 0.0, 0
        self.lock = threading.Lock()

    def submit(self, prompt, reset=False):
        with self.lock:
            self.pending = (prompt, reset)

    def generate(self):
        s = self.stream
        s.reset(seed=self.seed)
        s.set_prompt(s.encode([self.prompt], offload=False)[0])
        recent = collections.deque(maxlen=20)
        while True:
            with self.lock:
                request, self.pending = self.pending, None
            if request:
                prompt, reset = request
                cond = s.encode([prompt], offload=False)[0]
                if reset:  # hard cut: empty cache, the next frame is a new opening frame
                    self.resets += 1
                    s.reset(seed=self.seed + self.resets)
                s.set_prompt(cond)
                self.prompt = prompt
                print(f"{'reset' if reset else 'prompt'} -> {prompt}")
            while len(self.buffer) >= self.max_frames:  # bounded latency for new prompts
                time.sleep(0.01)
            t = time.perf_counter()
            frames = s.step().numpy()
            recent.append((len(frames), time.perf_counter() - t))
            self.gen_fps = sum(n for n, _ in recent) / sum(dt for _, dt in recent)
            self.buffer.extend(frames)

    def play(self):
        while len(self.buffer) < self.start_frames:
            time.sleep(0.01)
        next_t, frame = time.perf_counter(), None
        while True:
            if self.buffer:
                frame = self.buffer.popleft()
            else:
                self.stalls += 1  # nothing ready: hold the last frame
            if frame is not None:
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self.jpeg_ready:
                    self.jpeg = jpg.tobytes()
                    self.jpeg_ready.notify_all()
            next_t += 1 / PLAYBACK_FPS
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def status(self):
        return {"gen_fps": self.gen_fps, "playback_fps": PLAYBACK_FPS, "buffer_s": len(self.buffer) / PLAYBACK_FPS,
                "stalls": self.stalls, "pending": self.pending is not None, "prompt": self.prompt,
                "model": self.model_name,
                "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)}


def make_handler(live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/status":
                self._send(json.dumps(live.status()).encode(), "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with live.jpeg_ready:
                            live.jpeg_ready.wait()
                            jpg = live.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/prompt":
                return self.send_error(404)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            try:  # {"prompt": ..., "reset": bool} from the page; plain text also works
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict):
                data = {"prompt": body}
            prompt = str(data.get("prompt", "")).strip()
            if prompt:
                live.submit(prompt, reset=bool(data.get("reset")))
            self._send(b"ok", "text/plain")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A first-person drone flight over a misty pine forest at sunrise, "
                                            "the camera gliding steadily forward between the trees, cinematic lighting.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--sink", type=int, default=1)
    parser.add_argument("--switch", default="flush+resink",
                        choices=["none", "flush", "recache", "flush+recache", "resink", "flush+resink"])
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--guidance_s", type=float, default=3.0)
    parser.add_argument("--refresh_s", type=float, default=8.0, help="regenerate the sink every N seconds (0 = off)")
    parser.add_argument("--lora", action="append", default=[],
                        help="PATH[:SCALE] of a Wan 2.1 T2V 1.3B LoRA or full fine-tune, or a preset name from "
                             "loras/presets.json; repeatable. Without it the stock weights run.")
    parser.add_argument("--start_buffer", type=float, default=0.5, help="seconds buffered before playback starts")
    parser.add_argument("--max_buffer", type=float, default=1.5, help="cap on buffered seconds (prompt latency)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    stream = CausalStream(steps=args.steps, window=args.window, sink=args.sink, switch_mode=args.switch,
                          keep=args.keep, guidance=args.guidance,
                          guidance_frames=round(args.guidance_s * PLAYBACK_FPS / 4),
                          refresh_frames=round(args.refresh_s * PLAYBACK_FPS / 4),
                          loras=weights.parse_specs(args.lora, LORA_PRESETS, LAUNCH_DIR))
    print("warming up (compiling)...")
    cond = stream.encode([args.prompt], offload=False)[0]
    stream.reset(seed=args.seed + 1000)
    stream.set_prompt(cond)
    for _ in range(args.window + 8):
        stream.step()
    stream.set_prompt(cond)  # exercise the switch path too
    stream.step()
    torch.cuda.empty_cache()  # drop compile-time scratch memory before going live

    live = Live(stream, args.prompt, args.seed, args.start_buffer, args.max_buffer,
                model_name=" + ".join(args.lora) or "vanilla")
    threading.Thread(target=live.generate, daemon=True).start()
    threading.Thread(target=live.play, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(live))
    print(f"live on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
