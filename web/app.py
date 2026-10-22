"""One-page live view: replay a meeting script through the gate and stream events over SSE.

One meeting runs at a time in a background thread (the decider is best driven with one
request at a time). Every event from `kuuki.transcript.meeting_events` is kept in memory and
streamed to any open page, so a reload picks the meeting up from the start.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kuuki.client import Decider
from kuuki.gate import InterjectionGate
from kuuki.speaker import Speaker
from kuuki.transcript import load_script, meeting_events

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "dev": ROOT / "scripts/dev.jsonl",
    "eval": ROOT / "scripts/eval.jsonl",
    "smoke": ROOT / "scripts/smoke.jsonl",
}
STATIC = Path(__file__).resolve().parent / "static"


class _Stopped(Exception):
    pass


class Session:
    def __init__(self, script_name: str, speak: bool, realtime: bool) -> None:
        self.id = id(self)
        self.script_name = script_name
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._resume = threading.Event()
        self._resume.set()
        self._stop = False
        script = load_script(SCRIPTS[script_name])
        self.gate = InterjectionGate(script.context, Decider())
        self.speaker = Speaker(script.meeting, script.reference, speak_aloud=speak)
        self._script = script
        self._realtime = realtime
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def start(self) -> None:
        self._thread.start()

    def _before_line(self) -> None:
        self._resume.wait()
        if self._stop:
            raise _Stopped

    def _run(self) -> None:
        try:
            for event in meeting_events(
                self._script, self.gate, self.speaker, realtime=self._realtime, before_line=self._before_line
            ):
                self.events.append(event)
        except _Stopped:
            self.events.append({"type": "stopped"})
        except Exception as exc:  # noqa: BLE001 -- surface failures on the page instead of a silent stall
            self.events.append({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.done = True

    def pause(self) -> None:
        self._resume.clear()
        self.events.append({"type": "paused"})

    def resume(self) -> None:
        self.events.append({"type": "resumed"})
        self._resume.set()

    def stop(self) -> None:
        self._stop = True
        self._resume.set()


app = FastAPI(title="Kuuki")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_session: Session | None = None
_lock = threading.Lock()


class StartRequest(BaseModel):
    script: str = "dev"
    speak: bool = True
    realtime: bool = True


class SpeakRequest(BaseModel):
    on: bool


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/scripts")
def scripts() -> list[str]:
    return [name for name, path in SCRIPTS.items() if path.exists()]


@app.post("/api/start")
def start(req: StartRequest) -> dict[str, Any]:
    global _session
    if req.script not in SCRIPTS or not SCRIPTS[req.script].exists():
        raise HTTPException(404, f"unknown script {req.script!r}")
    with _lock:
        if _session is not None and not _session.done:
            _session.stop()
        _session = Session(req.script, req.speak, req.realtime)
        _session.start()
        return {"session": _session.id}


def _current() -> Session:
    if _session is None:
        raise HTTPException(409, "no meeting started")
    return _session


@app.post("/api/pause")
def pause() -> dict[str, bool]:
    s = _current()
    if not s.paused and not s.done:
        s.pause()
    return {"paused": s.paused}


@app.post("/api/resume")
def resume() -> dict[str, bool]:
    s = _current()
    if s.paused:
        s.resume()
    return {"paused": s.paused}


@app.post("/api/speak")
def speak(req: SpeakRequest) -> dict[str, bool]:
    s = _current()
    s.speaker.speak_aloud = req.on
    return {"on": s.speaker.speak_aloud}


@app.get("/api/events")
async def events() -> StreamingResponse:
    session = _current()

    async def stream():
        sent = 0
        yield f"event: session\ndata: {json.dumps({'session': session.id, 'script': session.script_name})}\n\n"
        while True:
            while sent < len(session.events):
                yield f"data: {json.dumps(session.events[sent])}\n\n"
                sent += 1
            if session.done and sent >= len(session.events):
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
