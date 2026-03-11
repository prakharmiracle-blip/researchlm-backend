"""
ResearchLM Backend — Railway
Handles YouTube scraping (yt-dlp) and NotebookLM pipeline
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess, sys, json, asyncio, os, tempfile, base64
from pathlib import Path

app = FastAPI(title="ResearchLM Backend")

# Allow requests from your Netlify site (update with your real domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replace * with your Netlify URL after deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class YouTubeRequest(BaseModel):
    topic: str
    count: int = 25

class NotebookLMRequest(BaseModel):
    topic: str
    urls: list[str]
    analysis: bool = True
    infographic: bool = True
    slides: bool = False
    flashcards: bool = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def format_duration(seconds):
    if not seconds:
        return "unknown"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def format_views(n):
    if not n:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ResearchLM backend running ✓"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/youtube")
def youtube_search(req: YouTubeRequest):
    """Search YouTube and return video metadata using yt-dlp"""
    if not req.topic.strip():
        raise HTTPException(400, "Topic cannot be empty")

    count = max(1, min(req.count, 50))
    query = f"{req.topic} 2025"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        "--no-warnings",
        f"ytsearch{count}:{query}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "YouTube search timed out — try fewer videos")

    if result.returncode != 0 and not result.stdout:
        raise HTTPException(502, f"yt-dlp error: {result.stderr[:300]}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            videos.append({
                "title":      d.get("title", "Unknown"),
                "url":        d.get("webpage_url") or f"https://youtube.com/watch?v={d.get('id','')}",
                "channel":    d.get("uploader") or d.get("channel", "Unknown"),
                "views":      d.get("view_count", 0),
                "duration_s": d.get("duration", 0),
                "duration":   format_duration(d.get("duration", 0)),
                "upload_date":d.get("upload_date", ""),
                "thumbnail":  d.get("thumbnail", ""),
                "description":(d.get("description") or "")[:300],
                "id":         d.get("id", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return {"videos": videos, "count": len(videos), "topic": req.topic}


@app.post("/api/notebooklm")
async def notebooklm_pipeline(req: NotebookLMRequest):
    """
    Full NotebookLM pipeline:
    1. Create notebook
    2. Add YouTube URLs as sources
    3. Get analysis
    4. Generate requested artifacts
    """
    from notebooklm import NotebookLMClient

    if not req.urls:
        raise HTTPException(400, "No URLs provided")

    result = {
        "notebook_id": None,
        "analysis": None,
        "infographic_url": None,
        "slides_url": None,
        "flashcards_url": None,
    }

    try:
        async with await NotebookLMClient.from_storage() as client:

            # 1. Create notebook
            nb = await client.notebooks.create(f"Research: {req.topic}")
            result["notebook_id"] = nb.id

            # 2. Add YouTube sources (up to 25 — NotebookLM limit)
            urls_to_add = req.urls[:25]
            for url in urls_to_add:
                try:
                    await client.sources.add_url(nb.id, url, wait=True)
                except Exception:
                    pass  # Skip failed sources, continue with rest

            # 3. Analysis
            if req.analysis:
                chat_result = await client.chat.ask(
                    nb.id,
                    f"What are the top 5–7 key findings, themes, and trends across these "
                    f"YouTube videos about '{req.topic}'? Be specific and highlight patterns."
                )
                result["analysis"] = chat_result.answer

            # 4. Infographic
            if req.infographic:
                status = await client.artifacts.generate_infographic(
                    nb.id, orientation="portrait", detail_level="detailed"
                )
                await client.artifacts.wait_for_completion(nb.id, status.task_id)

                # Save to temp file and return as base64 data URL
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name

                await client.artifacts.download_infographic(nb.id, tmp_path)
                img_data = Path(tmp_path).read_bytes()
                b64 = base64.b64encode(img_data).decode()
                result["infographic_url"] = f"data:image/png;base64,{b64}"
                os.unlink(tmp_path)

            # 5. Slides
            if req.slides:
                status = await client.artifacts.generate_slide_deck(nb.id)
                await client.artifacts.wait_for_completion(nb.id, status.task_id)
                result["slides_url"] = f"/api/download/slides/{nb.id}"

            # 6. Flashcards
            if req.flashcards:
                status = await client.artifacts.generate_flashcards(nb.id)
                await client.artifacts.wait_for_completion(nb.id, status.task_id)
                result["flashcards_url"] = f"/api/download/flashcards/{nb.id}"

    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "login" in error_msg.lower() or "401" in error_msg:
            raise HTTPException(
                401,
                "NotebookLM authentication failed. Run 'notebooklm login' on the server."
            )
        raise HTTPException(500, f"NotebookLM pipeline error: {error_msg[:300]}")

    return result
