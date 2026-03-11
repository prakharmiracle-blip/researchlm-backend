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
    """Search YouTube using the official YouTube Data API v3"""
    import httpx

    if not req.topic.strip():
        raise HTTPException(400, "Topic cannot be empty")

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise HTTPException(500, "YouTube API key not configured on server")

    count = max(1, min(req.count, 50))

    # YouTube Data API v3 search
    try:
        with httpx.Client(timeout=30) as client:
            # Step 1: Search for video IDs
            search_resp = client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": req.topic,
                    "type": "video",
                    "maxResults": min(count, 50),
                    "order": "relevance",
                    "key": api_key,
                }
            )

            if search_resp.status_code != 200:
                error = search_resp.json().get("error", {}).get("message", "Unknown error")
                raise HTTPException(502, f"YouTube API error: {error}")

            search_data = search_resp.json()
            items = search_data.get("items", [])

            if not items:
                return {"videos": [], "count": 0, "topic": req.topic}

            # Step 2: Get video statistics (views, duration)
            video_ids = [item["id"]["videoId"] for item in items]
            stats_resp = client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "statistics,contentDetails",
                    "id": ",".join(video_ids),
                    "key": api_key,
                }
            )

            stats_map = {}
            if stats_resp.status_code == 200:
                for v in stats_resp.json().get("items", []):
                    vid_id = v["id"]
                    stats = v.get("statistics", {})
                    details = v.get("contentDetails", {})
                    stats_map[vid_id] = {
                        "views": int(stats.get("viewCount", 0)),
                        "duration_iso": details.get("duration", ""),
                    }

            # Step 3: Build response
            videos = []
            for item in items:
                vid_id = item["id"]["videoId"]
                snippet = item.get("snippet", {})
                extra = stats_map.get(vid_id, {})
                duration_s = parse_iso_duration(extra.get("duration_iso", ""))

                videos.append({
                    "title":       snippet.get("title", "Unknown"),
                    "url":         f"https://www.youtube.com/watch?v={vid_id}",
                    "channel":     snippet.get("channelTitle", "Unknown"),
                    "views":       extra.get("views", 0),
                    "duration_s":  duration_s,
                    "duration":    format_duration(duration_s),
                    "upload_date": snippet.get("publishedAt", "")[:10],
                    "thumbnail":   snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "description": snippet.get("description", "")[:300],
                    "id":          vid_id,
                })

    except httpx.TimeoutException:
        raise HTTPException(504, "YouTube API timed out — please try again")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"YouTube search error: {str(e)[:200]}")

    return {"videos": videos, "count": len(videos), "topic": req.topic}


def parse_iso_duration(duration: str) -> int:
    """Convert ISO 8601 duration (PT1H2M3S) to seconds"""
    import re
    if not duration:
        return 0
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 3600 + m * 60 + s


@app.get("/api/test-youtube")
def test_youtube():
    """Debug endpoint to test YouTube API key"""
    import httpx
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return {"error": "YOUTUBE_API_KEY not set"}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "q": "test", "maxResults": 1, "key": api_key}
            )
            if resp.status_code == 200:
                return {"status": "YouTube API working ✓"}
            return {"error": resp.json().get("error", {}).get("message", "Unknown")}
    except Exception as e:
        return {"error": str(e)}


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

    # Write auth token from env variable to disk so NotebookLM can read it
    auth_json = os.environ.get("NOTEBOOKLM_AUTH_JSON")
    if auth_json:
        auth_dir = Path.home() / ".notebooklm"
        auth_dir.mkdir(parents=True, exist_ok=True)
        (auth_dir / "storage_state.json").write_text(auth_json)
        print("Auth token written to disk from environment variable")
    else:
        raise HTTPException(401, "NOTEBOOKLM_AUTH_JSON not set in Render environment variables")

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
