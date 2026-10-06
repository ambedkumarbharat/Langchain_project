import re
from urllib.parse import urlparse, parse_qs

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Extract 11-character YouTube video ID from URL or ID."""

    value = url_or_id.strip()

    if _VIDEO_ID_RE.match(value):
        return value

    if not value.startswith(("http://", "https://")):
        value = "https://" + value

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().removeprefix("www.")

    video_id = None

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]

    elif host in (
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
    ):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]

        else:
            parts = [p for p in parsed.path.split("/") if p]

            if (
                len(parts) >= 2
                and parts[0] in ("shorts", "embed", "live", "v")
            ):
                video_id = parts[1]

    if video_id and _VIDEO_ID_RE.match(video_id):
        return video_id

    raise ValueError(
        f"Valid YouTube video ID nahi mila: {url_or_id!r}"
    )