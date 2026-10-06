"""YouTube URL to transcript loader module."""

import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from .url_to_id import extract_video_id


logger = logging.getLogger(__name__)


def get_transcript(
    url_or_id: str,
    preferred_langs=("hi", "en"),
):
    """Fetch YouTube transcript and return normalized transcript data."""

    video_id = extract_video_id(url_or_id)

    api = YouTubeTranscriptApi()

    try:
        available = list(api.list(video_id))

    except TranscriptsDisabled:
        logger.warning(
            "%s: captions disabled or unavailable",
            video_id,
        )
        return None

    except VideoUnavailable:
        logger.warning(
            "%s: video unavailable",
            video_id,
        )
        return None

    except NoTranscriptFound:
        logger.warning(
            "%s: no transcript found",
            video_id,
        )
        return None

    if not available:
        return None

    def pick_transcript():
        # First preference:
        # manually created transcript in preferred language
        for want_generated in (False, True):
            for lang in preferred_langs:
                for transcript in available:
                    if (
                        transcript.language_code == lang
                        and transcript.is_generated == want_generated
                    ):
                        return transcript

        # Otherwise manually created transcript
        for transcript in available:
            if not transcript.is_generated:
                return transcript

        # Last fallback
        return available[0]

    chosen = pick_transcript()

    try:
        fetched = chosen.fetch()
    except Exception as exc:
        logger.exception(
            "%s: transcript fetch failed: %s",
            video_id,
            exc,
        )
        return None

    return {
        "video_id": video_id,
        "language": chosen.language_code,
        "is_generated": chosen.is_generated,
        "page_content": " ".join(
            snippet.text for snippet in fetched.snippets
        ),
    }