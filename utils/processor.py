import re

def get_video_id(url):
    """Extracts the unique 11-character Video ID from any YouTube link."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None