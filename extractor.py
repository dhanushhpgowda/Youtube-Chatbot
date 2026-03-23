import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

def get_video_id(url):
    """Regex to grab the ID whether it's a long, short, or mobile URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_and_display():
    print("--- YouTube Caption Extractor ---")
    url = input("Paste YouTube Link: ").strip()
    
    video_id = get_video_id(url)
    
    if not video_id:
        print("Error: Could not find a valid Video ID in that link.")
        return

    print(f"Connecting to ID: {video_id}...")

    try:
        # Using the instance method your friend recommended
        api_instance = YouTubeTranscriptApi()
        fetched_transcript = api_instance.fetch(video_id, languages=['en', 'en-GB'])
        
        transcript_list = fetched_transcript.to_raw_data()
        
        print("\n" + "="*30)
        print("TRANSCRIPT START")
        print("="*30 + "\n")

        # Joining chunks into one block of text for the CMD
        full_text = " ".join(chunk["text"] for chunk in transcript_list)
        print(full_text)

        print("\n" + "="*30)
        print("END OF TRANSCRIPT")
        print("="*30)
        
    except TranscriptsDisabled:
        print("\n[!] Error: Captions are disabled for this specific video.")
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_and_display()