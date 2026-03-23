import os
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from utils.processor import get_video_id

load_dotenv()

# --- INITIALIZATION ---
print("Initializing AI models and Vector DB...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:19530")

def start_rag_flow():
    url = input("\nPaste YouTube Link: ").strip()
    video_id = get_video_id(url)
    
    if not video_id:
        print("Invalid YouTube URL.")
        return

    try:
        # 1. Extraction (Using your friend's suggested logic)
        print(f"Extracting transcript for {video_id}...")
        api_instance = YouTubeTranscriptApi()
        fetched_transcript = api_instance.fetch(video_id, languages=['en'])
        transcript_list = fetched_transcript.to_raw_data()
        full_text = " ".join(chunk["text"] for chunk in transcript_list)
        
        # 2. Ingestion (Store in Milvus)
        coll_name = f"yt_{video_id.replace('-', '_')}"
        if client.has_collection(coll_name): client.drop_collection(coll_name)
        client.create_collection(collection_name=coll_name, dimension=768, auto_id=True)
        
        # Chunking text into segments for better retrieval
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 400)]
        data = [{"vector": model.encode(c), "text": c} for c in chunks]
        client.insert(collection_name=coll_name, data=data)
        print("Transcript indexed successfully.")

        # 3. Chat Loop
        while True:
            query = input("\nAsk something about the video (or type 'exit'): ")
            if query.lower() == 'exit': break
            
            # Search Milvus for relevant context
            search_res = client.search(
                collection_name=coll_name,
                data=model.encode([query]),
                limit=3,
                output_fields=["text"]
            )
            context = "\n".join([r['entity']['text'] for r in search_res[0]])
            
            # Query Groq LLM (Llama 3.3)
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": f"Answer based ONLY on this: {context}"},
                    {"role": "user", "content": query}
                ]
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                json=payload
            )
            print(f"\nAI: {resp.json()['choices'][0]['message']['content']}")

    except TranscriptsDisabled:
        print("Error: Captions are disabled for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_rag_flow()