import os
import sqlite3
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from utils.processor import get_video_id

load_dotenv()

app = Flask(__name__)
DB_PATH = "chat_history.db"

# --- INITIALIZATION ---
print("Initializing AI models and Vector DB...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
milvus_client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:19530")


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            video_url TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    conn = get_db()
    sessions = conn.execute(
        'SELECT * FROM sessions ORDER BY created_at DESC'
    ).fetchall()
    result = []
    for s in sessions:
        msg_count = conn.execute(
            'SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?', (s['id'],)
        ).fetchone()['cnt']
        result.append({
            'id': s['id'],
            'video_id': s['video_id'],
            'video_url': s['video_url'],
            'title': s['title'] or f"Video: {s['video_id']}",
            'created_at': s['created_at'],
            'message_count': msg_count
        })
    conn.close()
    return jsonify(result)


@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id):
    conn = get_db()
    session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    messages = conn.execute(
        'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC', (session_id,)
    ).fetchall()
    conn.close()
    return jsonify({
        'id': session['id'],
        'video_id': session['video_id'],
        'video_url': session['video_url'],
        'title': session['title'],
        'created_at': session['created_at'],
        'messages': [dict(m) for m in messages]
    })


@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = get_db()
    conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/load_video', methods=['POST'])
def load_video():
    data = request.json
    url = data.get('url', '').strip()
    video_id = get_video_id(url)

    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL. Could not extract video ID.'}), 400

    try:
        print(f"Extracting transcript for {video_id}...")
        api_instance = YouTubeTranscriptApi()

        # ── Smart transcript fetching ──
        # Lists ALL available transcripts (manual + auto-generated)
        transcript_list_obj = api_instance.list(video_id)

        fetched_transcript = None
        used_lang = None

        # 1. Try manual English transcripts first
        for lang in ['en', 'en-GB', 'en-US']:
            try:
                fetched_transcript = api_instance.fetch(video_id, languages=[lang])
                used_lang = lang
                print(f"  → Found manual transcript: {lang}")
                break
            except Exception:
                continue

        # 2. Fall back to ANY auto-generated transcript
        if fetched_transcript is None:
            for t in transcript_list_obj:
                try:
                    fetched_transcript = t.fetch()
                    used_lang = t.language_code
                    print(f"  → Using {'auto-generated' if t.is_generated else 'manual'} transcript: {t.language_code} ({t.language})")
                    break
                except Exception:
                    continue

        if fetched_transcript is None:
            return jsonify({'error': 'No transcript available for this video (tried all languages).'}), 400

        transcript_data = fetched_transcript.to_raw_data()
        full_text = " ".join(chunk["text"] for chunk in transcript_data)
        print(f"  → Transcript length: {len(full_text)} chars, {len(transcript_data)} segments")

        # Store in Milvus
        coll_name = f"yt_{video_id.replace('-', '_')}"
        if milvus_client.has_collection(coll_name):
            milvus_client.drop_collection(coll_name)
        milvus_client.create_collection(collection_name=coll_name, dimension=768, auto_id=True)
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 400)]
        embed_data = [{"vector": model.encode(c).tolist(), "text": c} for c in chunks]
        milvus_client.insert(collection_name=coll_name, data=embed_data)

        # Get video title from oEmbed
        title = None
        try:
            oembed = requests.get(
                f"https://www.youtube.com/oembed?url={url}&format=json", timeout=5
            ).json()
            title = oembed.get('title')
        except Exception:
            title = f"Video {video_id}"

        # Save session to DB
        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO sessions (video_id, video_url, title) VALUES (?, ?, ?)',
            (video_id, url, title)
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'session_id': session_id,
            'video_id': video_id,
            'title': title,
            'chunk_count': len(chunks),
            'transcript_lang': used_lang
        })

    except TranscriptsDisabled:
        return jsonify({'error': 'Transcripts are completely disabled for this video by the uploader.'}), 400
    except Exception as e:
        print(f"  → Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query', '').strip()

    if not session_id or not query:
        return jsonify({'error': 'Missing session_id or query'}), 400

    conn = get_db()
    session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not session:
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    video_id = session['video_id']
    coll_name = f"yt_{video_id.replace('-', '_')}"

    # Save user message
    conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                 (session_id, 'user', query))
    conn.commit()

    try:
        # Search Milvus for relevant context
        search_res = milvus_client.search(
            collection_name=coll_name,
            data=model.encode([query]).tolist(),
            limit=3,
            output_fields=["text"]
        )
        context = "\n".join([r['entity']['text'] for r in search_res[0]])

        # Query Groq LLM
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant. Answer based ONLY on this transcript context:\n\n{context}"},
                {"role": "user", "content": query}
            ]
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload
        )
        answer = resp.json()['choices'][0]['message']['content']

        # Save assistant message
        conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                     (session_id, 'assistant', answer))
        conn.commit()
        conn.close()

        return jsonify({'answer': answer})

    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query', '').strip()

    if not session_id or not query:
        return jsonify({'error': 'Missing session_id or query'}), 400

    conn = get_db()
    session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not session:
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    video_id = session['video_id']
    coll_name = f"yt_{video_id.replace('-', '_')}"

    # Save user message
    conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                 (session_id, 'user', query))
    conn.commit()

    try:
        # Search Milvus for relevant context
        search_res = milvus_client.search(
            collection_name=coll_name,
            data=model.encode([query]).tolist(),
            limit=3,
            output_fields=["text"]
        )
        context = "\n".join([r['entity']['text'] for r in search_res[0]])
        conn.close()

    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

    def generate():
        full_answer = ''
        try:
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": f"You are a helpful assistant. Answer based ONLY on this transcript context:\n\n{context}"},
                    {"role": "user", "content": query}
                ],
                "stream": True
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                json=payload,
                stream=True
            )

            import json as _json
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    chunk_data = line[6:]
                    if chunk_data == '[DONE]':
                        yield 'data: [DONE]\n\n'
                        break
                    try:
                        chunk = _json.loads(chunk_data)
                        token = chunk['choices'][0]['delta'].get('content', '')
                        if token:
                            full_answer += token
                            yield f'data: {_json.dumps({"token": token})}\n\n'
                    except Exception:
                        continue

            # Save complete answer to DB
            save_conn = get_db()
            save_conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                              (session_id, 'assistant', full_answer))
            save_conn.commit()
            save_conn.close()

        except Exception as e:
            import json as _json
            yield f'data: {_json.dumps({"token": f" [Error: {str(e)}]"})}\n\n'
            yield 'data: [DONE]\n\n'

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)