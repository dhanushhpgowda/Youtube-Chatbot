# =============================================================================
# web_app.py  —  Flask RAG App (Upgraded: SQLite → PostgreSQL + Hybrid Search)
# =============================================================================
# CHANGE LOG (every change is marked inline with [CHANGED] or [NEW]):
#   [CHANGED] sqlite3 import removed → psycopg2 added
#   [CHANGED] get_db() → get_db_connection() returning a psycopg2 connection
#   [CHANGED] init_db() schema: AUTOINCREMENT→SERIAL, ? placeholders→%s,
#             added transcript_chunks table with tsvector column
#   [NEW]     save_chunks_to_postgres() — persists chunks + tsvector to PG
#   [NEW]     keyword_search() — full-text search via tsvector / to_tsquery
#   [CHANGED] /api/load_video — calls save_chunks_to_postgres() after Milvus
#   [CHANGED] /api/chat — runs keyword_search() + Milvus, merges context
#   [CHANGED] /api/chat_stream — same hybrid context as /api/chat
#   All other routes (sessions CRUD, streaming logic) are UNCHANGED.
# =============================================================================


# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# ── YouTube ───────────────────────────────────────────────────────────────────
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# ── ML / Vector ───────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# ── Utilities ─────────────────────────────────────────────────────────────────
from utils.processor import get_video_id

# [CHANGED] psycopg2 replaces sqlite3
# pip install psycopg2-binary
import psycopg2
import psycopg2.extras  # gives us RealDictCursor (behaves like sqlite3.Row)

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# [CHANGED] PostgreSQL connection string built from .env variables.
#           Previously was:  DB_PATH = "chat_history.db"
#


PG_DSN = (
    f"host={os.getenv('PG_HOST', 'localhost')} "
    f"port={os.getenv('PG_PORT', '5432')} "
    f"dbname={os.getenv('PG_DBNAME', 'rag_d')} "
    f"user={os.getenv('PG_USER', 'rag_user')} "
    f"password={os.getenv('PG_PASSWORD', 'rag_password')}"
)

# =============================================================================
# INITIALIZATION  (unchanged — same models, same Milvus client)
# =============================================================================

print("Initializing AI models and Vector DB...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
milvus_client = MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:19530")


# =============================================================================
# [CHANGED] DATABASE SETUP — PostgreSQL schema
#   Key differences from the old SQLite schema:
#     • AUTOINCREMENT  →  SERIAL  (PG auto-increment type)
#     • ? placeholders →  %s      (psycopg2 uses %s)
#     • No FOREIGN KEY pragma needed (PG enforces FKs natively)
#   [NEW] transcript_chunks table stores raw text + tsvector for FTS
# =============================================================================

def init_db():
    """Create all tables if they don't already exist."""
    conn = get_db_connection()
    cur = conn.cursor()

    # --- sessions table (same columns, PG syntax) ----------------------------
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id         SERIAL PRIMARY KEY,
            video_id   TEXT        NOT NULL,
            video_url  TEXT        NOT NULL,
            title      TEXT,
            created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # --- messages table (same columns, PG syntax) ----------------------------
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id         SERIAL PRIMARY KEY,
            session_id INTEGER     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role       TEXT        NOT NULL,
            content    TEXT        NOT NULL,
            created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # [NEW] transcript_chunks — one row per ~500-char chunk per video
    #   chunk_text   : the raw text of the chunk
    #   search_vector: a tsvector column that PostgreSQL uses for full-text search
    #                  GIN index makes queries very fast even with thousands of chunks
    cur.execute('''
        CREATE TABLE IF NOT EXISTS transcript_chunks (
            id            SERIAL PRIMARY KEY,
            session_id    INTEGER  NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            video_id      TEXT     NOT NULL,
            chunk_index   INTEGER  NOT NULL,
            chunk_text    TEXT     NOT NULL,
            search_vector TSVECTOR,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # [NEW] GIN index — makes full-text search fast (create only if missing)
    cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
        ON transcript_chunks USING GIN (search_vector)
    ''')

    conn.commit()
    cur.close()
    conn.close()
    print("PostgreSQL tables ready.")


# [CHANGED] get_db() renamed to get_db_connection() to be explicit.
#           Returns a psycopg2 connection with RealDictCursor as the default
#           cursor factory — rows behave like dicts, same as sqlite3.Row.
def get_db_connection():
    """Open and return a new PostgreSQL connection."""
    conn = psycopg2.connect(PG_DSN, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


init_db()  # run on startup — same as before


# =============================================================================
# [NEW] HELPER: Save transcript chunks to PostgreSQL
# =============================================================================

def save_chunks_to_postgres(session_id: int, video_id: str, chunks: list[str]):
    """
    Persist every transcript chunk into the transcript_chunks table.
    The tsvector column is populated automatically using to_tsvector()
    so that keyword_search() can query it instantly.

    Args:
        session_id : the session row id this video belongs to
        video_id   : YouTube video id string
        chunks     : list of plain-text chunk strings
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Delete any existing chunks for this session (safe to re-process a video)
    cur.execute(
        "DELETE FROM transcript_chunks WHERE session_id = %s",
        (session_id,)
    )

    for idx, chunk_text in enumerate(chunks):
        cur.execute(
            """
            INSERT INTO transcript_chunks
                (session_id, video_id, chunk_index, chunk_text, search_vector)
            VALUES
                (%s, %s, %s, %s, to_tsvector('english', %s))
            """,
            (session_id, video_id, idx, chunk_text, chunk_text)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"  → Saved {len(chunks)} chunks to PostgreSQL for session {session_id}.")


# =============================================================================
# [NEW] HELPER: Keyword search using PostgreSQL full-text search
# =============================================================================

def keyword_search(session_id: int, query: str, limit: int = 3) -> list[str]:
    """
    Search transcript_chunks for the given session using PostgreSQL FTS.

    How it works:
      • to_tsquery('english', ...) converts the query into a FTS query object.
        e.g. "machine learning" → 'machine' & 'learning'
      • The @@ operator checks if search_vector matches the query.
      • ts_rank() scores matches so the most relevant chunks come first.
      • plainto_tsquery is used because it's more forgiving of natural language
        (no need to manually add & between words).

    Args:
        session_id : filter chunks to this session only
        query      : the user's natural language question
        limit      : how many top chunks to return

    Returns:
        list of chunk_text strings, best matches first
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            chunk_text,
            ts_rank(search_vector, plainto_tsquery('english', %s)) AS rank
        FROM transcript_chunks
        WHERE
            session_id = %s
            AND search_vector @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, session_id, query, limit)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row['chunk_text'] for row in rows]


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')  # unchanged


# -----------------------------------------------------------------------------
# GET /api/sessions  [CHANGED] sqlite3 → psycopg2, ? → %s
# -----------------------------------------------------------------------------
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT * FROM sessions ORDER BY created_at DESC')
    sessions = cur.fetchall()

    result = []
    for s in sessions:
        cur.execute(
            'SELECT COUNT(*) AS cnt FROM messages WHERE session_id = %s',
            (s['id'],)
        )
        msg_count = cur.fetchone()['cnt']
        result.append({
            'id':            s['id'],
            'video_id':      s['video_id'],
            'video_url':     s['video_url'],
            'title':         s['title'] or f"Video: {s['video_id']}",
            'created_at':    str(s['created_at']),
            'message_count': msg_count
        })

    cur.close()
    conn.close()
    return jsonify(result)


# -----------------------------------------------------------------------------
# GET /api/sessions/<id>  [CHANGED] sqlite3 → psycopg2, ? → %s
# -----------------------------------------------------------------------------
@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session(session_id):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT * FROM sessions WHERE id = %s', (session_id,))
    session = cur.fetchone()
    if not session:
        cur.close()
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    cur.execute(
        'SELECT * FROM messages WHERE session_id = %s ORDER BY created_at ASC',
        (session_id,)
    )
    messages = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify({
        'id':         session['id'],
        'video_id':   session['video_id'],
        'video_url':  session['video_url'],
        'title':      session['title'],
        'created_at': str(session['created_at']),
        'messages':   [dict(m) for m in messages]
    })


# -----------------------------------------------------------------------------
# DELETE /api/sessions/<id>  [CHANGED] sqlite3 → psycopg2, ? → %s
# -----------------------------------------------------------------------------
@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # transcript_chunks and messages both have ON DELETE CASCADE,
    # so deleting the session automatically removes child rows.
    cur.execute('DELETE FROM sessions WHERE id = %s', (session_id,))
    conn.commit()

    cur.close()
    conn.close()
    return jsonify({'success': True})


# -----------------------------------------------------------------------------
# POST /api/load_video
# [CHANGED] sqlite3 → psycopg2, ? → %s, lastrowid → RETURNING id
# [NEW]     calls save_chunks_to_postgres() after Milvus insert
# -----------------------------------------------------------------------------
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

        transcript_list_obj = api_instance.list(video_id)

        fetched_transcript = None
        used_lang = None

        # 1. Try manual English transcripts first (unchanged logic)
        for lang in ['en', 'en-GB', 'en-US']:
            try:
                fetched_transcript = api_instance.fetch(video_id, languages=[lang])
                used_lang = lang
                print(f"  → Found manual transcript: {lang}")
                break
            except Exception:
                continue

        # 2. Fall back to any auto-generated transcript (unchanged logic)
        if fetched_transcript is None:
            for t in transcript_list_obj:
                try:
                    fetched_transcript = t.fetch()
                    used_lang = t.language_code
                    print(f"  → Using {'auto-generated' if t.is_generated else 'manual'} "
                          f"transcript: {t.language_code} ({t.language})")
                    break
                except Exception:
                    continue

        if fetched_transcript is None:
            return jsonify({'error': 'No transcript available for this video.'}), 400

        transcript_data = fetched_transcript.to_raw_data()
        full_text = " ".join(chunk["text"] for chunk in transcript_data)
        print(f"  → Transcript length: {len(full_text)} chars, {len(transcript_data)} segments")

        # ── Milvus (unchanged) ────────────────────────────────────────────────
        coll_name = f"yt_{video_id.replace('-', '_')}"
        if milvus_client.has_collection(coll_name):
            milvus_client.drop_collection(coll_name)
        milvus_client.create_collection(collection_name=coll_name, dimension=768, auto_id=True)

        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 400)]
        embed_data = [{"vector": model.encode(c).tolist(), "text": c} for c in chunks]
        milvus_client.insert(collection_name=coll_name, data=embed_data)
        print(f"  → Inserted {len(chunks)} chunks into Milvus.")

        # ── Video title via oEmbed (unchanged) ───────────────────────────────
        title = None
        try:
            oembed = requests.get(
                f"https://www.youtube.com/oembed?url={url}&format=json", timeout=5
            ).json()
            title = oembed.get('title')
        except Exception:
            title = f"Video {video_id}"

        # [CHANGED] Save session — psycopg2 uses RETURNING to get the new id
        #           (sqlite3 used cursor.lastrowid)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            '''
            INSERT INTO sessions (video_id, video_url, title)
            VALUES (%s, %s, %s)
            RETURNING id
            ''',
            (video_id, url, title)
        )
        session_id = cur.fetchone()['id']
        conn.commit()
        cur.close()
        conn.close()

        # [NEW] Also save chunks into PostgreSQL for keyword search
        save_chunks_to_postgres(session_id, video_id, chunks)

        return jsonify({
            'session_id':     session_id,
            'video_id':       video_id,
            'title':          title,
            'chunk_count':    len(chunks),
            'transcript_lang': used_lang
        })

    except TranscriptsDisabled:
        return jsonify({'error': 'Transcripts are disabled for this video.'}), 400
    except Exception as e:
        print(f"  → Error: {e}")
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------------------------------------
# POST /api/chat
# [CHANGED] sqlite3 → psycopg2, ? → %s
# [NEW]     Hybrid search: keyword_search() + Milvus, context merged
# -----------------------------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query', '').strip()

    if not session_id or not query:
        return jsonify({'error': 'Missing session_id or query'}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT * FROM sessions WHERE id = %s', (session_id,))
    session = cur.fetchone()
    if not session:
        cur.close()
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    video_id = session['video_id']
    coll_name = f"yt_{video_id.replace('-', '_')}"

    # Save user message [CHANGED] ? → %s
    cur.execute(
        'INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)',
        (session_id, 'user', query)
    )
    conn.commit()

    try:
        # [UNCHANGED] Semantic search via Milvus
        search_res = milvus_client.search(
            collection_name=coll_name,
            data=model.encode([query]).tolist(),
            limit=3,
            output_fields=["text"]
        )
        semantic_chunks = [r['entity']['text'] for r in search_res[0]]

        # [NEW] Keyword search via PostgreSQL full-text search
        keyword_chunks = keyword_search(session_id, query, limit=3)

        # [NEW] Merge results — deduplicate while preserving order.
        #       Semantic results go first (usually higher quality),
        #       then any keyword-only results that weren't already included.
        seen = set()
        combined_chunks = []
        for chunk in semantic_chunks + keyword_chunks:
            # Use first 80 chars as a dedup key (avoids exact-string comparison cost)
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                combined_chunks.append(chunk)

        # Build context string from merged chunks
        context = "\n\n---\n\n".join(combined_chunks)
        print(f"  → Context: {len(semantic_chunks)} semantic + "
              f"{len(keyword_chunks)} keyword chunks → {len(combined_chunks)} unique")

        # [UNCHANGED] Query Groq LLM
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer based ONLY on this "
                        f"transcript context:\n\n{context}"
                    )
                },
                {"role": "user", "content": query}
            ]
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload
        )
        answer = resp.json()['choices'][0]['message']['content']

        # Save assistant message [CHANGED] ? → %s
        cur.execute(
            'INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)',
            (session_id, 'assistant', answer)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'answer': answer})

    except Exception as e:
        cur.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------------------------------------
# POST /api/chat_stream
# [CHANGED] sqlite3 → psycopg2, ? → %s
# [NEW]     Same hybrid context as /api/chat
# -----------------------------------------------------------------------------
@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query', '').strip()

    if not session_id or not query:
        return jsonify({'error': 'Missing session_id or query'}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT * FROM sessions WHERE id = %s', (session_id,))
    session = cur.fetchone()
    if not session:
        cur.close()
        conn.close()
        return jsonify({'error': 'Session not found'}), 404

    video_id = session['video_id']
    coll_name = f"yt_{video_id.replace('-', '_')}"

    # Save user message
    cur.execute(
        'INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)',
        (session_id, 'user', query)
    )
    conn.commit()

    try:
        # [UNCHANGED] Semantic search
        search_res = milvus_client.search(
            collection_name=coll_name,
            data=model.encode([query]).tolist(),
            limit=3,
            output_fields=["text"]
        )
        semantic_chunks = [r['entity']['text'] for r in search_res[0]]

        # [NEW] Keyword search
        keyword_chunks = keyword_search(session_id, query, limit=3)

        # [NEW] Merge and deduplicate (same logic as /api/chat)
        seen = set()
        combined_chunks = []
        for chunk in semantic_chunks + keyword_chunks:
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                combined_chunks.append(chunk)

        context = "\n\n---\n\n".join(combined_chunks)
        cur.close()
        conn.close()

    except Exception as e:
        cur.close()
        conn.close()
        return jsonify({'error': str(e)}), 500

    # [UNCHANGED] Streaming generator
    def generate():
        full_answer = ''
        try:
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer based ONLY on this "
                            f"transcript context:\n\n{context}"
                        )
                    },
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

            # Persist full streamed answer
            save_conn = get_db_connection()
            save_cur = save_conn.cursor()
            save_cur.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)',
                (session_id, 'assistant', full_answer)
            )
            save_conn.commit()
            save_cur.close()
            save_conn.close()

        except Exception as e:
            import json as _json
            yield f'data: {_json.dumps({"token": f" [Error: {str(e)}]"})}\n\n'
            yield 'data: [DONE]\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)