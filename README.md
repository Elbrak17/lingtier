# LingTier
AI code-review companion built on Gemini 3 (Flash/Pro/Image) with tri-level analysis, caching, and iterative refinement.

## Highlights
- 🔍 Tri-analysis: Low / Medium / High “thinking levels” in parallel.
- 🧠 Thought signatures: Preserve reasoning across iterations.
- 🧪 Code execution (high level) and optional Google grounding.
- 🎨 Architecture diagram generation.
- ⚡ Context caching to save tokens and bandwidth.
- 🛡️ Hardened ZIP handling (path traversal, ZIP bombs, .gitignore respect).

## Requirements
- Python 3.10+
- `GEMINI_API_KEY` set in your environment or `.env`
- (Optional) Node/NPM if you want to self-host Tailwind instead of the CDN.

## Quick Start
```bash
git clone https://github.com/<your-org>/lingtier.git
cd lingtier
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # create and set GEMINI_API_KEY
uvicorn main:app --reload --port 8888
```
Open http://localhost:8888

## Configuration
- `GEMINI_API_KEY` (required)
- `CORS_ALLOW_ORIGINS` (optional, CSV) defaults to `*` for local/demo. Set your frontend origin(s) in production.

## Usage (UI)
1) Upload a file or ZIP (up to 50MB; respects `.gitignore`; skips binaries and huge files).  
2) Enter a prompt, optionally enable grounding, caching, and diagram generation.  
3) Click “Run Tri-Analysis” to see Low/Medium/High side by side.  
4) For High analyses with a thought signature, use “Iterate” to refine with feedback.  
5) Caching: check “Cache context,” run once, then subsequent runs can reuse cache without re-uploading.

## API Overview
- `POST /upload` — stream upload file/ZIP; returns normalized content and project stats.
- `POST /analyze` — run one analysis with `level` (`low|medium|high`), optional `grounding`, `generate_image`, `cache_id`.
- `POST /iterate` — refine using a `thought_signature` plus feedback and original context.
- `POST /cache-content` — store content server-side for reuse (TTL cache).

Minimal example:
```json
{
  "file_content": "def foo(): ...",
  "prompt": "Find bugs and suggest fixes",
  "level": "high",
  "grounding": false,
  "generate_image": false
}
```

## Security & Limits
- ZIP defenses: path normalization, hidden file skip, per-file compression ratio, max decompressed size (200MB), per-file size (100KB), file count cap (150), context cap.
- `.gitignore` respected (multiple files, anchored patterns).
- CORS is configurable; lock it down for production.
- DOMPurify sanitization for rendered analysis (with fallback if CDN is unavailable).

## Testing
- Syntax check: `python3 -m py_compile main.py`
- Local run: `uvicorn main:app --reload --port 8888` and exercise the UI flows (upload, tri-analysis, iterate, caching).
- ZIP smoke: upload a ZIP with nested folders and `.gitignore`; verify project stats and skip counts.

## Production Notes
- Replace Tailwind CDN with a built CSS bundle (Tailwind CLI/PostCSS) to avoid the CDN warning and reduce size.
- Set `CORS_ALLOW_ORIGINS` to your deployed frontend origin(s).
- Consider enabling Google GenAI server-side caching (client.caches) if you want true token-level cache benefits beyond the local TTL cache.

## License
MIT
