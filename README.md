# LingTier - Adaptive Cognitive Tiers

**Hackathon Demo: Demonstrating Gemini 3's Exclusive Features**

## 🚀 Quick Start

```bash
cd /home/elbrak17/lingtier
source venv/bin/activate
set -a && source .env && set +a
uvicorn main:app --reload --port 8888
```

Open **http://localhost:8888**

---

## 🎯 Gemini 3 Features Demonstrated

| Feature | Description | Endpoint |
|---------|-------------|----------|
| `thinking_level` | Control cognitive depth (low/medium/high) | `/analyze` |
| `thought_signatures` | Persist reasoning across iterations | `/iterate` |
| `code_execution` | Sandbox validation on HIGH level | `/analyze` |
| `google_search` | Ground facts with web search | `/analyze?grounding=true` |
| `image_generation` | Architecture diagram from analysis | `/analyze?generate_image=true` |
| `context_caching` | Cache content for cost savings | `/cache-content` |
| `1M context window` | Support files up to 50MB | `/upload` |
| `structured_output` | Pydantic response models | All endpoints |

---

## 📁 Files

```
/lingtier/
├── main.py       # FastAPI backend (~320 lines)
├── index.html    # Tailwind frontend
├── .env          # GEMINI_API_KEY
└── README.md     # This file
```

---

## 🔧 API Endpoints

### POST /analyze
Analyze with thinking level + optional grounding/image gen.

```json
{
  "file_content": "def foo(): ...",
  "prompt": "Find bugs",
  "level": "high",
  "grounding": true,
  "generate_image": true
}
```

### POST /iterate
Refine analysis with thought_signature context.

### POST /cache-content
Cache file content for faster subsequent calls.

---

## 🎨 New Features (v2)

- **🔍 Grounding**: Verify facts with Google Search
- **🎨 Image Gen**: Generate architecture diagrams
- **⚡ Caching**: 60% cost savings on iterations

---

## 📜 License

MIT - Hackathon Project
