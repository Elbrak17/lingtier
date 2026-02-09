"""
LingTier - Adaptive Cognitive Tiers
Hackathon demo: Gemini 3's thinking_level + thought_signatures + grounding + image_gen + ZIP projects
SECURITY HARDENED V2: Path traversal, per-file ZIP bomb, JSON mode, multilingual
"""

import os
import time
import asyncio
import base64
import uuid
import zipfile
import io
import re
import json
import fnmatch
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from cachetools import TTLCache

# Load environment variables from .env file
load_dotenv()

# Configuration - GEMINI 3 MODELS (DO NOT CHANGE TO GEMINI 2.0)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"  # Gemini 3 Flash for thinking_level
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"  # For architecture diagrams
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ZIP Processing Configuration
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                      '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs', '.vue',
                      '.html', '.css', '.scss', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}
SKIP_DIRS = {'node_modules', '__pycache__', '.git', '.svn', 'venv', 'env', '.env', '.venv',
             'dist', 'build', 'target', '.idea', '.vscode', 'vendor', '.next', 'coverage',
             '.pytest_cache', '.mypy_cache', '.tox', 'htmlcov', 'site-packages', 'wheels',
             '.eggs', 'lib', 'lib64', 'parts', 'sdist', 'var', 'eggs', '*.egg-info'}
MAX_FILE_SIZE_PER_FILE = 100 * 1024  # 100KB
MAX_FILES_TO_ANALYZE = 150
MAX_CONTEXT_CHARS = 800000

# SECURITY: ZIP bomb protection
MAX_DECOMPRESSED_SIZE = 200 * 1024 * 1024  # 200MB
MAX_COMPRESSION_RATIO = 100  # Per-file ratio limit


def parse_gitignore(gitignore_content: str) -> set:
    """
    Parse .gitignore content and return set of patterns to ignore.
    Supports basic gitignore syntax (comments, negation, directories).
    """
    patterns = set()
    for line in gitignore_content.split('\n'):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        # Skip negation patterns (!) for simplicity
        if line.startswith('!'):
            continue
        # Remove trailing slashes
        line = line.rstrip('/')
        patterns.add(line)
    return patterns


def should_skip_path(file_path: str, gitignore_patterns: set) -> bool:
    """
    Check if a file path should be skipped based on gitignore patterns.
    Supports wildcards (*) and directory matching.
    """
    path_parts = file_path.split('/')
    
    # Check if any part is a hidden file/directory (starts with .)
    for part in path_parts:
        if part.startswith('.') and part not in {'.', '..'}:
            return True
    
    # Check against gitignore patterns
    for pattern in gitignore_patterns:
        # Exact match
        if pattern in path_parts:
            return True
        
        # Wildcard match (simple implementation)
        if '*' in pattern:
            import fnmatch
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            # Also check full path
            if fnmatch.fnmatch(file_path, pattern):
                return True
        
        # Check if pattern matches any directory in path
        if pattern in file_path:
            return True
    
    return False

app = FastAPI(title="LingTier", description="Adaptive Cognitive Tiers with Gemini 3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = None

# TTL cache with auto-expiration
context_cache = TTLCache(maxsize=50, ttl=3600)


def get_client():
    global client
    if client is None:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
        client = genai.Client(api_key=GEMINI_API_KEY)
    return client


# Language detection for multilingual support
def detect_language(text: str) -> str:
    """Detect language from text using character and word heuristics"""
    french_chars = ['é', 'è', 'ê', 'à', 'ù', 'ç', 'ô', 'î', 'â', 'û', 'ë', 'ï', 'ü', 'œ', 'æ']
    french_words = ['analyse', 'projet', 'fonction', 'fichier', 'erreur', 'améliorer', 'corriger']
    
    text_lower = text.lower()
    
    # Check for French accented characters
    if sum(1 for c in text_lower if c in french_chars) > 2:
        return "french"
    
    # Check for French words
    if any(word in text_lower for word in french_words):
        return "french"
    
    return "english"


def get_image_prompt(user_prompt: str, analysis_summary: str) -> str:
    """Generate localized image prompt based on detected language"""
    lang = detect_language(user_prompt)
    
    prompts = {
        "french": f"Diagramme d'architecture technique du projet. Style blueprint professionnel, 4K, lignes claires, texte en FRANÇAIS lisible. Architecture: {analysis_summary[:1000]}",
        "english": f"Technical architecture diagram of the project. Professional blueprint style, 4K, clean lines, readable ENGLISH text. Architecture: {analysis_summary[:1000]}"
    }
    
    return prompts.get(lang, prompts["english"])


def clean_analysis_output(text: str) -> str:
    """Remove any markdown artifacts from analysis output"""
    if not text:
        return ""
    # Remove code block markers
    text = re.sub(r'```[\w]*\n?', '', text)
    # Remove trailing backticks
    text = re.sub(r'```$', '', text)
    return text.strip()


# Request/Response Models
class AnalyzeRequest(BaseModel):
    file_content: str
    prompt: str
    level: str
    grounding: bool = False
    generate_image: bool = False
    cache_id: Optional[str] = None
    project_mode: bool = False


class ProjectStats(BaseModel):
    mode: str = "project_archive"
    files_scanned: int = 0
    files_analyzed: int = 0
    files_skipped: int = 0
    skipped_reasons: Dict[str, int] = {}
    estimated_tokens: int = 0


class AnalyzeResponse(BaseModel):
    level: str
    analysis: str
    execution_time_ms: int
    tokens_input: int
    tokens_output: int
    thought_signature: Optional[str] = None
    thought_process: Optional[str] = None  # NEW: Display thinking
    code_executed: bool = False
    code_result: Optional[str] = None
    grounded_sources: Optional[List[str]] = None
    image_url: Optional[str] = None
    cache_hit: bool = False
    tokens_saved: int = 0
    cache_expires_in: int = 0  # NEW: TTL countdown
    project_stats: Optional[ProjectStats] = None


class IterateRequest(BaseModel):
    signature: str
    feedback: str
    original_context: str


class IterateResponse(BaseModel):
    refined_analysis: str
    new_signature: Optional[str] = None


class CacheRequest(BaseModel):
    file_content: str
    ttl_seconds: int = 3600


class CacheResponse(BaseModel):
    cache_id: str
    tokens_cached: int
    expires_in_seconds: int
    expires_at: int  # NEW: Timestamp for frontend timer


# SECURITY: ZIP Processing with path traversal and per-file bomb protection
def process_zip_file(zip_content: bytes) -> tuple:
    """
    Extract and filter files from ZIP archive.
    SECURITY: Path traversal protection, per-file compression ratio check.
    SMART FILTERING: Respects .gitignore, skips hidden files/dirs.
    """
    stats = {
        "files_scanned": 0,
        "files_analyzed": 0,
        "files_skipped": 0,
        "skipped_reasons": {
            "binary": 0,
            "too_large": 0,
            "excluded_dir": 0,
            "excluded_ext": 0,
            "encoding_error": 0,
            "security_risk": 0,
            "gitignore": 0,
            "hidden": 0
        }
    }
    
    files_content = []
    total_chars = 0
    compressed_size = len(zip_content)
    gitignore_patterns = set()
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
            # SECURITY: Check total uncompressed size
            total_uncompressed = sum(info.file_size for info in zf.infolist() if not info.is_dir())
            
            if total_uncompressed > MAX_DECOMPRESSED_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"Archive extracts to {total_uncompressed//1024//1024}MB (max: 200MB)"
                )
            
            # SECURITY: Check overall compression ratio
            if compressed_size > 0:
                ratio = total_uncompressed / compressed_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Suspicious compression ratio ({ratio:.0f}:1, max: {MAX_COMPRESSION_RATIO}:1)"
                    )
            
            # SMART FILTERING: Parse .gitignore if present
            for file_info in zf.infolist():
                if file_info.filename.endswith('.gitignore') and not file_info.is_dir():
                    try:
                        gitignore_content = zf.read(file_info.filename).decode('utf-8')
                        gitignore_patterns = parse_gitignore(gitignore_content)
                        print(f"📋 Found .gitignore with {len(gitignore_patterns)} patterns")
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to parse .gitignore: {e}")
            
            # Sort by size (smaller first)
            file_list = sorted(zf.infolist(), key=lambda x: x.file_size)
            
            for file_info in file_list:
                stats["files_scanned"] += 1
                file_path = file_info.filename
                
                # Skip directories
                if file_info.is_dir():
                    continue
                
                # SECURITY: Path traversal protection
                if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["security_risk"] += 1
                    continue
                
                # SECURITY: Normalize and verify path
                safe_path = os.path.normpath(file_path)
                if safe_path.startswith("..") or os.path.isabs(safe_path):
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["security_risk"] += 1
                    continue
                
                # SMART FILTERING: Skip hidden files/directories (.*) 
                path_parts = file_path.split('/')
                if any(part.startswith('.') and part not in {'.', '..'} for part in path_parts):
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["hidden"] += 1
                    continue
                
                # SMART FILTERING: Check gitignore patterns
                if gitignore_patterns and should_skip_path(file_path, gitignore_patterns):
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["gitignore"] += 1
                    continue
                
                # SECURITY: Per-file compression ratio check
                if file_info.compress_size > 0:
                    file_ratio = file_info.file_size / file_info.compress_size
                    if file_ratio > MAX_COMPRESSION_RATIO:
                        stats["files_skipped"] += 1
                        stats["skipped_reasons"]["security_risk"] += 1
                        continue
                
                # Check for excluded directories
                if any(skip_dir in path_parts for skip_dir in SKIP_DIRS):
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["excluded_dir"] += 1
                    continue
                
                # Check extension
                _, ext = os.path.splitext(file_path)
                if ext.lower() not in ALLOWED_EXTENSIONS:
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["excluded_ext"] += 1
                    continue
                
                # Check file size
                if file_info.file_size > MAX_FILE_SIZE_PER_FILE:
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["too_large"] += 1
                    continue
                
                # Check limits
                if stats["files_analyzed"] >= MAX_FILES_TO_ANALYZE:
                    stats["files_skipped"] += 1
                    continue
                
                if total_chars >= MAX_CONTEXT_CHARS:
                    stats["files_skipped"] += 1
                    continue
                
                # Try multiple encodings
                raw_bytes = zf.read(file_path)
                content = None
                
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content = raw_bytes.decode(encoding)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                if content is None:
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["binary"] += 1
                    continue
                
                # Truncate if too large
                if len(content) > MAX_FILE_SIZE_PER_FILE:
                    content = content[:MAX_FILE_SIZE_PER_FILE] + "\n[... truncated ...]"
                
                # Format file content
                file_header = f"\n{'='*60}\n📄 FILE: {file_path}\n{'='*60}\n"
                file_block = file_header + content
                
                if total_chars + len(file_block) <= MAX_CONTEXT_CHARS:
                    files_content.append(file_block)
                    total_chars += len(file_block)
                    stats["files_analyzed"] += 1
                else:
                    stats["files_skipped"] += 1
                    
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    
    project_context = f"""
# PROJECT ANALYSIS
# Files Analyzed: {stats['files_analyzed']} / {stats['files_scanned']} scanned
# Estimated Tokens: {total_chars // 4}

{''.join(files_content)}
"""
    
    stats["estimated_tokens"] = total_chars // 4
    
    return project_context, stats


# Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # STREAMING: Read file in chunks instead of all at once
    chunk_size = 8192  # 8KB chunks
    content_chunks = []
    total_size = 0
    
    print(f"📥 Starting upload: {file.filename}")
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content_chunks.append(chunk)
        total_size += len(chunk)
        
        # Log progress every 1MB for Railway monitoring
        if total_size % (1024 * 1024) < chunk_size:
            print(f"📥 Uploaded {total_size / (1024 * 1024):.1f}MB...")
        
        # Check size limit during streaming
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Combine chunks
    content = b''.join(content_chunks)
    print(f"✅ Upload complete: {total_size / (1024 * 1024):.1f}MB")
    
    is_zip = file.filename.endswith('.zip') or content[:4] == b'PK\x03\x04'
    
    if is_zip:
        project_context, stats = process_zip_file(content)
        return {
            "filename": file.filename,
            "size": len(content),
            "content": project_context,
            "is_project": True,
            "project_stats": stats
        }
    else:
        text_content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                text_content = content.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if text_content is None:
            text_content = f"[Binary file: {file.filename}, {len(content)} bytes]"
        
        return {
            "filename": file.filename,
            "size": len(content),
            "content": text_content,
            "is_project": False,
            "project_stats": None
        }


async def generate_architecture_image(cli, analysis_summary: str, user_prompt: str) -> Optional[str]:
    """Generate architecture diagram with language-aware prompts"""
    try:
        prompt = get_image_prompt(user_prompt, analysis_summary)
        
        response = await asyncio.to_thread(
            cli.models.generate_content,
            model=IMAGE_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"]
            )
        )
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                mime_type = part.inline_data.mime_type
                image_bytes = part.inline_data.data
                b64_data = base64.b64encode(image_bytes).decode('utf-8')
                return f"data:{mime_type};base64,{b64_data}"
        
        return None
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None


def extract_grounding_sources(response) -> List[str]:
    sources = []
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            if hasattr(chunk.web, 'uri'):
                                sources.append(chunk.web.uri)
    except Exception as e:
        print(f"Error extracting grounding sources: {e}")
    return sources[:5]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    cli = get_client()
    start_time = time.time()
    
    user_lang = detect_language(request.prompt)
    
    full_prompt = f"{request.prompt}\n\n---\nFILE CONTENT:\n{request.file_content}"
    
    try:
        thinking_config = types.ThinkingConfig(thinking_level=request.level)
        
        tools = []
        if request.grounding:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if request.level == "high":
            tools.append(types.Tool(code_execution=types.ToolCodeExecution))
        
        if tools:
            config = types.GenerateContentConfig(
                thinking_config=thinking_config,
                tools=tools
            )
        else:
            config = types.GenerateContentConfig(
                thinking_config=thinking_config
            )
        
        # Check cache
        cache_hit = False
        tokens_saved = 0
        cache_expires_in = 0
        if request.cache_id and request.cache_id in context_cache:
            cache_hit = True
            cached_data = context_cache[request.cache_id]
            tokens_saved = cached_data.get('tokens', 0)
            cache_expires_in = int(3600 - (time.time() - cached_data.get('created', time.time())))
        
        # ASYNC Gemini call
        response = await asyncio.to_thread(
            cli.models.generate_content,
            model=MODEL_NAME,
            contents=full_prompt,
            config=config
        )
        
        end_time = time.time()
        execution_time_ms = int((end_time - start_time) * 1000)
        
        # Extract response
        analysis_text = ""
        thought_signature = None
        thought_process = None
        code_executed = False
        code_result = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                analysis_text += part.text
            
            if hasattr(part, 'thought_signature') and part.thought_signature:
                sig = part.thought_signature
                thought_signature = base64.b64encode(sig).decode('utf-8') if isinstance(sig, bytes) else sig
            
            # NEW: Extract thought process for display
            if hasattr(part, 'thought') and part.thought:
                thought_process = getattr(part.thought, 'text', None)
            
            if hasattr(part, 'executable_code') and part.executable_code:
                code_executed = True
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                code_result = part.code_execution_result.output
        
        # Clean any markdown artifacts
        analysis_text = clean_analysis_output(analysis_text)
        
        tokens_input = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        tokens_output = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        
        grounded_sources = None
        if request.grounding:
            grounded_sources = extract_grounding_sources(response)
        
        # Generate image with language awareness
        image_url = None
        if request.generate_image and request.level == "high" and analysis_text:
            image_url = await generate_architecture_image(cli, analysis_text, request.prompt)
        
        return AnalyzeResponse(
            level=request.level,
            analysis=analysis_text,
            execution_time_ms=execution_time_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            thought_signature=thought_signature,
            thought_process=thought_process,
            code_executed=code_executed,
            code_result=code_result,
            grounded_sources=grounded_sources,
            image_url=image_url,
            cache_hit=cache_hit,
            tokens_saved=tokens_saved,
            cache_expires_in=cache_expires_in,
            project_stats=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/iterate", response_model=IterateResponse)
async def iterate(request: IterateRequest):
    cli = get_client()
    
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=request.original_context)]
            ),
            types.Content(
                role="model", 
                parts=[types.Part(
                    text="[Previous analysis with signature]",
                    thought_signature=request.signature
                )]
            ),
            types.Content(
                role="user",
                parts=[types.Part(text=f"REFINEMENT REQUEST: {request.feedback}")]
            )
        ]
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
        
        response = await asyncio.to_thread(
            cli.models.generate_content,
            model=MODEL_NAME,
            contents=contents,
            config=config
        )
        
        refined_text = ""
        new_signature = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                refined_text += part.text
            if hasattr(part, 'thought_signature') and part.thought_signature:
                sig = part.thought_signature
                new_signature = base64.b64encode(sig).decode('utf-8') if isinstance(sig, bytes) else sig
        
        # Clean output
        refined_text = clean_analysis_output(refined_text)
        
        return IterateResponse(
            refined_analysis=refined_text,
            new_signature=new_signature
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache-content", response_model=CacheResponse)
async def cache_content(request: CacheRequest):
    cache_id = str(uuid.uuid4())[:8]
    
    tokens_estimated = len(request.file_content.split()) * 2
    created_time = time.time()
    
    context_cache[cache_id] = {
        "content": request.file_content,
        "tokens": tokens_estimated,
        "created": created_time
    }
    
    return CacheResponse(
        cache_id=cache_id,
        tokens_cached=tokens_estimated,
        expires_in_seconds=request.ttl_seconds,
        expires_at=int((created_time + request.ttl_seconds) * 1000)  # JS timestamp
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)