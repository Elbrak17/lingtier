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
import tempfile
import re
import json
import fnmatch
import traceback
from dataclasses import dataclass
from typing import Optional, List, Dict, BinaryIO
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
MODEL_CONTEXT_PROMPT = (
    "You are running inside the LingTier app using the Gemini 3 Flash Preview model "
    "(model name: gemini-3-flash-preview). "
    "In the official Gemini API documentation, Gemini 3 preview model names include "
    "gemini-3-flash-preview, gemini-3-pro-preview, and gemini-3-pro-image-preview. "
    "Treat 'Gemini 3' as the model family name used by this app. "
    "Do not speculate about public availability, naming disputes, or whether the model exists. "
    "If asked about the model, state that you are a Gemini 3 preview model accessed via the API. "
    "Focus on the user's request and the provided code context."
)

# ZIP Processing Configuration
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', 
                      '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs', '.vue',
                      '.html', '.css', '.scss', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}
SKIP_DIRS = {'node_modules', '__pycache__', '.git', '.svn', 'venv', 'env', '.env', '.venv',
             'dist', 'build', 'target', '.idea', '.vscode', 'vendor', '.next', 'coverage',
             '.pytest_cache', '.mypy_cache', '.tox', 'htmlcov', 'site-packages', 'wheels',
             '.eggs', 'lib', 'lib64', 'parts', 'sdist', 'var', 'eggs', '*.egg-info'}
EXACT_SKIP_DIRS = {entry for entry in SKIP_DIRS if '*' not in entry}
WILDCARD_SKIP_DIRS = [entry for entry in SKIP_DIRS if '*' in entry]
MAX_FILE_SIZE_PER_FILE = 100 * 1024  # 100KB
MAX_FILES_TO_ANALYZE = 150
MAX_CONTEXT_CHARS = 800000

# SECURITY: ZIP bomb protection
MAX_DECOMPRESSED_SIZE = 200 * 1024 * 1024  # 200MB
MAX_COMPRESSION_RATIO = 100  # Per-file ratio limit


@dataclass(frozen=True)
class GitignorePattern:
    pattern: str
    base_dir: str
    anchored: bool
    has_slash: bool
    has_wildcards: bool


def parse_gitignore(gitignore_content: str, base_dir: str) -> List[GitignorePattern]:
    """
    Parse .gitignore content and return a list of patterns to ignore.
    Supports basic gitignore syntax (comments, negation, directories).
    """
    patterns: List[GitignorePattern] = []
    normalized_base = base_dir.strip("/")
    for line in gitignore_content.split('\n'):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        # Skip negation patterns (!) for simplicity
        if line.startswith('!'):
            continue
        anchored = line.startswith('/')
        # Remove leading/trailing slashes
        line = line.lstrip('/').rstrip('/')
        if not line:
            continue
        patterns.append(GitignorePattern(
            pattern=line,
            base_dir=normalized_base,
            anchored=anchored,
            has_slash='/' in line,
            has_wildcards=any(token in line for token in ('*', '?', '['))
        ))
    return patterns


def matches_gitignore_pattern(file_path: str, pattern: GitignorePattern) -> bool:
    normalized_path = file_path.strip('/')

    if pattern.base_dir:
        base_prefix = f"{pattern.base_dir}/"
        if not normalized_path.startswith(base_prefix):
            return False
        relative_path = normalized_path[len(base_prefix):]
    else:
        relative_path = normalized_path

    if not relative_path:
        return False

    if pattern.anchored:
        if fnmatch.fnmatch(relative_path, pattern.pattern):
            return True
        if not pattern.has_wildcards:
            return relative_path.startswith(f"{pattern.pattern}/")
        return False

    if pattern.has_slash:
        if fnmatch.fnmatch(relative_path, pattern.pattern):
            return True
        if not pattern.has_wildcards:
            return relative_path.startswith(f"{pattern.pattern}/")
        return False

    relative_parts = relative_path.split('/')
    return any(fnmatch.fnmatch(part, pattern.pattern) for part in relative_parts)


def should_skip_path(file_path: str, gitignore_patterns: List[GitignorePattern]) -> bool:
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
        if matches_gitignore_pattern(file_path, pattern):
            return True
    
    return False

app = FastAPI(title="LingTier", description="Adaptive Cognitive Tiers with Gemini 3")

CORS_ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
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


@app.on_event("startup")
async def startup_event():
    global client
    if client is None and GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)


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
def process_zip_file(zip_stream: BinaryIO, compressed_size: int) -> tuple:
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
    gitignore_patterns: List[GitignorePattern] = []
    gitignore_files = 0
    
    try:
        with zipfile.ZipFile(zip_stream, 'r') as zf:
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
                        base_dir = os.path.dirname(file_info.filename)
                        gitignore_patterns.extend(parse_gitignore(gitignore_content, base_dir))
                        gitignore_files += 1
                    except Exception as e:
                        print(f"⚠️ Failed to parse .gitignore: {e}")

            if gitignore_patterns:
                print(f"📋 Loaded {len(gitignore_patterns)} patterns from {gitignore_files} .gitignore files")
            
            # Sort by size (smaller first)
            file_list = sorted(zf.infolist(), key=lambda x: x.file_size)
            
            print(f"📦 Processing ZIP with {len(file_list)} entries...")
            
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
                
                # Parse path parts for filtering
                path_parts = file_path.split('/')
                
                # SMART FILTERING: Skip hidden files/directories (.*)
                has_hidden = False
                for part in path_parts:
                    if part.startswith('.') and part not in {'.', '..'}:
                        has_hidden = True
                        break
                
                if has_hidden:
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["hidden"] += 1
                    continue
                
                # SMART FILTERING: Check for excluded directories (PRIORITY CHECK)
                # This must come BEFORE gitignore to catch venv, node_modules, etc.
                is_excluded_dir = False
                for part in path_parts:
                    if part in EXACT_SKIP_DIRS:
                        is_excluded_dir = True
                        break
                    if WILDCARD_SKIP_DIRS and any(fnmatch.fnmatch(part, skip_pattern) for skip_pattern in WILDCARD_SKIP_DIRS):
                        is_excluded_dir = True
                        break
                    if is_excluded_dir:
                        break
                
                if is_excluded_dir:
                    stats["files_skipped"] += 1
                    stats["skipped_reasons"]["excluded_dir"] += 1
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
    
    # Log summary
    print(f"✅ ZIP processing complete:")
    print(f"   📊 Scanned: {stats['files_scanned']}, Analyzed: {stats['files_analyzed']}, Skipped: {stats['files_skipped']}")
    print(f"   🚫 Skip reasons: {stats['skipped_reasons']}")
    
    estimated_tokens = total_chars // 4
    project_context = build_project_context(files_content, stats, estimated_tokens)

    stats["estimated_tokens"] = estimated_tokens

    return project_context, stats, files_content


def build_project_context(files_content: List[str], stats: Dict[str, int], estimated_tokens: int) -> str:
    return f"""
# PROJECT ANALYSIS
# Files Analyzed: {stats['files_analyzed']} / {stats['files_scanned']} scanned
# Estimated Tokens: {estimated_tokens}

{''.join(files_content)}
"""


async def update_project_token_estimate(project_context: str, stats: Dict[str, int], files_content: List[str]) -> str:
    if not GEMINI_API_KEY:
        return project_context

    try:
        cli = get_client()
    except HTTPException:
        return project_context

    try:
        response = await asyncio.to_thread(
            cli.models.count_tokens,
            model=MODEL_NAME,
            contents=project_context
        )
        token_count = getattr(response, "total_tokens", None) or getattr(response, "token_count", None)
        if isinstance(token_count, int) and token_count > 0:
            stats["estimated_tokens"] = token_count
            return build_project_context(files_content, stats, token_count)
    except Exception as e:
        print(f"Token count failed: {e}")

    return project_context


# Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # STREAMING: Read file in chunks instead of all at once
    chunk_size = 8192  # 8KB chunks
    total_size = 0
    temp_file = tempfile.SpooledTemporaryFile(max_size=MAX_FILE_SIZE, mode="w+b")
    
    print(f"📥 Starting upload: {file.filename}")
    
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            temp_file.write(chunk)
            total_size += len(chunk)

            # Log progress every 1MB for Railway monitoring
            if total_size % (1024 * 1024) < chunk_size:
                print(f"📥 Uploaded {total_size / (1024 * 1024):.1f}MB...")

            # Check size limit during streaming
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

        print(f"✅ Upload complete: {total_size / (1024 * 1024):.1f}MB")

        temp_file.seek(0)
        signature = temp_file.read(4)
        is_zip = file.filename.endswith('.zip') or signature == b'PK\x03\x04'
        temp_file.seek(0)

        if is_zip:
            project_context, stats, files_content = process_zip_file(temp_file, total_size)
            project_context = await update_project_token_estimate(project_context, stats, files_content)
            return {
                "filename": file.filename,
                "size": total_size,
                "content": project_context,
                "is_project": True,
                "project_stats": stats
            }
        else:
            content = temp_file.read()
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
    finally:
        temp_file.close()


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
        file_content = request.file_content or ""
        if request.cache_id and request.cache_id in context_cache:
            cache_hit = True
            cached_data = context_cache[request.cache_id]
            file_content = cached_data.get('content', file_content)
            tokens_saved = cached_data.get('tokens', 0)
            cache_expires_in = int(3600 - (time.time() - cached_data.get('created', time.time())))
        elif request.cache_id and not file_content:
            raise HTTPException(status_code=400, detail="Cache miss. Please re-upload or refresh the cache.")

        if not file_content:
            raise HTTPException(status_code=400, detail="File content is empty.")

        full_prompt = f"{MODEL_CONTEXT_PROMPT}\n\n{request.prompt}\n\n---\nFILE CONTENT:\n{file_content}"
        
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
        print(f"Analyze failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/iterate", response_model=IterateResponse)
async def iterate(request: IterateRequest):
    cli = get_client()
    
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=MODEL_CONTEXT_PROMPT)]
            ),
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
        
        return IterateResponse(
            refined_analysis=refined_text,
            new_signature=new_signature
        )
        
    except Exception as e:
        print(f"Iteration failed: {e}\n{traceback.format_exc()}")
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
