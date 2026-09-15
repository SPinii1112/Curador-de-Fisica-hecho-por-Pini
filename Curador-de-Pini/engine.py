"""
engine.py — Motor de análisis de Curador de Pini
=================================================
Responsabilidades:
  - Extraer texto de PDFs digitales.
  - Transcribir audio de archivos de video con Whisper.
  - Mantener una base de conocimiento en JSON (fragmentos de referencia).
  - Calcular similitud semántica entre un texto nuevo y la base.

No depende de Streamlit: puede usarse desde cualquier frontend o script.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASE_PATH = DATA_DIR / "base.json"
CACHE_PATH = DATA_DIR / "embeddings_cache.json"
RESULTS_PATH = DATA_DIR / "historial.json"

# ---------------------------------------------------------------------------
# Modelos cargados de forma diferida (lazy) para no bloquear el arranque
# ---------------------------------------------------------------------------
_embed_model = None
_whisper_models: dict[str, Any] = {}

WHISPER_SIZES = {
    "Rapido (tiny)": "tiny",
    "Equilibrado (base)": "base",
    "Preciso (small)": "small",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PDF_EXTENSION = ".pdf"

MAX_HISTORIAL = 50  # entradas maximas que se conservan en historial.json


# ===========================================================================
# Extraccion de texto — PDF
# ===========================================================================

def extract_pdf_text(path: str | Path) -> str:
    """Extrae texto de un PDF digital."""
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise ValueError(
            "No se encontro texto en el PDF. "
            "Si es un escaneo, primero aplicale OCR (por ejemplo con Adobe Acrobat o SmallPDF)."
        )
    return text


# ===========================================================================
# Fragmentacion de texto (chunking)
# ===========================================================================

def chunk_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    """Divide un texto largo en fragmentos solapados."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ===========================================================================
# Base de conocimiento (biblioteca de referencia)
# ===========================================================================

def _load_base() -> list[dict[str, str]]:
    if not BASE_PATH.exists():
        return []
    return json.loads(BASE_PATH.read_text(encoding="utf-8"))


def _save_base(base: list[dict[str, str]]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    BASE_PATH.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


# ===========================================================================
# Importacion de la base original (base_conocimiento.json del Evaluador)
# ===========================================================================

# Ruta por defecto donde vive base_conocimiento.json en el proyecto original
_LEGACY_DEFAULT = Path(r"C:\Users\Usuario\Desktop\Evaluador_Fisica\base_conocimiento.json")


def bootstrap_from_legacy(legacy_path: Path | None = None) -> int:
    """Importa la base de conocimiento del proyecto Evaluador_Fisica si la
    biblioteca de Curador de Pini todavia esta vacia.

    Convierte el esquema antiguo (campo ``libro``) al nuevo (campo ``documento``).
    La copia queda en ``data/base.json`` y desde ese momento ambos proyectos
    evolucionan por separado.

    Devuelve la cantidad de fragmentos importados (0 si ya habia datos o no se
    encontro el archivo de origen).
    """
    if _load_base():
        return 0  # ya hay datos, no sobreescribir

    path = Path(legacy_path) if legacy_path else _LEGACY_DEFAULT
    if not path.exists():
        return 0

    legacy = json.loads(path.read_text(encoding="utf-8"))
    converted = []
    for item in legacy:
        texto = item.get("texto", "").strip()
        if not texto:
            continue
        documento = item.get("libro", "Referencia sin nombre")
        converted.append(
            {
                "documento": documento,
                "fragmento": str(item.get("fragmento", len(converted) + 1)),
                "texto": texto,
                "hash": hashlib.sha256((documento + texto).encode("utf-8")).hexdigest(),
            }
        )

    if converted:
        _save_base(converted)
    return len(converted)


def add_reference_pdf(path: str | Path, display_name: str) -> dict[str, int | str]:
    """Incorpora un PDF a la biblioteca de referencia."""
    text = extract_pdf_text(path)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    base = [item for item in _load_base() if item.get("documento") != display_name]
    chunks = chunk_text(text)
    base.extend(
        {
            "documento": display_name,
            "fragmento": str(i + 1),
            "texto": chunk,
            "hash": digest,
        }
        for i, chunk in enumerate(chunks)
    )
    _save_base(base)
    return {"documento": display_name, "fragmentos": len(chunks)}


def remove_reference(display_name: str) -> None:
    """Elimina un documento de la biblioteca por nombre."""
    base = [item for item in _load_base() if item.get("documento") != display_name]
    _save_base(base)


def references_summary() -> list[dict[str, int | str]]:
    """Devuelve un listado de documentos con su cantidad de fragmentos."""
    summary: dict[str, int] = {}
    for item in _load_base():
        summary[item["documento"]] = summary.get(item["documento"], 0) + 1
    return [{"documento": name, "fragmentos": count} for name, count in sorted(summary.items())]


def clear_references() -> None:
    """Vacia toda la biblioteca de referencia."""
    _save_base([])


# ===========================================================================
# Transcripcion de video con Whisper
# ===========================================================================

def _whisper_model(size: str):
    if size not in _whisper_models:
        import whisper
        _whisper_models[size] = whisper.load_model(size)
    return _whisper_models[size]


def transcribe_video(path: str | Path, model_size: str = "tiny") -> str:
    """Transcribe el audio de un video con Whisper."""
    import shutil

    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg no esta disponible en el entorno."
        )
    if model_size not in {"tiny", "base", "small"}:
        model_size = "tiny"

    model = _whisper_model(model_size)
    result = model.transcribe(str(path), language="es", fp16=False, verbose=False)
    text = result["text"].strip()
    if not text:
        raise ValueError("Whisper no detecto habla en el video. Verifica que tenga audio.")
    return text


# ===========================================================================
# Motor semantico (embeddings + similitud coseno)
# ===========================================================================

def _embed_sentence_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_embeddings(base: list[dict[str, str]]):
    """Devuelve embeddings de la base, usando cache si esta vigente."""
    from sentence_transformers import util
    import torch

    fingerprint = hashlib.sha256(
        "".join(item["hash"] for item in base).encode()
    ).hexdigest()

    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return torch.tensor(cached["values"], dtype=torch.float32), util

    model = _embed_sentence_model()
    values = model.encode(
        [item["texto"] for item in base], convert_to_tensor=True, show_progress_bar=False
    )
    CACHE_PATH.write_text(
        json.dumps({"fingerprint": fingerprint, "values": values.tolist()}),
        encoding="utf-8",
    )
    return values, util


def compare_text(text: str, limit: int = 8) -> dict[str, Any]:
    """Compara un texto contra la biblioteca de referencia."""
    base = _load_base()
    if not base:
        raise ValueError(
            "La biblioteca esta vacia. "
            "Carga al menos un PDF de referencia en la pestana Biblioteca."
        )
    if not text.strip():
        raise ValueError("El archivo analizado no contiene texto util.")

    model = _embed_sentence_model()
    embeddings, util = _get_embeddings(base)

    queries = model.encode(
        chunk_text(text, size=900, overlap=100), convert_to_tensor=True, show_progress_bar=False
    )
    scores = util.cos_sim(queries, embeddings).max(dim=0).values
    best = scores.topk(min(limit, len(base)))

    matches = []
    for index, score in zip(best.indices, best.values):
        item = base[int(index)]
        matches.append(
            {
                "documento": item["documento"],
                "fragmento": item["fragmento"],
                "puntaje": round(float(score) * 100, 1),
                "texto": item["texto"],
            }
        )

    average = round(sum(m["puntaje"] for m in matches) / len(matches), 1)
    return {"promedio": average, "coincidencias": matches, "texto": text}


# ===========================================================================
# Historial de resultados
# ===========================================================================

def save_result(kind: str, filename: str, result: dict) -> None:
    """Guarda un resultado en el historial persistente (JSON)."""
    from datetime import datetime, timezone

    DATA_DIR.mkdir(exist_ok=True)
    previous = (
        json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if RESULTS_PATH.exists()
        else []
    )
    previous.append(
        {
            "fecha": datetime.now(timezone.utc).isoformat(),
            "tipo": kind,
            "archivo": filename,
            "promedio": result["promedio"],
            "coincidencias": result["coincidencias"],
        }
    )
    RESULTS_PATH.write_text(
        json.dumps(previous[-MAX_HISTORIAL:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_historial() -> list[dict]:
    """Devuelve el historial de analisis (mas reciente primero)."""
    if not RESULTS_PATH.exists():
        return []
    return list(reversed(json.loads(RESULTS_PATH.read_text(encoding="utf-8"))))
