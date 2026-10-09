"""Reading images and documents attached to chat requests, so the rails can screen them.

OpenAI-style requests carry attachments as content parts:
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,..."}}

SafeGate reads the text out of each one (OCR for images and screenshots; the text layer of PDFs;
Word and plain-text files), screens it with the input rails, and rewrites the part before it is
forwarded: images are re-encoded without metadata, with any redacted words blacked out;
documents with redactions are replaced by their redacted text.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import io
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw

# Refuse images that would decompress into more pixels than this (decompression bombs).
Image.MAX_IMAGE_PIXELS = 40_000_000

_DATA_URL = re.compile(r"^data:(?P<type>[\w.+-]+/[\w.+-]+)?(?:;[^,]*)?;base64,(?P<data>.*)$", re.S)

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"}
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_TYPES = {"application/json", "application/xml", "application/x-yaml", "application/yaml"}
_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": DOCX_TYPE,
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class UnreadableAttachment(ValueError):
    """An attachment SafeGate cannot read, so it cannot vouch for it."""


@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]  # left, top, right, bottom in image pixels
    line: tuple[int, int, int] = (0, 0, 0)  # Tesseract's block, paragraph and line numbers
    start: int = 0  # offset of the word in the extracted text
    end: int = 0


@dataclass
class Attachment:
    part: dict  # the request's content part, rewritten in place after screening
    kind: str  # "image" or "file"
    media_type: str
    filename: str | None
    data: bytes | None  # None when SafeGate cannot get at the bytes (a URL or a file id)
    text: str = ""
    words: list[Word] = field(default_factory=list)  # images: OCR words with their boxes
    pages: int = 0
    hidden_words: int = 0  # images: words blacked out by redaction

    @property
    def label(self) -> str:
        return self.filename or self.media_type


def find_attachments(messages: list[Any], roles: list[str]) -> list[Attachment]:
    """Every image and file part in the messages of the screened roles."""
    found: list[Attachment] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in roles:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                media_type, data = _decode(url if isinstance(url, str) else "")
                found.append(Attachment(part, "image", media_type or "image", None, data))
            elif part.get("type") == "file" and isinstance(part.get("file"), dict):
                file = part["file"]
                filename = file.get("filename") if isinstance(file.get("filename"), str) else None
                media_type, data = _decode(file.get("file_data") or "")
                media_type = media_type or _guess_type(filename)
                found.append(Attachment(part, "file", media_type, filename, data))
    return found


def from_upload(name: str | None, media_type: str | None, data: str) -> Attachment:
    """An attachment for a file sent on its own (to /v1/check or the RAG index): `data` is a
    data: URL, or bare base64 with `media_type` set."""
    if not data.startswith("data:"):
        data = f"data:{media_type or 'application/octet-stream'};base64,{data}"
    part = {"type": "file", "file": {"filename": name, "file_data": data}}
    attachment = find_attachments([{"role": "user", "content": [part]}], ["user"])[0]
    if attachment.media_type in IMAGE_TYPES:
        attachment.kind = "image"
        attachment.part = {"type": "image_url", "image_url": {"url": data}}
    return attachment


def _decode(url: str) -> tuple[str | None, bytes | None]:
    match = _DATA_URL.match(url)
    if not match:
        return None, None
    try:
        return match["type"], base64.b64decode(match["data"], validate=False)
    except (binascii.Error, ValueError):
        return match["type"], None


def _guess_type(filename: str | None) -> str:
    if filename:
        for extension, media_type in _EXTENSIONS.items():
            if filename.lower().endswith(extension):
                return media_type
    return "application/octet-stream"


def extract(attachment: Attachment, *, max_bytes: int, max_pages: int, ocr_lang: str) -> None:
    """Fill in the attachment's text (and OCR words for images). Raises UnreadableAttachment."""
    if attachment.data is None:
        raise UnreadableAttachment(
            f"{attachment.label}: only inline base64 attachments can be screened"
        )
    if len(attachment.data) > max_bytes:
        raise UnreadableAttachment(f"{attachment.label}: larger than {max_bytes} bytes")
    media_type = attachment.media_type
    if attachment.kind == "image" or media_type in IMAGE_TYPES:
        attachment.words = ocr_words(_open_image(attachment), ocr_lang)
        attachment.text = _join_words(attachment.words)
    elif media_type == "application/pdf":
        attachment.text, attachment.pages = _pdf_text(attachment, max_pages)
    elif media_type == DOCX_TYPE:
        attachment.text = _docx_text(attachment)
    elif media_type.startswith("text/") or media_type in TEXT_TYPES:
        attachment.text = attachment.data.decode("utf-8", errors="replace")
    else:
        raise UnreadableAttachment(f"{attachment.label}: unsupported type {media_type}")


def _open_image(attachment: Attachment) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(attachment.data))
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise UnreadableAttachment(f"{attachment.label}: not a readable image ({exc})") from exc
    attachment.media_type = Image.MIME.get(image.format or "", attachment.media_type)
    return image


def ocr_words(image: Image.Image, lang: str = "eng") -> list[Word]:
    """Words Tesseract reads in the image, in reading order, with their bounding boxes."""
    import pytesseract

    gray = image.convert("L")
    # Tesseract reads small screenshot text far better at about twice the size.
    scale = 2 if max(gray.size) < 2000 else 1
    if scale > 1:
        gray = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(gray, lang=lang, output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i]) < 0:
            continue
        left, top = data["left"][i] // scale, data["top"][i] // scale
        right = left + data["width"][i] // scale
        bottom = top + data["height"][i] // scale
        line = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words.append(Word(text, (left, top, right, bottom), line))
    return words


def _join_words(words: list[Word]) -> str:
    """Words joined into lines, recording where each word sits in the text."""
    parts: list[str] = []
    offset = 0
    previous = None
    for word in words:
        if parts:
            separator = " " if word.line == previous else "\n"
            parts.append(separator)
            offset += 1
        word.start, word.end = offset, offset + len(word.text)
        parts.append(word.text)
        offset = word.end
        previous = word.line
    return "".join(parts)


def _pdf_text(attachment: Attachment, max_pages: int) -> tuple[str, int]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(attachment.data))
        if reader.is_encrypted:
            raise UnreadableAttachment(f"{attachment.label}: encrypted PDF")
        pages = reader.pages
        if len(pages) > max_pages:
            raise UnreadableAttachment(f"{attachment.label}: more than {max_pages} pages")
        text = "\n\n".join((page.extract_text() or "").strip() for page in pages)
    except (PdfReadError, ValueError, KeyError) as exc:
        raise UnreadableAttachment(f"{attachment.label}: not a readable PDF ({exc})") from exc
    return text.strip(), len(pages)


def _docx_text(attachment: Attachment) -> str:
    import zipfile

    from docx import Document

    try:
        document = Document(io.BytesIO(attachment.data))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise UnreadableAttachment(f"{attachment.label}: not a readable .docx ({exc})") from exc
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(line for line in lines if line.strip())


def changed_spans(original: str, redacted: str) -> list[tuple[int, int]]:
    """Character ranges of `original` that a redaction replaced."""
    matcher = difflib.SequenceMatcher(None, original, redacted, autojunk=False)
    return [(i1, i2) for tag, i1, i2, _, _ in matcher.get_opcodes() if tag in ("replace", "delete")]


def redact_image(attachment: Attachment, redacted_text: str) -> int:
    """Black out every OCR word the rails redacted, re-encode the image without metadata and
    write it back into the request. Returns the number of words blacked out."""
    image = _open_image(attachment)
    spans = (
        changed_spans(attachment.text, redacted_text) if redacted_text != attachment.text else []
    )
    hidden = [w for w in attachment.words if any(w.start < e and s < w.end for s, e in spans)]
    if hidden:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for word in hidden:
            left, top, right, bottom = word.box
            draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill="black")
    write_image(attachment, image)
    attachment.hidden_words = len(hidden)
    return len(hidden)


def write_image(attachment: Attachment, image: Image.Image) -> None:
    """Re-encode without EXIF and other metadata (location, camera, timestamps)."""
    fmt = "JPEG" if attachment.media_type == "image/jpeg" else "PNG"
    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    clean = Image.new(image.mode, image.size)
    clean.paste(image)
    out = io.BytesIO()
    clean.save(out, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    attachment.media_type = media_type
    attachment.data = out.getvalue()
    url = f"data:{media_type};base64,{base64.b64encode(attachment.data).decode()}"
    image_url = attachment.part.get("image_url")
    if isinstance(image_url, dict):
        image_url["url"] = url
    else:
        attachment.part["image_url"] = url


def replace_with_text(attachment: Attachment, text: str) -> None:
    """Forward a document as its (redacted) text instead of the original file."""
    attachment.part.clear()
    attachment.part.update({"type": "text", "text": f"[Attached file: {attachment.label}]\n{text}"})


def chunks(text: str, size: int) -> list[str]:
    """Split text into pieces of at most `size` characters, at paragraph or line breaks where
    possible, so each piece is screened like a prompt. Joining the pieces gives back the text."""
    pieces: list[str] = []
    rest = text
    while len(rest) > size:
        cut = max(rest.rfind("\n\n", 0, size), rest.rfind("\n", 0, size))
        if cut <= size // 4:
            cut = rest.rfind(" ", 0, size)
        if cut <= size // 4:
            cut = size
        pieces.append(rest[:cut])
        rest = rest[cut:]
    pieces.append(rest)
    return pieces
