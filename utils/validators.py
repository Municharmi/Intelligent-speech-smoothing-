def validate_file(file, config) -> tuple:
    if not file or file.filename == "":
        return False, "No file selected."
    fname = file.filename.lower()
    if "." not in fname:
        return False, "File must have an extension."
    ext = fname.rsplit(".", 1)[-1]
    cfg = config if isinstance(config, dict) else vars(type(config))
    allowed = set()
    if isinstance(config, dict):
        allowed = config.get("ALLOWED_AUDIO_EXTENSIONS",set()) | config.get("ALLOWED_VIDEO_EXTENSIONS",set())
    else:
        allowed = getattr(config,"ALLOWED_AUDIO_EXTENSIONS",set()) | getattr(config,"ALLOWED_VIDEO_EXTENSIONS",set())
    if ext not in allowed:
        return False, f"Unsupported file type .{ext}"
    return True, ""

def validate_text_input(text: str) -> tuple:
    if not text:          return False, "Text is empty."
    if len(text) < 2:     return False, "Text too short."
    if len(text) > 10000: return False, "Text too long."
    return True, ""
