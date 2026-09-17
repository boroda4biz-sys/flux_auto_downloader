import asyncio
import os

try:
    import folder_paths
except Exception:  # pragma: no cover — вне ComfyUI
    folder_paths = None

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None

# Файлы меньше порога = заглушки для UI-валидации Comfy до Queue.
_MIN_REAL_BYTES = 1024

_BASE_MODEL_SPECS = (
    {
        "repo": "black-forest-labs/FLUX.1-dev",
        "file": "flux1-dev.safetensors",
        "folder": "diffusion_models",
    },
    {
        "repo": "black-forest-labs/FLUX.1-dev",
        "file": "ae.safetensors",
        "folder": "vae",
    },
    {
        "repo": "comfyanonymous/flux_text_encoders",
        "file": "t5xxl_fp16.safetensors",
        "folder": "text_encoders",
    },
    {
        "repo": "comfyanonymous/flux_text_encoders",
        "file": "clip_l.safetensors",
        "folder": "text_encoders",
    },
)


def _models_root() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.models_dir
        except Exception:
            pass
    return "/workspace/runpod-slim/ComfyUI/models"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _is_real_file(path: str) -> bool:
    """Файл есть, не stub и похож на safetensors (не HTML от Drive)."""
    try:
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size <= _MIN_REAL_BYTES:
            return False
        # LoRA стиля/персонажа почти всегда > 1 МБ; Drive HTML / confirm — килобайты
        if path.endswith(".safetensors") and size < 1_000_000:
            return False
        with open(path, "rb") as f:
            head = f.read(64)
        if not head:
            return False
        low = head.lstrip()[:20].lower()
        if low.startswith(b"<!") or low.startswith(b"<html") or low.startswith(b"<!doctype"):
            return False
        # safetensors: 8-byte little-endian header length, then JSON starting with '{'
        if path.endswith(".safetensors") and size > 8:
            import struct

            (hlen,) = struct.unpack("<Q", head[:8])
            if hlen <= 0 or hlen > size - 8 or hlen > 100_000_000:
                return False
            if len(head) > 8 and head[8:9] != b"{":
                return False
        return True
    except OSError:
        return False


def _remove_bad_file(path: str, reason: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"[AutoDownloader] Удалён битый файл ({reason}): {path}")
    except OSError as e:
        print(f"[AutoDownloader] WARN remove {path}: {e}")


def _safe_lora_filename(raw: str) -> str:
    """Только имя файла в loras/, без путей и мусора после | / view."""
    s = str(raw or "").strip().replace("\\", "/")
    if not s or s in (".", ".."):
        return ""
    # отрезать хвост вроде sveta.safetensors/view
    if "/" in s:
        head = s.split("/")[0].strip()
        if head.endswith((".safetensors", ".pt")):
            s = head
        else:
            s = os.path.basename(s)
    s = os.path.basename(s).strip()
    if "|" in s or " " in s or ".." in s:
        return ""
    if not s.endswith((".safetensors", ".pt")):
        return ""
    return s


def _is_hf_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return (
        u.startswith("hf://")
        or "huggingface.co/" in u
        or "hf.co/" in u
    )


def _is_gdrive_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return "drive.google.com" in u or ("id=" in u and "google" in u)


def _parse_hf_repo_and_file(url: str, filename_hint: str = "") -> tuple[str, str] | None:
    """
    HF URL → (repo_id, filename_in_repo).
    Примеры:
      hf://user/repo/mtxcomic.safetensors
      https://huggingface.co/user/repo/resolve/main/mtxcomic.safetensors
      https://huggingface.co/user/repo/blob/main/foo/mtxcomic.safetensors
      https://huggingface.co/user/repo  (+ hint filename)
    """
    raw = (url or "").strip()
    if not raw:
        return None
    hint = _safe_lora_filename(filename_hint) or ""

    if raw.lower().startswith("hf://"):
        body = raw[5:].strip().strip("/")
        parts = [p for p in body.split("/") if p]
        if len(parts) < 2:
            return None
        repo_id = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 3:
            return repo_id, "/".join(parts[2:])
        if hint:
            return repo_id, hint
        return None

    # strip query
    path = raw.split("?", 1)[0]
    for host in ("https://huggingface.co/", "http://huggingface.co/", "https://hf.co/", "http://hf.co/"):
        if path.lower().startswith(host):
            path = path[len(host) :]
            break
    else:
        if "huggingface.co/" in path.lower():
            path = path.lower().split("huggingface.co/", 1)[1]
        elif "hf.co/" in path.lower():
            path = path.lower().split("hf.co/", 1)[1]
        else:
            return None

    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    rest = parts[2:]
    if rest and rest[0] in ("resolve", "blob", "raw"):
        # resolve/main/file or blob/main/path/file
        if len(rest) >= 3:
            return repo_id, "/".join(rest[2:])
        if hint:
            return repo_id, hint
        return None
    if rest:
        return repo_id, "/".join(rest)
    if hint:
        return repo_id, hint
    return None


def _parse_lora_source_line(line: str) -> tuple[str, str] | None:
    """Строка `url | name.safetensors` → (url, filename). Drive или Hugging Face."""
    line = (line or "").strip()
    if not line or line.startswith("#") or "|" not in line:
        return None
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) < 2:
        return None
    url = parts[0]
    filename = ""
    for part in parts[1:]:
        cand = _safe_lora_filename(part)
        if cand:
            filename = cand
            break
    if not filename:
        return None
    if not (_is_hf_url(url) or _is_gdrive_url(url)):
        return None
    return url, filename


# backward alias
_parse_gdrive_line = _parse_lora_source_line


def _download_hf_lora(url: str, filename: str, loras_dir: str) -> tuple[bool, str]:
    """Скачать LoRA с HF (stream + прогресс в _LORA_DOWNLOAD_STATE)."""
    token = _hf_token()
    parsed = _parse_hf_repo_and_file(url, filename)
    if not parsed:
        return False, f"не разобрал HF URL: {url}"
    repo_id, repo_file = parsed
    target_path = os.path.join(loras_dir, filename)
    os.makedirs(loras_dir, exist_ok=True)
    print(f"[AutoDownloader] HF LoRA {filename} ← {repo_id}/{repo_file}")

    # Прямой resolve URL — стрим с прогрессом (понятнее, чем тихий hf_hub_download)
    resolve_url = url.strip()
    if "resolve/" not in resolve_url.lower():
        resolve_url = f"https://huggingface.co/{repo_id}/resolve/main/{repo_file}"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        import urllib.request

        req = urllib.request.Request(resolve_url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = 0
            try:
                total = int(resp.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                total = 0
            _set_lora_progress(
                filename=filename, current_bytes=0, total_bytes=total, message=""
            )
            tmp_path = target_path + ".part"
            written = 0
            last_report = 0
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if written - last_report >= 2 * 1024 * 1024 or written == total:
                        last_report = written
                        _set_lora_progress(
                            filename=filename,
                            current_bytes=written,
                            total_bytes=total,
                        )
                        print(
                            f"[AutoDownloader] HF {filename}: {_fmt_bytes(written)}"
                            + (f" / {_fmt_bytes(total)}" if total else "")
                        )
            if os.path.isfile(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            os.replace(tmp_path, target_path)

        if _is_real_file(target_path):
            _set_lora_progress(
                filename=filename,
                current_bytes=os.path.getsize(target_path),
                total_bytes=os.path.getsize(target_path),
                message=f"{filename}: готово",
            )
            return True, ""
        return False, "файл после HF download не похож на safetensors"
    except Exception as e:
        # fallback: huggingface_hub (без побайтового прогресса в файл)
        if hf_hub_download is None:
            return False, str(e)
        print(f"[AutoDownloader] HF stream fail ({e}) — fallback hf_hub_download")
        try:
            _set_lora_progress(
                filename=filename,
                message=f"{filename}: hf_hub_download…",
            )
            got = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                local_dir=loras_dir,
                token=token,
            )
            if got and os.path.isfile(got):
                abs_got = os.path.abspath(got)
                abs_tgt = os.path.abspath(target_path)
                if abs_got != abs_tgt:
                    if os.path.isfile(target_path):
                        try:
                            os.remove(target_path)
                        except OSError:
                            pass
                    os.replace(got, target_path)
            if _is_real_file(target_path):
                sz = os.path.getsize(target_path)
                _set_lora_progress(
                    filename=filename,
                    current_bytes=sz,
                    total_bytes=sz,
                    message=f"{filename}: готово",
                )
                return True, ""
            return False, "файл после HF download не похож на safetensors"
        except Exception as e2:
            return False, str(e2)


def _download_gdrive_lora(url: str, filename: str, loras_dir: str) -> tuple[bool, str]:
    """Скачать LoRA с Google Drive через aria2. Возвращает (ok, error)."""
    target_path = os.path.join(loras_dir, filename)
    if "/folders/" in url or "/drive/folders/" in url:
        return False, (
            f"это ссылка на ПАПКУ Drive, нужна ссылка на ФАЙЛ "
            f"(.safetensors): …/file/d/ID/view — сейчас: {url}"
        )
    file_id = None
    if "/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    if not file_id or file_id.startswith("folders"):
        return False, f"Не разобрал Drive FILE id (нужен /file/d/…): {url}"

    direct_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    _set_lora_progress(filename=filename, message=f"{filename}: aria2 Drive…")
    cmd = (
        f"aria2c --max-connection-per-server=16 -x16 -s16 --continue=true "
        f"--allow-overwrite=true --auto-file-renaming=false "
        f"-o '{filename}' -d '{loras_dir}' '{direct_url}'"
    )
    # фоновый монитор размера файла пока крутится aria2
    import subprocess
    import threading
    import time

    stop_mon = threading.Event()

    def _mon() -> None:
        while not stop_mon.wait(1.5):
            try:
                if os.path.isfile(target_path):
                    sz = os.path.getsize(target_path)
                    _set_lora_progress(filename=filename, current_bytes=sz)
            except OSError:
                pass

    mon = threading.Thread(target=_mon, name="aria2-size", daemon=True)
    mon.start()
    try:
        rc = subprocess.call(cmd, shell=True)
    finally:
        stop_mon.set()
        mon.join(timeout=2)

    if rc != 0 or not _is_real_file(target_path):
        sniff = ""
        try:
            if os.path.isfile(target_path):
                with open(target_path, "rb") as f:
                    sniff = f.read(40).lstrip()[:20].lower().decode("latin-1", "ignore")
        except OSError:
            pass
        why = "HTML/virus-scan Drive" if sniff.startswith("<!") or sniff.startswith("<html") else "invalid"
        if os.path.isfile(target_path):
            _remove_bad_file(target_path, f"failed download / {why}")
        return False, (
            f"скачивание не дало настоящий .safetensors ({why}); "
            f"для файлов >~100МБ лучше Hugging Face URL + HF_TOKEN"
        )
    sz = os.path.getsize(target_path)
    _set_lora_progress(
        filename=filename,
        current_bytes=sz,
        total_bytes=sz,
        message=f"{filename}: готово",
    )
    return True, ""


def _ensure_placeholder(path: str) -> bool:
    """Пустой stub, если файла нет. True = создали сейчас."""
    try:
        if os.path.isfile(path):
            return False
        if os.path.isdir(path):
            return False
        parent = os.path.dirname(path)
        if parent:
            # если parent — уже файл (битое имя с /), не лезем в makedirs
            if os.path.isfile(parent):
                print(f"[AutoDownloader] WARN: parent is a file, skip stub: {path}")
                return False
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb"):
            pass
        print(f"[AutoDownloader] Placeholder: {path}")
        return True
    except OSError as e:
        print(f"[AutoDownloader] WARN placeholder {path}: {e}")
        return False


def _invalidate_lora_filename_cache() -> None:
    if folder_paths is None:
        return
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop("loras", None)
        cache.clear()


def _ensure_aria2() -> None:
    if os.system("which aria2c") != 0:
        os.system("apt-get update && apt-get install -y aria2")


def ensure_lora_placeholders(filenames: list[str]) -> dict:
    """
    Создаёт 0-byte stubs в models/loras по именам из карточек персонажей/стилей.
    Имена не хардкодятся в образе — приходят с бэкенда/API.
    """
    root = _models_root()
    loras_dir = os.path.join(root, "loras")
    os.makedirs(loras_dir, exist_ok=True)
    created: list[str] = []
    existed: list[str] = []
    for raw in filenames or []:
        name = _safe_lora_filename(raw)
        if not name:
            print(f"[AutoDownloader] skip bad LoRA name: {raw!r}")
            continue
        path = os.path.join(loras_dir, name)
        if _ensure_placeholder(path):
            created.append(name)
        else:
            existed.append(name)
    _invalidate_lora_filename_cache()
    return {"ok": True, "created": created, "existed": existed, "loras_dir": loras_dir}


def ensure_base_model_placeholders() -> None:
    """Фиксированные имена Flux — stubs до реального HF download."""
    root = _models_root()
    for item in _BASE_MODEL_SPECS:
        folder = os.path.join(root, item["folder"])
        _ensure_placeholder(os.path.join(folder, item["file"]))


def download_base_models_now() -> dict:
    """Скачать базовые веса Flux с HF (вызов с Boot / Queue)."""
    ensure_base_model_placeholders()
    if hf_hub_download is None:
        msg = "huggingface_hub не установлен"
        print(f"[AutoDownloader] {msg}")
        return {"ok": False, "error": msg, "downloaded": [], "skipped": []}

    token = _hf_token()
    if not token:
        print(
            "[AutoDownloader] WARN: нет HF_TOKEN / HUGGING_FACE_HUB_TOKEN — "
            "FLUX.1-dev gated, скачивание может упасть"
        )

    models_root = _models_root()
    downloaded: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for item in _BASE_MODEL_SPECS:
        target_folder = os.path.join(models_root, item["folder"])
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, item["file"])

        if _is_real_file(target_path):
            print(f"[AutoDownloader] Базовый файл уже есть: {item['file']}")
            skipped.append(item["file"])
            continue

        print(
            f"[AutoDownloader] Заглушка/нет файла — HF download "
            f"{item['file']} ← {item['repo']} ..."
        )
        if os.path.isfile(target_path) and not _is_real_file(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass

        try:
            got = hf_hub_download(
                repo_id=item["repo"],
                filename=item["file"],
                local_dir=target_folder,
                token=token,
            )
            if got and os.path.abspath(got) != os.path.abspath(target_path):
                if os.path.isfile(got) and not os.path.exists(target_path):
                    os.replace(got, target_path)
            if _is_real_file(target_path):
                print(f"[AutoDownloader] OK: {target_path}")
                downloaded.append(item["file"])
            else:
                errors.append(f"{item['file']}: file missing after download")
                _ensure_placeholder(target_path)
        except Exception as e:
            print(f"[AutoDownloader] Ошибка HF {item['file']}: {e}")
            errors.append(f"{item['file']}: {e}")
            _ensure_placeholder(target_path)

    return {
        "ok": len(errors) == 0,
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
    }


def download_loras_from_list(gdrive_loras_list: str) -> dict:
    """Скачать LoRA по строкам `url | filename` — Google Drive или Hugging Face."""
    _ensure_aria2()
    models_root = _models_root()
    loras_dir = os.path.join(models_root, "loras")
    os.makedirs(loras_dir, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for line in (gdrive_loras_list or "").splitlines():
        parsed = _parse_lora_source_line(line)
        if not parsed:
            if line.strip() and "|" in line:
                print(f"[AutoDownloader] skip bad line: {line.strip()[:120]}")
            continue
        rows.append(parsed)

    stubs = ensure_lora_placeholders([fn for _url, fn in rows])
    downloaded: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for url, filename in rows:
        target_path = os.path.join(loras_dir, filename)

        if _is_real_file(target_path):
            print(f"[AutoDownloader] LoRA уже скачана: {filename}")
            skipped.append(filename)
            continue

        print(
            f"[AutoDownloader] Обнаружена заглушка или файл отсутствует, качаем: {filename}"
        )
        _set_lora_progress(filename=filename, message=f"качаем {filename}…")

        if os.path.isfile(target_path) and not _is_real_file(target_path):
            _remove_bad_file(target_path, "stub or invalid before download")

        if _is_hf_url(url):
            ok, err = _download_hf_lora(url, filename, loras_dir)
        elif _is_gdrive_url(url):
            ok, err = _download_gdrive_lora(url, filename, loras_dir)
        else:
            ok, err = False, f"неизвестный источник (нужен Drive или Hugging Face): {url}"

        if ok:
            downloaded.append(filename)
            print(f"[AutoDownloader] OK LoRA: {filename}")
        else:
            print(f"[AutoDownloader] WARN LoRA {filename}: {err}")
            errors.append(f"{filename}: {err}")
            # НЕ ставим placeholder — иначе LoraLoader → JSONDecodeError

    return {
        "ok": len(errors) == 0,
        "stubs": stubs,
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
    }


def start_loras_download_background(gdrive_loras_list: str) -> dict:
    """Фоновое скачивание LoRA — proxy не режет длинный HTTP; статус через GET."""
    import threading

    if _LORA_DOWNLOAD_STATE.get("running"):
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "status": loras_status(),
        }

    text = gdrive_loras_list or ""

    def _worker() -> None:
        _LORA_DOWNLOAD_STATE["running"] = True
        _LORA_DOWNLOAD_STATE["message"] = "старт скачивания LoRA…"
        try:
            result = download_loras_from_list(text)
            _LORA_DOWNLOAD_STATE["last"] = result
            _LORA_DOWNLOAD_STATE["message"] = (
                "готово" if result.get("ok") else f"ошибки: {result.get('errors')}"
            )
        except Exception as e:  # pragma: no cover
            _LORA_DOWNLOAD_STATE["last"] = {"ok": False, "error": str(e)}
            _LORA_DOWNLOAD_STATE["message"] = str(e)
            print(f"[AutoDownloader] background LoRA download error: {e}")
        finally:
            _LORA_DOWNLOAD_STATE["running"] = False

    threading.Thread(target=_worker, name="flux-lora-download", daemon=True).start()
    names = []
    for line in text.splitlines():
        parsed = _parse_lora_source_line(line)
        if parsed:
            names.append(parsed[1])
    return {
        "ok": True,
        "started": True,
        "already_running": False,
        "status": loras_status(names or None),
    }


try:
    ensure_base_model_placeholders()
except Exception as e:  # pragma: no cover
    print(f"[AutoDownloader] WARN base placeholders: {e}")

_extra = os.environ.get("FLUX_LORA_PLACEHOLDERS", "").strip()
if _extra:
    try:
        ensure_lora_placeholders(
            [p.strip() for p in _extra.replace(";", ",").split(",") if p.strip()]
        )
    except Exception as e:  # pragma: no cover
        print(f"[AutoDownloader] WARN env placeholders: {e}")


_BASE_DOWNLOAD_STATE: dict = {
    "running": False,
    "last": None,
}

_LORA_DOWNLOAD_STATE: dict = {
    "running": False,
    "last": None,
    "current_file": "",
    "current_bytes": 0,
    "total_bytes": 0,
    "message": "",
}


def _fmt_bytes(n: int | float) -> str:
    try:
        b = float(n)
    except (TypeError, ValueError):
        return "0 B"
    if b < 1024:
        return f"{int(b)} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / (1024 * 1024 * 1024):.2f} GB"


def _set_lora_progress(
    *,
    filename: str = "",
    current_bytes: int = 0,
    total_bytes: int = 0,
    message: str = "",
) -> None:
    _LORA_DOWNLOAD_STATE["current_file"] = filename or ""
    _LORA_DOWNLOAD_STATE["current_bytes"] = int(current_bytes or 0)
    _LORA_DOWNLOAD_STATE["total_bytes"] = int(total_bytes or 0)
    if message:
        _LORA_DOWNLOAD_STATE["message"] = message
    elif filename:
        tot = int(total_bytes or 0)
        cur = int(current_bytes or 0)
        if tot > 0:
            pct = min(100.0, 100.0 * cur / tot)
            _LORA_DOWNLOAD_STATE["message"] = (
                f"{filename}: {_fmt_bytes(cur)} / {_fmt_bytes(tot)} ({pct:.0f}%)"
            )
        else:
            _LORA_DOWNLOAD_STATE["message"] = f"{filename}: {_fmt_bytes(cur)}"


def loras_status(filenames: list[str] | None = None) -> dict:
    """Готовность LoRA на диске (для poll / gate перед кадрами)."""
    models_root = _models_root()
    loras_dir = os.path.join(models_root, "loras")
    names = [str(x).strip() for x in (filenames or []) if str(x).strip()]
    if not names and os.path.isdir(loras_dir):
        names = sorted(
            f for f in os.listdir(loras_dir) if f.endswith((".safetensors", ".pt"))
        )
    files = []
    ready = 0
    for name in names:
        path = os.path.join(loras_dir, os.path.basename(name))
        size = 0
        exists = os.path.isfile(path)
        if exists:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
        # aria2 partial рядом
        for suffix in (".aria2", ".tmp", ".part"):
            p2 = path + suffix
            if os.path.isfile(p2):
                try:
                    size = max(size, os.path.getsize(p2))
                except OSError:
                    pass
        is_real = _is_real_file(path)
        if is_real:
            ready += 1
        files.append(
            {
                "file": os.path.basename(name),
                "exists": exists,
                "bytes": size,
                "ready": is_real,
                "human": _fmt_bytes(size),
            }
        )
    cur = str(_LORA_DOWNLOAD_STATE.get("current_file") or "")
    cur_b = int(_LORA_DOWNLOAD_STATE.get("current_bytes") or 0)
    tot_b = int(_LORA_DOWNLOAD_STATE.get("total_bytes") or 0)
    return {
        "ok": bool(names) and ready == len(names),
        "ready": ready,
        "total": len(names),
        "files": files,
        "loras_dir": loras_dir,
        "download_running": bool(_LORA_DOWNLOAD_STATE.get("running")),
        "current_file": cur,
        "current_bytes": cur_b,
        "total_bytes": tot_b,
        "message": str(_LORA_DOWNLOAD_STATE.get("message") or ""),
        "last": _LORA_DOWNLOAD_STATE.get("last"),
    }


def base_models_status() -> dict:
    """Размеры базовых файлов Flux на диске (для poll с бэкенда)."""
    models_root = _models_root()
    files = []
    ready = 0
    for item in _BASE_MODEL_SPECS:
        path = os.path.join(models_root, item["folder"], item["file"])
        size = 0
        exists = os.path.isfile(path)
        if exists:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
        is_real = _is_real_file(path)
        if is_real:
            ready += 1
        files.append(
            {
                "file": item["file"],
                "folder": item["folder"],
                "path": path,
                "exists": exists,
                "bytes": size,
                "ready": is_real,
                "human": _fmt_bytes(size),
            }
        )
    return {
        "ok": ready == len(_BASE_MODEL_SPECS),
        "ready": ready,
        "total": len(_BASE_MODEL_SPECS),
        "files": files,
        "download_running": bool(_BASE_DOWNLOAD_STATE.get("running")),
        "last": _BASE_DOWNLOAD_STATE.get("last"),
        "message": _format_files_progress(files, ready, len(_BASE_MODEL_SPECS), "Flux"),
    }


def _format_files_progress(files: list, ready: int, total: int, label: str) -> str:
    bits = []
    for f in files or []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("file") or "")
        if f.get("ready"):
            bits.append(f"{name} ✓ {_fmt_bytes(f.get('bytes') or 0)}")
        else:
            bits.append(f"{name} {_fmt_bytes(f.get('bytes') or 0)}")
    detail = " · ".join(bits[:4])
    return f"{label} {ready}/{total}" + (f" · {detail}" if detail else "")


def start_base_models_download_background() -> dict:
    """Старт HF-скачивания в фоне — proxy RunPod не режет длинный HTTP."""
    import threading

    if _BASE_DOWNLOAD_STATE.get("running"):
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "status": base_models_status(),
        }

    def _worker() -> None:
        _BASE_DOWNLOAD_STATE["running"] = True
        try:
            result = download_base_models_now()
            _BASE_DOWNLOAD_STATE["last"] = result
        except Exception as e:  # pragma: no cover
            _BASE_DOWNLOAD_STATE["last"] = {"ok": False, "error": str(e)}
            print(f"[AutoDownloader] background base download error: {e}")
        finally:
            _BASE_DOWNLOAD_STATE["running"] = False

    threading.Thread(target=_worker, name="flux-base-download", daemon=True).start()
    return {"ok": True, "started": True, "already_running": False, "status": base_models_status()}


def _register_http_routes() -> None:
    """HTTP для Boot из мониторинга (без Queue workflow)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as e:
        print(f"[AutoDownloader] HTTP routes skip: {e}")
        return

    routes = PromptServer.instance.routes

    @routes.post("/flux_auto_downloader/ensure_placeholders")
    async def ensure_placeholders_handler(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        filenames = []
        if isinstance(data, dict):
            filenames = list(data.get("filenames") or data.get("files") or [])
        result = await asyncio.to_thread(ensure_lora_placeholders, filenames)
        return web.json_response(result)

    @routes.post("/flux_auto_downloader/download_base")
    async def download_base_handler(request):
        # ?sync=1 — старое блокирующее поведение; по умолчанию фон (прокси не 502)
        sync = False
        try:
            if request.rel_url.query.get("sync") in ("1", "true", "yes"):
                sync = True
            else:
                data = await request.json()
                if isinstance(data, dict) and data.get("sync"):
                    sync = True
        except Exception:
            pass
        if sync:
            result = await asyncio.to_thread(download_base_models_now)
            return web.json_response(result)
        result = await asyncio.to_thread(start_base_models_download_background)
        return web.json_response(result)

    @routes.get("/flux_auto_downloader/download_base_status")
    async def download_base_status_handler(_request):
        return web.json_response(base_models_status())

    @routes.get("/flux_auto_downloader/loras_status")
    async def loras_status_handler(request):
        files_q = request.rel_url.query.get("files") or ""
        names = [p.strip() for p in files_q.split(",") if p.strip()]
        return web.json_response(loras_status(names or None))

    @routes.get("/flux_auto_downloader/download_loras_status")
    async def download_loras_status_handler(request):
        files_q = request.rel_url.query.get("files") or ""
        names = [p.strip() for p in files_q.split(",") if p.strip()]
        return web.json_response(loras_status(names or None))

    @routes.post("/flux_auto_downloader/download_loras")
    async def download_loras_handler(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        gdrive_list = ""
        sync = False
        if isinstance(data, dict):
            gdrive_list = str(data.get("gdrive_loras_list") or data.get("list") or "")
            sync = bool(data.get("sync"))
        if request.rel_url.query.get("sync") in ("1", "true", "yes"):
            sync = True
        if sync:
            result = await asyncio.to_thread(download_loras_from_list, gdrive_list)
            return web.json_response(result)
        result = await asyncio.to_thread(start_loras_download_background, gdrive_list)
        return web.json_response(result)

    print(
        "[AutoDownloader] HTTP: "
        "POST /flux_auto_downloader/ensure_placeholders | "
        "download_base | download_loras | "
        "GET download_base_status | loras_status | download_loras_status"
    )


try:
    _register_http_routes()
except Exception as e:  # pragma: no cover
    print(f"[AutoDownloader] WARN register routes: {e}")


class FluxAutoDownloaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gdrive_loras_list": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "trigger": ("MODEL",),
            },
            "optional": {
                "download_base_models": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "download_and_pass"
    CATEGORY = "utils/flux"

    def download_and_pass(self, gdrive_loras_list, trigger, download_base_models=True):
        if download_base_models:
            download_base_models_now()
        download_loras_from_list(gdrive_loras_list or "")
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "FluxAutoDownloaderNode": FluxAutoDownloaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoDownloaderNode": "⚡ FLUX & Multi-LoRA Downloader",
}
