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
    try:
        return os.path.isfile(path) and os.path.getsize(path) > _MIN_REAL_BYTES
    except OSError:
        return False


def _ensure_placeholder(path: str) -> bool:
    """Пустой stub, если файла нет. True = создали сейчас."""
    if os.path.exists(path):
        return False
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb"):
        pass
    print(f"[AutoDownloader] Placeholder: {path}")
    return True


def _invalidate_lora_filename_cache() -> None:
    if folder_paths is None:
        return
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop("loras", None)
        cache.clear()


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
        name = os.path.basename(str(raw or "").strip().replace("\\", "/"))
        if not name or name in (".", ".."):
            continue
        path = os.path.join(loras_dir, name)
        if _ensure_placeholder(path):
            created.append(name)
        else:
            existed.append(name)
    _invalidate_lora_filename_cache()
    return {"ok": True, "created": created, "existed": existed, "loras_dir": loras_dir}


def ensure_base_model_placeholders() -> None:
    """Фиксированные имена Flux — можно оставить в образе; каст не трогаем."""
    root = _models_root()
    for item in _BASE_MODEL_SPECS:
        folder = os.path.join(root, item["folder"])
        _ensure_placeholder(os.path.join(folder, item["file"]))


try:
    ensure_base_model_placeholders()
except Exception as e:  # pragma: no cover
    print(f"[AutoDownloader] WARN base placeholders: {e}")

# Опционально: FLUX_LORA_PLACEHOLDERS=a.safetensors,b.safetensors (редко нужно)
_extra = os.environ.get("FLUX_LORA_PLACEHOLDERS", "").strip()
if _extra:
    try:
        ensure_lora_placeholders(
            [p.strip() for p in _extra.replace(";", ",").split(",") if p.strip()]
        )
    except Exception as e:  # pragma: no cover
        print(f"[AutoDownloader] WARN env placeholders: {e}")


def _register_http_routes() -> None:
    """Бэкенд после boot: POST /flux_auto_downloader/ensure_placeholders {filenames:[...]}"""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as e:
        print(f"[AutoDownloader] HTTP routes skip: {e}")
        return

    @PromptServer.instance.routes.post("/flux_auto_downloader/ensure_placeholders")
    async def ensure_placeholders_handler(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        filenames = []
        if isinstance(data, dict):
            filenames = list(data.get("filenames") or data.get("files") or [])
        result = ensure_lora_placeholders(filenames)
        return web.json_response(result)

    print("[AutoDownloader] HTTP: POST /flux_auto_downloader/ensure_placeholders")


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
                        "default": "https://drive.google.com/file/d/.../view | lora_name.safetensors",
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
        if os.system("which aria2c") != 0:
            os.system("apt-get update && apt-get install -y aria2")

        models_root = _models_root()

        # 1) Base Flux weights from Hugging Face
        if download_base_models:
            if hf_hub_download is None:
                print(
                    "[AutoDownloader] huggingface_hub не установлен — "
                    "базовые модели Flux пропущены (pip install huggingface_hub)"
                )
            else:
                token = _hf_token()
                if not token:
                    print(
                        "[AutoDownloader] WARN: нет HF_TOKEN / HUGGING_FACE_HUB_TOKEN — "
                        "FLUX.1-dev gated, скачивание может упасть"
                    )

                for item in _BASE_MODEL_SPECS:
                    target_folder = os.path.join(models_root, item["folder"])
                    os.makedirs(target_folder, exist_ok=True)
                    target_path = os.path.join(target_folder, item["file"])

                    if _is_real_file(target_path):
                        print(f"[AutoDownloader] Базовый файл уже есть: {item['file']}")
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
                        downloaded = hf_hub_download(
                            repo_id=item["repo"],
                            filename=item["file"],
                            local_dir=target_folder,
                            token=token,
                        )
                        if downloaded and os.path.abspath(downloaded) != os.path.abspath(
                            target_path
                        ):
                            if os.path.isfile(downloaded) and not os.path.exists(target_path):
                                os.replace(downloaded, target_path)
                        print(f"[AutoDownloader] OK: {target_path}")
                    except Exception as e:
                        print(f"[AutoDownloader] Ошибка HF {item['file']}: {e}")
                        _ensure_placeholder(target_path)

        # 2) LoRA с Google Drive (список из карточек каста → бэкенд → этот input)
        loras_dir = os.path.join(models_root, "loras")
        os.makedirs(loras_dir, exist_ok=True)

        lines = (gdrive_loras_list or "").strip().split("\n")
        pending_names: list[str] = []
        for line in lines:
            if "|" not in line:
                continue
            _url_part, filename_part = line.split("|", 1)
            filename = filename_part.strip()
            if filename:
                pending_names.append(filename)

        # Stubs до кача — на случай если HTTP ensure ещё не вызывали
        ensure_lora_placeholders(pending_names)

        for line in lines:
            if "|" not in line:
                continue

            url_part, filename_part = line.split("|", 1)
            url = url_part.strip()
            filename = filename_part.strip()
            if not url or not filename:
                continue

            target_path = os.path.join(loras_dir, filename)

            if _is_real_file(target_path):
                print(f"[AutoDownloader] LoRA уже скачана: {filename}")
                continue

            print(
                f"[AutoDownloader] Обнаружена заглушка или файл отсутствует, качаем: {filename}"
            )

            file_id = None
            if "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]

            if not file_id:
                print(f"[AutoDownloader] Не разобрал Drive ID: {url}")
                continue

            if os.path.isfile(target_path) and not _is_real_file(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass

            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            cmd = (
                f"aria2c --max-connection-per-server=16 -x16 -s16 --continue=true "
                f"-o '{filename}' -d '{loras_dir}' '{direct_url}'"
            )
            rc = os.system(cmd)
            if rc != 0 or not _is_real_file(target_path):
                print(f"[AutoDownloader] WARN: скачивание LoRA не подтверждено: {filename}")
                _ensure_placeholder(target_path)

        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "FluxAutoDownloaderNode": FluxAutoDownloaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoDownloaderNode": "⚡ FLUX & Multi-LoRA Downloader",
}
