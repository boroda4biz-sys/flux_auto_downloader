import os

try:
    import folder_paths
except Exception:  # pragma: no cover — вне ComfyUI
    folder_paths = None

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None


def _models_root() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.models_dir
        except Exception:
            pass
    return "/workspace/runpod-slim/ComfyUI/models"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


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

        # 1) Base Flux weights from Hugging Face (gated: needs HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
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

                # CLIP DualCLIPLoader в Comfy 0.3x смотрит text_encoders (как в Missing models UI)
                base_models = [
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
                ]

                for item in base_models:
                    target_folder = os.path.join(models_root, item["folder"])
                    os.makedirs(target_folder, exist_ok=True)
                    target_path = os.path.join(target_folder, item["file"])

                    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                        print(f"[AutoDownloader] Базовый файл уже есть: {item['file']}")
                        continue

                    print(f"[AutoDownloader] HF download {item['file']} ← {item['repo']} ...")
                    try:
                        downloaded = hf_hub_download(
                            repo_id=item["repo"],
                            filename=item["file"],
                            local_dir=target_folder,
                            token=token,
                        )
                        # На случай если hub положил не туда / с другим именем
                        if downloaded and os.path.abspath(downloaded) != os.path.abspath(target_path):
                            if os.path.isfile(downloaded) and not os.path.exists(target_path):
                                os.replace(downloaded, target_path)
                        print(f"[AutoDownloader] OK: {target_path}")
                    except Exception as e:
                        print(f"[AutoDownloader] Ошибка HF {item['file']}: {e}")

        # 2) LoRA с Google Drive (как раньше)
        loras_dir = os.path.join(models_root, "loras")
        os.makedirs(loras_dir, exist_ok=True)

        lines = (gdrive_loras_list or "").strip().split("\n")
        for line in lines:
            if "|" not in line:
                continue

            url_part, filename_part = line.split("|", 1)
            url = url_part.strip()
            filename = filename_part.strip()
            if not url or not filename:
                continue

            target_path = os.path.join(loras_dir, filename)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                print(f"[AutoDownloader] LoRA уже есть: {filename}")
                continue

            print(f"[AutoDownloader] GDrive download {filename} ...")

            file_id = None
            if "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]

            if not file_id:
                print(f"[AutoDownloader] Не разобрал Drive ID: {url}")
                continue

            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            cmd = (
                f"aria2c --max-connection-per-server=16 -x16 -s16 --continue=true "
                f"-o '{filename}' -d '{loras_dir}' '{direct_url}'"
            )
            os.system(cmd)

        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "FluxAutoDownloaderNode": FluxAutoDownloaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoDownloaderNode": "⚡ FLUX & Multi-LoRA Downloader",
}
