import os
import subprocess

class FluxAutoDownloaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gdrive_loras_list": ("STRING", {
                    "multiline": True, 
                    "default": "https://drive.google.com/file/d/.../view | lora_name.safetensors"
                }),
                "trigger": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "download_and_pass"
    CATEGORY = "utils/flux"

    def download_and_pass(self, gdrive_loras_list, trigger):
        if os.system("which aria2c") != 0:
            os.system("apt-get update && apt-get install -y aria2c")

        loras_dir = "/workspace/runpod-slim/ComfyUI/models/loras"
        os.makedirs(loras_dir, exist_ok=True)

        lines = gdrive_loras_list.strip().split("\n")
        for line in lines:
            if "|" not in line:
                continue
            
            url_part, filename_part = line.split("|")
            url = url_part.strip()
            filename = filename_part.strip()
            
            target_path = os.path.join(loras_dir, filename)

            if os.path.exists(target_path):
                print(f"[AutoDownloader] Файл уже существует: {filename}")
                continue

            print(f"[AutoDownloader] Скачиваем {filename} с Google Drive...")
            
            file_id = None
            if "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]

            if file_id:
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                cmd = f"aria2c --max-connection-per-server=16 -x16 -s16 --continue=true -o '{filename}' -d '{loras_dir}' '{direct_url}'"
                os.system(cmd)

        return (trigger,)

NODE_CLASS_MAPPINGS = {
    "FluxAutoDownloaderNode": FluxAutoDownloaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAutoDownloaderNode": "⚡ FLUX & Multi-LoRA Downloader"
}
