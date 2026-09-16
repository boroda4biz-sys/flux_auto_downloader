#!/usr/bin/env bash
# RunPod template Start Command / Docker command wrapper for matrix-comic ComfyUI.
#
# Что делает:
# 1) Ждёт появления ComfyUI на диске (первый boot копирует bake → /workspace/runpod-slim)
# 2) Ставит/обновляет custom node flux_auto_downloader ДО старта Comfy
# 3) Запускает штатный entrypoint образа (передаёт все аргументы дальше)
#
# В RunPod Template → Container Start Command вставь ОДНУ строку (или сохрани файл и вызови):
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/boroda4biz-sys/flux_auto_downloader/main/runpod_prestart.sh | bash' 
# Или скопируй этот файл в шаблон / на volume и:
#   bash /workspace/runpod_comfy_prestart.sh
#
# Важно: clone ДО Comfy. Иначе UI стартует без ноды (как у вас уже было).

set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}"
NODES_DIR="${COMFY_ROOT}/custom_nodes"
REPO_URL="${FLUX_AUTO_DOWNLOADER_REPO:-https://github.com/boroda4biz-sys/flux_auto_downloader.git}"
NODE_DIR="${NODES_DIR}/flux_auto_downloader"
ORIG_ENTRY="${RUNPOD_ORIG_ENTRY:-/start.sh}"

echo "[matrix-prestart] waiting for ${NODES_DIR} ..."
for _ in $(seq 1 180); do
  if [[ -d "${NODES_DIR}" ]]; then
    break
  fi
  # Sometimes bake lands a second later
  if [[ -d /opt/comfyui-baked ]] && [[ ! -d "${COMFY_ROOT}" ]]; then
    echo "[matrix-prestart] waiting for bake → workspace copy..."
  fi
  sleep 2
done

if [[ ! -d "${NODES_DIR}" ]]; then
  echo "[matrix-prestart] ERROR: ${NODES_DIR} not found — cannot install custom node"
else
  echo "[matrix-prestart] ensuring flux_auto_downloader in ${NODE_DIR}"
  if [[ -d "${NODE_DIR}/.git" ]]; then
    git -C "${NODE_DIR}" fetch --depth 1 origin main 2>/dev/null || git -C "${NODE_DIR}" fetch --depth 1 origin master 2>/dev/null || true
    git -C "${NODE_DIR}" reset --hard FETCH_HEAD 2>/dev/null || git -C "${NODE_DIR}" pull --ff-only || true
  elif [[ -d "${NODE_DIR}" ]]; then
    # leftover broken folder without .git
    rm -rf "${NODE_DIR}"
    git clone --depth 1 "${REPO_URL}" "${NODE_DIR}"
  else
    git clone --depth 1 "${REPO_URL}" "${NODE_DIR}"
  fi

  if [[ -f "${NODE_DIR}/__init__.py" ]]; then
    echo "[matrix-prestart] OK: ${NODE_DIR}/__init__.py present"
  else
    echo "[matrix-prestart] ERROR: __init__.py missing after clone"
  fi
fi

# Hand off to image entrypoint (starts Jupyter + Comfy with node already on disk)
if [[ -x "${ORIG_ENTRY}" ]]; then
  echo "[matrix-prestart] exec ${ORIG_ENTRY} $*"
  exec "${ORIG_ENTRY}" "$@"
fi

# Fallback: start Comfy directly (slim images)
if [[ -f "${COMFY_ROOT}/main.py" ]]; then
  cd "${COMFY_ROOT}"
  # Prefer image venv if present
  if [[ -f .venv-cu128/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-cu128/bin/activate
  elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  echo "[matrix-prestart] fallback: python main.py"
  exec python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header
fi

echo "[matrix-prestart] ERROR: no entrypoint and no main.py"
exit 1
