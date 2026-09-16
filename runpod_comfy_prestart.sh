#!/usr/bin/env bash
# RunPod PID1 wrapper for runpod/comfyui (template uwflr4zwaj).
#
# Образ: ENTRYPOINT=/start.sh. Одно поле CMD часто игнорируется.
# В шаблоне RunPod нужен JSON с переопределением entrypoint:
#
#   {"entrypoint":["/bin/bash","-c"],"cmd":["curl -fsSL https://raw.githubusercontent.com/boroda4biz-sys/flux_auto_downloader/main/runpod_comfy_prestart.sh | bash"]}
#
# Bake path (из start.sh образа): cp -r /opt/comfyui-baked → /workspace/runpod-slim/ComfyUI
# Значит ноду кладём в /opt/comfyui-baked/custom_nodes ДО exec /start.sh.
#
# MATRIX_INSTALL_ONLY=1 — только clone, без exec /start.sh (для ручных хуков; не для sed-рекурсии).

set -euo pipefail

REPO_URL="${FLUX_AUTO_DOWNLOADER_REPO:-https://github.com/boroda4biz-sys/flux_auto_downloader.git}"
ORIG_ENTRY="${RUNPOD_ORIG_ENTRY:-/start.sh}"
COMFY_ROOT="${COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}"
BAKED_ROOT="${BAKED_COMFYUI_DIR:-/opt/comfyui-baked}"
INSTALL_ONLY="${MATRIX_INSTALL_ONLY:-0}"

install_into() {
  local nodes_dir="$1"
  local node_dir="${nodes_dir}/flux_auto_downloader"

  mkdir -p "${nodes_dir}"
  echo "[matrix-prestart] ensuring node in ${node_dir}"

  if [[ -d "${node_dir}/.git" ]]; then
    git -C "${node_dir}" fetch --depth 1 origin main 2>/dev/null \
      || git -C "${node_dir}" fetch --depth 1 origin master 2>/dev/null \
      || true
    git -C "${node_dir}" reset --hard FETCH_HEAD 2>/dev/null \
      || git -C "${node_dir}" pull --ff-only \
      || true
  elif [[ -d "${node_dir}" ]]; then
    rm -rf "${node_dir}"
    git clone --depth 1 "${REPO_URL}" "${node_dir}"
  else
    git clone --depth 1 "${REPO_URL}" "${node_dir}"
  fi

  if [[ -f "${node_dir}/__init__.py" ]]; then
    echo "[matrix-prestart] OK: ${node_dir}/__init__.py"
    return 0
  fi
  echo "[matrix-prestart] ERROR: __init__.py missing in ${node_dir}"
  return 1
}

echo "[matrix-prestart] begin pid=$$ INSTALL_ONLY=${INSTALL_ONLY}"

installed=0

# 1) Bake = корень ComfyUI в образе (НЕ .../ComfyUI/ComfyUI)
if [[ -d "${BAKED_ROOT}" ]]; then
  if install_into "${BAKED_ROOT}/custom_nodes"; then
    installed=1
  fi
fi

# 2) Workspace уже есть (рестарт / volume)
if [[ -d "${COMFY_ROOT}" ]] || [[ -d "${COMFY_ROOT}/custom_nodes" ]]; then
  if install_into "${COMFY_ROOT}/custom_nodes"; then
    installed=1
  fi
fi

if [[ "${installed}" -eq 0 ]]; then
  echo "[matrix-prestart] WARN: neither ${BAKED_ROOT} nor ${COMFY_ROOT} ready"
fi

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "[matrix-prestart] install-only done"
  exit 0
fi

if [[ -x "${ORIG_ENTRY}" ]]; then
  echo "[matrix-prestart] exec ${ORIG_ENTRY}"
  exec "${ORIG_ENTRY}" "$@"
fi

if [[ -f "${COMFY_ROOT}/main.py" ]]; then
  cd "${COMFY_ROOT}"
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

echo "[matrix-prestart] ERROR: no ${ORIG_ENTRY} and no main.py"
exit 1
