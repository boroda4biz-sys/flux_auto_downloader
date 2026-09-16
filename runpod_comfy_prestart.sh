#!/usr/bin/env bash
# RunPod template wrapper for runpod/comfyui.
#
# Важно про этот образ:
# - ENTRYPOINT обычно /start.sh. Одно поле «Container Start Command» часто
#   только подменяет CMD → наш curl игнорируется, Comfy стартует без ноды.
# - На первом буте custom_nodes в /workspace появляется ТОЛЬКО после того,
#   как /start.sh скопирует bake. Ждать workspace ДО /start.sh = бессмысленно.
#   Ставим ноду в bake (если есть) и/или в уже существующий workspace, потом exec /start.sh.
#
# В шаблоне RunPod предпочтительно JSON (entrypoint + cmd), иначе Start Command не выполнится:
#   {"entrypoint":["/bin/bash","-lc"],"cmd":["curl -fsSL https://raw.githubusercontent.com/boroda4biz-sys/flux_auto_downloader/main/runpod_comfy_prestart.sh | bash"]}
#
# Либо Entrypoint = /bin/bash и Start Command =
#   -lc "curl -fsSL https://raw.githubusercontent.com/boroda4biz-sys/flux_auto_downloader/main/runpod_comfy_prestart.sh | bash"

set -euo pipefail

REPO_URL="${FLUX_AUTO_DOWNLOADER_REPO:-https://github.com/boroda4biz-sys/flux_auto_downloader.git}"
ORIG_ENTRY="${RUNPOD_ORIG_ENTRY:-/start.sh}"
COMFY_ROOT="${COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}"

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

echo "[matrix-prestart] begin"

installed=0

# 1) Bake image copy source — на первом буте /start.sh копирует это в /workspace
for bake_nodes in \
  /opt/comfyui-baked/ComfyUI/custom_nodes \
  /opt/ComfyUI/custom_nodes \
  /ComfyUI/custom_nodes
do
  if [[ -d "$(dirname "${bake_nodes}")" ]] || [[ -d "${bake_nodes}" ]]; then
    if install_into "${bake_nodes}"; then
      installed=1
    fi
  fi
done

# 2) Уже распакованный workspace (рестарт пода / volume)
if [[ -d "${COMFY_ROOT}/custom_nodes" ]] || [[ -d "${COMFY_ROOT}" ]]; then
  if install_into "${COMFY_ROOT}/custom_nodes"; then
    installed=1
  fi
fi

if [[ "${installed}" -eq 0 ]]; then
  echo "[matrix-prestart] WARN: no bake/workspace Comfy yet — will rely on /start.sh copy, node may be missing on first attempt"
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
