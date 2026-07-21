#!/usr/bin/env bash
set -Eeuo pipefail

ARCHIVE_PATH="${1:?release archive path is required}"
RELEASE_ID="${2:?release ID is required}"
APP_ROOT="/opt/deplab"
RELEASES_ROOT="${APP_ROOT}/releases"
SHARED_ROOT="${APP_ROOT}/shared"
RELEASE_PATH="${RELEASES_ROOT}/${RELEASE_ID}"
CURRENT_LINK="${APP_ROOT}/current"

if [[ ! "${RELEASE_ID}" =~ ^[0-9a-f]{40}-[0-9]+-[0-9]+$ ]]; then
  echo "Invalid release ID." >&2
  exit 2
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Release archive was not uploaded." >&2
  exit 2
fi

if [[ ! -f "${SHARED_ROOT}/.env" ]]; then
  echo "Missing ${SHARED_ROOT}/.env. EC2 must be provisioned before deployment." >&2
  exit 3
fi

required_artifacts=(
  "deplab-expanded-development-v2.0.0/features.csv"
  "deplab-expanded-weighted-logistic-v2.0.0/model.json"
  "deplab-advanced-model-comparison-v3.0.0/model.json"
  "deplab-hybrid-validation-v3.0.0/metrics.json"
)

for artifact in "${required_artifacts[@]}"; do
  if [[ ! -f "${SHARED_ROOT}/outputs/${artifact}" ]]; then
    echo "Missing private model artifact: ${artifact}" >&2
    exit 3
  fi
done

if [[ -e "${RELEASE_PATH}" ]]; then
  echo "Release path already exists: ${RELEASE_PATH}" >&2
  exit 4
fi

mkdir -p "${RELEASES_ROOT}" "${RELEASE_PATH}"
tar -xzf "${ARCHIVE_PATH}" -C "${RELEASE_PATH}" --no-same-owner
chown -R root:root "${RELEASE_PATH}"
chmod -R go-w "${RELEASE_PATH}"
rm -rf "${RELEASE_PATH}/outputs"
ln -s "${SHARED_ROOT}/outputs" "${RELEASE_PATH}/outputs"
ln -s "${SHARED_ROOT}/.env" "${RELEASE_PATH}/.env"

python3 -m venv "${RELEASE_PATH}/.venv"
"${RELEASE_PATH}/.venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"${RELEASE_PATH}/.venv/bin/python" -m pip install --disable-pip-version-check -e "${RELEASE_PATH}[api]"
"${RELEASE_PATH}/.venv/bin/python" -m py_compile "${RELEASE_PATH}/src/deplab/api/main.py"

previous_release=""
if [[ -L "${CURRENT_LINK}" ]]; then
  previous_release="$(readlink -f "${CURRENT_LINK}")"
fi

ln -s "${RELEASE_PATH}" "${APP_ROOT}/current.next"
mv -Tf "${APP_ROOT}/current.next" "${CURRENT_LINK}"
systemctl restart deplab-api.service

healthy=false
for _ in {1..15}; do
  if curl --fail --silent --show-error "http://127.0.0.1:8000/api/v1/health" >/dev/null; then
    healthy=true
    break
  fi
  sleep 2
done

if [[ "${healthy}" != "true" ]]; then
  echo "Health check failed; restoring the previous release." >&2
  if [[ -n "${previous_release}" && -d "${previous_release}" ]]; then
    ln -s "${previous_release}" "${APP_ROOT}/current.rollback"
    mv -Tf "${APP_ROOT}/current.rollback" "${CURRENT_LINK}"
    systemctl restart deplab-api.service
  fi
  exit 5
fi

nginx -t
systemctl reload nginx
rm -f "${ARCHIVE_PATH}" /tmp/deploy_release.sh
echo "DepLab release ${RELEASE_ID} is active."
