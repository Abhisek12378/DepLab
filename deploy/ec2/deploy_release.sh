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

MODEL_ROOT="${SHARED_ROOT}/models/current"
required_model_artifacts=(
  "outputs/deplab-large-features-v3.0.0/development-features.csv"
  "outputs/deplab-large-features-v3.0.0/validation-inputs.csv"
  "outputs/deplab-large-candidate-freeze-v3.0.0/candidate-structured_weighted_logistic.json"
  "outputs/deplab-large-candidate-freeze-v3.0.0/candidate-modernbert_stage_aware_hybrid.json"
  "outputs/large-release-modernbert-v3.0.0.jsonl"
)

for artifact in "${required_model_artifacts[@]}"; do
  if [[ ! -f "${MODEL_ROOT}/${artifact}" ]]; then
    echo "Missing production model artifact: ${artifact}" >&2
    exit 3
  fi
done

if ! grep -Eq '^DEPLAB_MODEL_ROOT=/opt/deplab/shared/models/current$' \
  "${SHARED_ROOT}/.env"; then
  echo "DEPLAB_MODEL_ROOT is missing from ${SHARED_ROOT}/.env." >&2
  exit 3
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "The uv resolver is not installed for the production service." >&2
  exit 3
fi

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
# systemd loads the protected shared environment file; releases must not link it.
install -d -m 755 -o deplab -g deplab "${SHARED_ROOT}/uv-cache"
install -m 644 \
  "${RELEASE_PATH}/deploy/ec2/deplab-api.service" \
  /etc/systemd/system/deplab-api.service
systemctl daemon-reload

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
  echo "DepLab service status for the failed release:" >&2
  systemctl status deplab-api.service --no-pager --full >&2 || true
  echo "Recent DepLab service logs:" >&2
  journalctl -u deplab-api.service -n 80 --no-pager >&2 || true
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
