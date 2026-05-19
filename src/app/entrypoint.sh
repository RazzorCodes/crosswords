#!/bin/sh

# If MODEL_RELEASE_TAG is set, we construct the GitHub Release URL.
# Otherwise, we use the provided MODEL_BASE_URL or fallback to /models.
# Example: GITHUB_REPOSITORY=andrei/crosswords, MODEL_RELEASE_TAG=v1.0.0
# URL: https://github.com/andrei/crosswords/releases/download/v1.0.0

FINAL_BASE_URL="/models"
APP_ROOT="${APP_ROOT:-/usr/share/nginx/html}"
CLIENT_MODEL_BASE_URL="/models"
FINAL_SRV_URL="${SRV_URL:-}"

if [ -n "$MODEL_RELEASE_TAG" ] && [ -n "$GITHUB_REPOSITORY" ]; then
    FINAL_BASE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${MODEL_RELEASE_TAG}"
elif [ -n "$MODEL_BASE_URL" ]; then
    FINAL_BASE_URL="$MODEL_BASE_URL"
fi

if [ "$FINAL_BASE_URL" != "/models" ]; then
    export UPSTREAM_MODEL_BASE_URL="$FINAL_BASE_URL"
fi

cat <<EOF > "${APP_ROOT}/env-config.js"
window.CROSSWORDS_CONFIG = {
  MODEL_BASE_URL: "${CLIENT_MODEL_BASE_URL}",
  SRV_URL: "${FINAL_SRV_URL}",
};
EOF

echo "Starting release server with MODEL_BASE_URL=${CLIENT_MODEL_BASE_URL} upstream=${UPSTREAM_MODEL_BASE_URL:-local} srv=${FINAL_SRV_URL:-disabled}"
exec node /server.mjs
