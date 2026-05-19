#!/bin/sh

# If MODEL_RELEASE_TAG is set, we construct the GitHub Release URL.
# Otherwise, we use the provided MODEL_BASE_URL or fallback to /models.
# Example: GITHUB_REPOSITORY=andrei/crosswords, MODEL_RELEASE_TAG=v1.0.0
# URL: https://github.com/andrei/crosswords/releases/download/v1.0.0

FINAL_BASE_URL="/models"
APP_ROOT="${APP_ROOT:-/usr/share/nginx/html}"

if [ -n "$MODEL_RELEASE_TAG" ] && [ -n "$GITHUB_REPOSITORY" ]; then
    FINAL_BASE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${MODEL_RELEASE_TAG}"
elif [ -n "$MODEL_BASE_URL" ]; then
    FINAL_BASE_URL="$MODEL_BASE_URL"
fi

cat <<EOF > "${APP_ROOT}/env-config.js"
window.CROSSWORDS_CONFIG = {
  MODEL_BASE_URL: "${FINAL_BASE_URL}",
};
EOF

if command -v serve >/dev/null 2>&1; then
    echo "Starting serve with MODEL_BASE_URL=${FINAL_BASE_URL}"
    exec serve -s "${APP_ROOT}" -l 80
fi

echo "Starting nginx with MODEL_BASE_URL=${FINAL_BASE_URL}"
exec nginx -g "daemon off;"
