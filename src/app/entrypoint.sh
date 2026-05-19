#!/bin/sh

# If MODEL_RELEASE_TAG is set, we construct the GitHub Release URL.
# Otherwise, we use the provided MODEL_BASE_URL or fallback to /models.
# Example: GITHUB_REPOSITORY=andrei/crosswords, MODEL_RELEASE_TAG=v1.0.0
# URL: https://github.com/andrei/crosswords/releases/download/v1.0.0

FINAL_BASE_URL="/models"

if [ -n "$MODEL_RELEASE_TAG" ] && [ -n "$GITHUB_REPOSITORY" ]; then
    FINAL_BASE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${MODEL_RELEASE_TAG}"
elif [ -n "$MODEL_BASE_URL" ]; then
    FINAL_BASE_URL="$MODEL_BASE_URL"
fi

cat <<EOF > /usr/share/nginx/html/env-config.js
window.CROSSWORDS_CONFIG = {
  MODEL_BASE_URL: "${FINAL_BASE_URL}",
};
EOF

echo "Starting nginx with MODEL_BASE_URL=${FINAL_BASE_URL}"
exec nginx -g "daemon off;"
