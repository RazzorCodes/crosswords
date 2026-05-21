#!/bin/bash
set -e
VERSION=$1
MODELS_SRC=${2:-models}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [models_src_dir]"
    exit 1
fi

echo "Packaging assets for version $VERSION..."
mkdir -p dist

# Package dataset (only regular and high_quality)
if [ -d data/regular ]; then
    zip -r dist/dataset-$VERSION.zip data/regular data/high_quality
else
    echo "Warning: data/regular not found, skipping dataset zip"
fi

# Package models
if [ -d "$MODELS_SRC" ]; then
    # We use -j to junk paths so the zip contains files directly
    # and we zip *.onnx and *.pkl
    zip -r dist/models-$VERSION.zip "$MODELS_SRC"
else
    echo "Error: Models source directory '$MODELS_SRC' not found"
    exit 1
fi

echo "Assets packaged in dist/"
ls -lh dist/
