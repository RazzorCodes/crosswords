#!/bin/bash
set -e
VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "Packaging assets for version $VERSION..."
mkdir -p dist

# Package dataset (only regular and high_quality)
zip -r dist/dataset-$VERSION.zip data/regular data/high_quality

# Package models
zip -r dist/models-$VERSION.zip models/

echo "Assets packaged in dist/"
ls -lh dist/
