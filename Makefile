IMAGE_NAME = crosswords-ui
PORT = 8080
WASM_OUT = public-service/wasm

.PHONY: build run stop restart clean wasm-builder-image wasm

# Creates a persistent builder with wasm-bindgen to save time
wasm-builder-image:
	@if ! docker image inspect crosswords-wasm-builder >/dev/null 2>&1; then \
		echo "Creating WASM builder image..."; \
		printf "FROM rust:1-slim\nRUN apt-get update && apt-get install -y lld pkg-config libssl-dev && rustup target add wasm32-unknown-unknown && cargo install wasm-bindgen-cli --version 0.2.121" | docker build -t crosswords-wasm-builder -; \
	fi

# Compiles WASM inside a container using your host's 23GB target folder as a cache
wasm: wasm-builder-image
	mkdir -p src/core/ui/$(WASM_OUT)/cnn
	mkdir -p src/core/ui/public-service/models
	docker run --rm \
		-v $(PWD)/src/core:/repo/src/core:z \
		-v $(PWD)/src/core/target:/repo/src/core/target:z \
		-v $(PWD)/src/core/ui/$(WASM_OUT):/out:z \
		-w /repo/src/core \
		crosswords-wasm-builder \
		/bin/bash -c " \
			CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER=wasm-ld cargo rustc --target wasm32-unknown-unknown --release --lib --crate-type=cdylib --no-default-features && \
			cp target/wasm32-unknown-unknown/release/handwriting_core.wasm /out/handwriting_core.wasm && \
			CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER=wasm-ld cargo rustc --target wasm32-unknown-unknown --release --lib --crate-type=cdylib --no-default-features --features wasm && \
			wasm-bindgen target/wasm32-unknown-unknown/release/handwriting_core.wasm --target web --out-dir /out/cnn --out-name handwriting_core_cnn"
	# Copy artifacts for the UI to pick up
	cp src/core/ui/$(WASM_OUT)/cnn/handwriting_core_cnn.js src/core/ui/public-service/models/
	cp src/core/ui/$(WASM_OUT)/cnn/handwriting_core_cnn_bg.wasm src/core/ui/public-service/models/
	cp models/cnn_model.json src/core/ui/public-service/models/
	cp models/cnn_model.safetensors src/core/ui/public-service/models/

build: wasm
	docker build -t $(IMAGE_NAME) -f src/core/ui/Containerfile src/core/ui

run:
	docker run -d -p $(PORT):80 --name $(IMAGE_NAME) $(IMAGE_NAME)

stop:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

restart: stop run

clean: stop
	docker rmi $(IMAGE_NAME) || true
	rm -rf src/core/ui/$(WASM_OUT)
