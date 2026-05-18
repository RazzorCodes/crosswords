{
  description = "Crossword Puzzle Web App";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = pkgs.buildNpmPackage {
          pname = "crosswords";
          version = "0.1.0";
          src = ./.;
          npmDepsHash = "sha256-YwV6T7f0CzlnDLSd3A4E9nDlYk/Dxm0xwAf5xSCu0U8=";
          installPhase = ''
            mkdir -p $out
            cp -r dist/* $out/
          '';
        };

        packages.node_modules = pkgs.buildNpmPackage {
          pname = "crosswords-node-modules";
          version = "0.1.0";
          src = ./.;
          npmDepsHash = "sha256-YwV6T7f0CzlnDLSd3A4E9nDlYk/Dxm0xwAf5xSCu0U8=";
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r node_modules $out/
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs_20
          ];
          shellHook = ''
            if [ ! -L node_modules ] || [ ! -d node_modules ]; then
              echo "Symlinking node_modules from Nix store..."
              ln -sfn $(nix build .#node_modules --no-link --print-out-paths)/node_modules node_modules
            fi
            export PATH="$PWD/node_modules/.bin:$PATH"
            echo "Crossword development environment loaded."
            echo "node_modules are managed by Nix. Vite cache is redirected to .vite_cache."
          '';
        };
      });
}
