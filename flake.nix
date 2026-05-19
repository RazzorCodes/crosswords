{
  description = "Crossword Puzzle Web App";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system; 
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            nodejs_20
            (python3.withPackages (ps: with ps; [
              numpy
              scikit-learn
              pillow
            ]))
          ];
          shellHook = ''
            export PYTHONPATH=$PYTHONPATH:$(pwd)/src/ml:$(pwd)/src/srv
            echo "Crosswords dev shell loaded (Node.js + lightweight Python)."
            echo "CNN/ONNX training remains container-based; the shell intentionally avoids torch-bin."
          '';
        };
      });
}
