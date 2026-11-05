{
  description = "Lumen Fingerprints — Rust scanner + mdBook study/explorer, one nix build";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAll (pkgs: {
        default = pkgs.stdenv.mkDerivation {
          pname = "wallet-fingerprint-survey-site";
          version = "0.1.0";
          src = ./docs;
          nativeBuildInputs = [ pkgs.mdbook ];
          buildPhase = "mdbook build --dest-dir book";
          installPhase = "cp -r book $out && touch $out/.nojekyll";
        };
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.mdbook pkgs.python3 pkgs.cargo pkgs.rustc pkgs.zstd ];
        };
      });

      apps = forAll (pkgs: {
        serve = {
          type = "app";
          program = "${pkgs.writeShellScript "serve" ''
            set -e
            ${pkgs.mdbook}/bin/mdbook build "$PWD/docs" --dest-dir "$PWD/docs/book"
            echo "Serving http://localhost:8000  (Ctrl-C to stop)"
            ${pkgs.python3}/bin/python3 -m http.server 8000 --directory "$PWD/docs/book"
          ''}";
        };

        emit = {
          type = "app";
          program = "${pkgs.writeShellScript "emit" ''
            set -e
            REPORT="''${1:?usage: nix run .#emit -- <path/to/report.json>}"
            ${pkgs.python3}/bin/python3 "$PWD/contrib/dashboard-data.py" "$REPORT" "$PWD/docs/src/explorer-data.json"
            echo "explorer-data.json updated — run 'nix run .#serve' to view."
          ''}";
        };

        scan = {
          type = "app";
          program = "${pkgs.writeShellScript "scan" ''
            set -e
            DATADIR="''${1:-$HOME/.wallet-fingerprint-survey/floresta}"
            command -v cargo >/dev/null || { echo "error: cargo not found — install rustup (https://rustup.rs)."; exit 1; }
            mkdir -p out "$DATADIR"
            echo "==> building the scanner (cargo, release)"
            cargo build --release --bin lumen
            echo "==> scanning $DATADIR"
            ./target/release/lumen scan --datadir "$DATADIR" --out out/epochs.jsonl
            echo "==> compressing the finished epochs file (zstd)"
            ${pkgs.zstd}/bin/zstd -q -f --rm out/epochs.jsonl -o out/epochs.jsonl.zst
            echo "==> aggregating the report (report.json + wallet-axis-cross.csv)"
            ./target/release/lumen report --epochs out/epochs.jsonl.zst --out-dir out/report
            echo "==> emitting explorer-data.json + the ML wallet×axis cross CSV from your scan"
            ${pkgs.python3}/bin/python3 "$PWD/contrib/dashboard-data.py" out/report/report.json docs/src/explorer-data.json
            cp out/report/wallet-axis-cross.csv docs/src/wallet-axis-cross.csv
            echo "==> building + serving the site with YOUR data"
            ${pkgs.mdbook}/bin/mdbook build "$PWD/docs" --dest-dir "$PWD/docs/book"
            echo "Serving http://localhost:8000  (Ctrl-C to stop)"
            ${pkgs.python3}/bin/python3 -m http.server 8000 --directory "$PWD/docs/book"
          ''}";
        };

        dashboard = {
          type = "app";
          program = "${pkgs.writeShellScript "dashboard" ''
            set -e
            DATADIR="''${1:-$HOME/.wallet-fingerprint-survey/floresta}"
            command -v cargo >/dev/null || { echo "error: cargo not found — install rustup (https://rustup.rs)."; exit 1; }
            mkdir -p out "$DATADIR"
            echo "==> building the scanner (first run only takes a minute)"
            cargo build --release --bin lumen
            ${pkgs.mdbook}/bin/mdbook build "$PWD/docs" --dest-dir "$PWD/docs/book"
            export LUMEN_BIN="$PWD/target/release/lumen"
            export DASHBOARD_EMIT="$PWD/contrib/dashboard-data.py"
            echo "==> opening the dashboard at http://127.0.0.1:8000 (Ctrl-C to stop)"
            ${pkgs.python3}/bin/python3 "$PWD/contrib/dashboard-server.py" \
              --book "$PWD/docs/book" --datadir "$DATADIR" \
              --epochs "$PWD/out/epochs.jsonl" --emit "$PWD/docs/book/explorer-data.json" \
              --emit-upstream "$PWD/docs/src/explorer-data.json" \
              --cross-upstream "$PWD/docs/src/wallet-axis-cross.csv" \
              --port 8000 --open
          ''}";
        };
      });
    };
}
