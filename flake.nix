{
  description = "Lumen Fingerprints — Rust scanner + mdBook study/explorer, one nix build";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # `nix build` -> ./result : the static site (study + explorer) with whatever
      # explorer-data.json is currently committed. Deploy ./result to GitHub Pages.
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

      # `nix develop` -> mdbook + python for the site, cargo/rustc for the scanner.
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.mdbook pkgs.python3 pkgs.cargo pkgs.rustc pkgs.zstd ];
        };
      });

      apps = forAll (pkgs: {
        # View the site locally with the data currently committed.
        serve = {
          type = "app";
          program = "${pkgs.writeShellScript "serve" ''
            set -e
            ${pkgs.mdbook}/bin/mdbook build "$PWD/docs" --dest-dir "$PWD/docs/book"
            echo "Serving http://localhost:8000  (Ctrl-C to stop)"
            ${pkgs.python3}/bin/python3 -m http.server 8000 --directory "$PWD/docs/book"
          ''}";
        };

        # Regenerate explorer-data.json from an existing report.json — no scan.
        emit = {
          type = "app";
          program = "${pkgs.writeShellScript "emit" ''
            set -e
            REPORT="''${1:?usage: nix run .#emit -- <path/to/report.json>}"
            ${pkgs.python3}/bin/python3 "$PWD/contrib/dashboard-data.py" "$REPORT" "$PWD/docs/src/explorer-data.json"
            echo "explorer-data.json updated — run 'nix run .#serve' to view."
          ''}";
        };

        # One-shot CLI scan: scan -> report -> data -> build -> serve, in one command.
        # The native scan uses your own Rust toolchain (rustup); the datadir defaults.
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

        # The interactive dashboard: builds the scanner + book, then starts the control
        # server. The server queries the live chain tip (lumen tip) and serves the site
        # with a working ▶ Run button that scans the full [floor, tip] range and fills the
        # Explorer in live. Localhost only.
        dashboard = {
          type = "app";
          program = "${pkgs.writeShellScript "dashboard" ''
            set -e
            # Datadir is optional: default to a hidden per-user dir so the user needs to
            # know nothing about Floresta — just `nix run .#dashboard`, then click Run.
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
