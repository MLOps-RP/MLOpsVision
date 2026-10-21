"""Write the API contract to docs/api_openapi.json.

Run from the C3-Model-Validation folder:

    python scripts/export_openapi.py            # rewrite the file
    python scripts/export_openapi.py --check    # exit 1 if the committed file is out of date

Other components of the platform (training, deployment, monitoring) integrate
against this file. A test fails when the API changes without the file being
regenerated, so the contract cannot drift silently.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from mqvg.api.app import create_app
from mqvg.api.settings import Settings

PROJECT = Path(__file__).resolve().parents[1]
TARGET = PROJECT / "docs" / "api_openapi.json"


def build_contract() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        app = create_app(Settings(artifact_root=root, report_dir=root / "reports",
                                  config_dir=PROJECT / "configs", api_key="x"))
        return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    contract = build_contract()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != contract:
            print(f"{TARGET} is out of date; run scripts/export_openapi.py")
            return 1
        print("the API contract is up to date")
        return 0
    TARGET.write_text(contract, encoding="utf-8", newline="\n")
    print(f"written {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
