from pathlib import Path
import argparse
import shutil
import sys

CURRENT_MODULE = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()

    files_to_sync = (CURRENT_MODULE / "files_to_sync.txt").read_text().splitlines()
    for file in files_to_sync:
        # The relative paths are identical in both modules.
        src = args.previous_module / file
        dst = CURRENT_MODULE / file
        if not src.exists():
            print(f"Error: Source file {src} does not exist.", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("previous_module", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
