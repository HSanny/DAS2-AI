# clean.py
import argparse
import shutil
from pathlib import Path

RAW_PATTERNS = [
    "histalmevt_fujitsu_????????-??????_to_????????-??????.csv",
    "history_fujitsu_????????-??????_to_????????-??????.csv",
]

def delete_matching_files(directory: Path, patterns, dry_run=False) -> list[Path]:
    deleted = []
    seen = set()
    for pat in patterns:
        for p in directory.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                if dry_run:
                    print(f"[DRY] would delete file: {p}")
                else:
                    try:
                        p.unlink()
                        print(f"[OK ] deleted file: {p}")
                        deleted.append(p)
                    except Exception as e:
                        print(f"[ERR] failed to delete {p}: {e}")
    return deleted

def delete_dir(path: Path, dry_run=False):
    if not path.exists():
        print(f"[SKIP] directory not found: {path}")
        return
    if dry_run:
        print(f"[DRY] would delete directory: {path}")
    else:
        try:
            shutil.rmtree(path)
            print(f"[OK ] deleted directory: {path}")
        except Exception as e:
            print(f"[ERR] failed to delete directory {path}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Clean generated files & folders.")
    ap.add_argument("--raw-dir", default="raw", help="Folder where merged raw CSVs are written.")
    # ap.add_argument("--outputs", nargs="*", default=["output_csv", "output_plots"],
    ap.add_argument("--outputs", nargs="*", default=["output_csv"],
                    help="Output folders to remove (space-separated).")
    ap.add_argument("--dry-run", action="store_true", help="Preview deletions without removing anything.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    print(f"[INFO] raw dir     : {raw_dir}")
    print(f"[INFO] output dirs : {', '.join(args.outputs)}")
    print(f"[INFO] dry-run     : {args.dry_run}")

    # 1) Delete specific merged raw files
    if not raw_dir.exists():
        print(f"[WARN] raw dir does not exist: {raw_dir}")
    else:
        delete_matching_files(raw_dir, RAW_PATTERNS, dry_run=args.dry_run)

    # 2) Delete output directories
    for out in args.outputs:
        delete_dir(Path(out).resolve(), dry_run=args.dry_run)

    print("[DONE] cleanup complete.")

if __name__ == "__main__":
    main()
