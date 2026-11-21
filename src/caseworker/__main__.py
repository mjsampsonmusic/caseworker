"""Command line: python -m caseworker {history,evaluate,site}"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings

SITE_URL = "https://mjsampsonmusic.github.io/caseworker/"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="caseworker")
    sub = parser.add_subparsers(dest="command", required=True)
    history = sub.add_parser("history", help="build the complaint-history database from CFPB archive ZIP files")
    history.add_argument("--archives", required=True, type=Path, help="folder of CCDB_Export_*.zip files")
    evaluate = sub.add_parser("evaluate", help="work held-out complaints end to end and write results/")
    evaluate.add_argument("--cases", required=True, type=Path, help="firstlook's data/splits/test.csv")
    evaluate.add_argument("--per-product", default=3, type=int)
    evaluate.add_argument("--out", default="results", type=Path)
    site = sub.add_parser("site", help="build the results page and update the README table")
    site.add_argument("--results", default="results/results.json", type=Path)
    site.add_argument("--out", default="docs", type=Path)
    site.add_argument("--readme", default="README.md", type=Path)
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "history":
        from .services import History

        Path(settings.history_db).parent.mkdir(parents=True, exist_ok=True)
        count = History(settings.history_db).load_archives(sorted(args.archives.glob("*.zip")))
        print(f"Loaded {count} complaints into {settings.history_db}")
    elif args.command == "evaluate":
        from .evaluate import evaluate as run_evaluate, pick_cases
        from .factory import build_agent

        cases = pick_cases(args.cases, args.per_product)
        results = run_evaluate(build_agent(settings), cases, args.out, settings.llm_model)
        print(json.dumps(results["summary"], indent=2))
    elif args.command == "site":
        from .evaluate import summary_markdown
        from .site import build_site, update_readme

        results = json.loads(args.results.read_text())
        page = build_site(results, args.out)
        updated = update_readme(args.readme, summary_markdown(results, SITE_URL))
        print(f"Wrote {page}; README {'updated' if updated else 'unchanged'}")


if __name__ == "__main__":
    main()
