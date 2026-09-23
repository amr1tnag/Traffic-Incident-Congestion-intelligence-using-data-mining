#!/usr/bin/env python3
"""End-to-end driver for the Traffic Incident & Congestion Intelligence project.

    python run_pipeline.py                 # full run
    python run_pipeline.py --skip-generate # reuse existing data/raw
    python run_pipeline.py --stages preprocess etl olap mine report
    python run_pipeline.py --stages export     # refresh the Power BI extract only

Stages: generate -> preprocess -> etl -> olap -> mine -> visualize -> report -> export
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src import etl, export_bi, generate_data, olap, preprocess, report, visualize
from src.config import load_config
from src.mining import anomaly, association, classification, clustering, forecasting

STAGES = ["generate", "preprocess", "etl", "olap", "mine", "visualize", "report",
          "export"]


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse the CSVs already in data/raw")
    args = ap.parse_args()

    stages = [s for s in args.stages if not (args.skip_generate and s == "generate")]
    cfg = load_config(args.config)
    proc = Path(cfg["data"]["processed_dir"])
    reports_dir = Path(cfg["reports"]["dir"])
    started = time.time()

    if "generate" in stages:
        _banner("STAGE 1/8  Generate raw data sources")
        generate_data.main(args.config)

    if "preprocess" in stages:
        _banner("STAGE 2/8  Clean, integrate and transform")
        preprocess.main(args.config)

    if "etl" in stages:
        _banner("STAGE 3/8  Load the star-schema warehouse")
        etl.main(args.config)

    olap_results: dict = {}
    if "olap" in stages:
        _banner("STAGE 4/8  OLAP cube operations")
        olap_results = olap.run_all(args.config)

    mining: dict = {}
    base = incident_features = None
    if {"mine", "visualize", "report"} & set(stages):
        base = pd.read_csv(proc / "analytical_base.csv")
        incident_features = pd.read_csv(proc / "incident_features.csv")

    if "mine" in stages:
        _banner("STAGE 5/8  Data mining")
        print("\n-- Classification: incident severity --")
        mining["classification"] = classification.run(incident_features, cfg)
        classification.save(mining["classification"], reports_dir)

        print("\n-- Clustering: congestion hotspots --")
        mining["clustering"] = clustering.run(base, cfg)
        clustering.save(mining["clustering"], reports_dir)

        print("\n-- Association rules: Apriori --")
        mining["association"] = association.run(incident_features, cfg)
        association.save(mining["association"], reports_dir)

        print("\n-- Forecasting: hourly congestion --")
        mining["forecasting"] = forecasting.run(base, cfg)
        forecasting.save(mining["forecasting"], reports_dir)

        print("\n-- Anomaly detection --")
        mining["anomaly"] = anomaly.run(base, cfg)
        anomaly.save(mining["anomaly"], reports_dir)

    if "visualize" in stages and mining:
        _banner("STAGE 6/8  Figures")
        visualize.generate_all(base, incident_features, mining,
                               Path(cfg["reports"]["figures_dir"]))

    if "report" in stages and mining and olap_results:
        _banner("STAGE 7/8  Findings report")
        report.build(cfg, base, incident_features, olap_results, mining)

    if "export" in stages:
        _banner("STAGE 8/8  Power BI extract")
        export_bi.main(args.config)

    print(f"\nPipeline finished in {time.time() - started:.1f}s. "
          f"Outputs in {reports_dir.relative_to(Path.cwd()) if reports_dir.is_relative_to(Path.cwd()) else reports_dir}/")


if __name__ == "__main__":
    main()
