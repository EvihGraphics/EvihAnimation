#!/usr/bin/env python3
"""Build Evih MimicKit visual result packages without promoting false mesh parity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "ai4animation" / "Standalone" / "MimicKitSkeletonReplay.py"
SPEC = importlib.util.spec_from_file_location("evih_mimickit_replay", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load replay implementation: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_true_mesh_replay = MODULE.build_true_mesh_replay
default_thresholds = MODULE.default_thresholds
load_replay_rows = MODULE.load_replay_rows
read_json = MODULE.read_json
render_skeleton_or_geom = MODULE.render_skeleton_or_geom
resolve_package_files = MODULE.resolve_package_files
thresholds_from_args = MODULE.thresholds_from_args
validate_package_files = MODULE.validate_package_files
write_json = MODULE.write_json


JSON = Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MimicKit visual replay results in EvihAnimation.")
    parser.add_argument("--package-dir", required=True, type=Path, help="MimicKit visual replay package.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Result package directory.")
    parser.add_argument("--label", default="MimicKitReplay")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--skip-skeleton", action="store_true")
    parser.add_argument("--skip-character-geom", action="store_true")
    parser.add_argument("--true-mesh", action="store_true", help="Build strict true-mesh artifacts.")
    parser.add_argument("--mesh-asset", type=Path, help="Exact MimicKit-exported GLB/GLTF.")
    parser.add_argument("--mesh-reference-manifest", type=Path, help="Passing MimicKit mesh-reference manifest.")
    parser.add_argument("--scene-contract", type=Path, help="Optional explicit scene_contract_v2.json.")
    parser.add_argument("--visual-review", type=Path, help="Required human visual_review.json for final true-mesh pass.")
    parser.add_argument("--mean-silhouette-iou-min", type=float, default=default_thresholds()["mean_silhouette_iou_min"])
    parser.add_argument("--p10-silhouette-iou-min", type=float, default=default_thresholds()["p10_silhouette_iou_min"])
    parser.add_argument("--mean-centroid-error-max", type=float, default=default_thresholds()["mean_centroid_error_max"])
    parser.add_argument("--p95-centroid-error-max", type=float, default=default_thresholds()["p95_centroid_error_max"])
    parser.add_argument("--bbox-area-ratio-p10-min", type=float, default=default_thresholds()["bbox_area_ratio_p10_min"])
    parser.add_argument("--bbox-area-ratio-p90-max", type=float, default=default_thresholds()["bbox_area_ratio_p90_max"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_dir = args.package_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    package_report = validate_package_files(package_dir)
    files = resolve_package_files(package_dir)
    aggregate: JSON = {
        "schema_version": 4,
        "label": args.label,
        "mimickit_package_dir": str(package_dir),
        "evih_output_dir": str(out_dir),
        "package_validation": package_report,
        "mesh_scope": bool(args.true_mesh or args.mesh_reference_manifest),
        "skeleton_report": {},
        "character_geom_report": {},
        "mesh_report": {},
        "evih_mesh_replay_pass": False,
        "mesh_binding_pass": False,
        "scene_contract_compare_pass": False,
        "visual_metric_pass": False,
        "visual_review_pass": False,
        "blocker": "",
    }
    if files["pose_dof_replay"].is_file() and files["joint_order"].is_file():
        rows = load_replay_rows(files["pose_dof_replay"])
        joint_order = read_json(files["joint_order"])
        if not args.skip_skeleton:
            aggregate["skeleton_report"] = render_skeleton_or_geom(
                rows,
                joint_order,
                out_dir / "evih_skeleton_replay",
                frames=args.frames,
                stride=args.stride,
                fps=args.fps,
                width=args.width,
                height=args.height,
                geom=False,
            )
        if not args.skip_character_geom:
            aggregate["character_geom_report"] = render_skeleton_or_geom(
                rows,
                joint_order,
                out_dir / "evih_character_geom_replay",
                frames=args.frames,
                stride=args.stride,
                fps=args.fps,
                width=args.width,
                height=args.height,
                geom=True,
            )
    elif not args.true_mesh:
        aggregate["blocker"] = str(package_report.get("blocker") or "replay_package_invalid")

    if args.true_mesh or args.mesh_reference_manifest:
        if not args.mesh_asset or not args.mesh_reference_manifest:
            aggregate["blocker"] = "true_mesh_requires_asset_and_manifest"
        else:
            mesh_report = build_true_mesh_replay(
                package_dir=package_dir,
                mesh_asset=args.mesh_asset.resolve(),
                mesh_reference_manifest_path=args.mesh_reference_manifest.resolve(),
                out_dir=out_dir / "evih_mesh_replay",
                scene_contract_path=args.scene_contract.resolve() if args.scene_contract else None,
                visual_review_path=args.visual_review.resolve() if args.visual_review else None,
                thresholds=thresholds_from_args(args),
            )
            aggregate["mesh_report"] = mesh_report
            for field in (
                "evih_mesh_replay_pass",
                "mesh_binding_pass",
                "scene_contract_compare_pass",
                "visual_metric_pass",
                "visual_review_pass",
            ):
                aggregate[field] = bool(mesh_report.get(field))
            aggregate["source_scene_contract_sha256"] = mesh_report.get("source_scene_contract_sha256", "")
            aggregate["evih_scene_contract_sha256"] = mesh_report.get("evih_scene_contract_sha256", "")
            aggregate["source_hashes"] = mesh_report.get("source_hashes", {})
            aggregate["comparison_sheet"] = mesh_report.get("reports", {}).get("comparison_sheet", "")
            aggregate["binding_report"] = mesh_report.get("reports", {}).get("mesh_binding", "")
            aggregate["scene_compare_report"] = mesh_report.get("reports", {}).get("scene_contract_compare", "")
            aggregate["visual_metric_report"] = mesh_report.get("reports", {}).get("visual_metric", "")
            aggregate["blocker"] = str(mesh_report.get("blocker", ""))

    write_json(out_dir / "visual_result_manifest.json", aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if args.true_mesh or args.mesh_reference_manifest:
        return 0 if aggregate["evih_mesh_replay_pass"] else 4
    non_mesh_pass = (
        (args.skip_skeleton or aggregate["skeleton_report"].get("skeleton_replay_pass"))
        and (args.skip_character_geom or aggregate["character_geom_report"].get("character_geom_replay_pass"))
    )
    return 0 if non_mesh_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
