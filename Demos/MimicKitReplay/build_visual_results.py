#!/usr/bin/env python3
"""Build Evih MimicKit visual result packages without promoting false mesh parity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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
build_dynamic_comparison_mp4 = MODULE.build_dynamic_comparison_mp4
build_dynamic_sequence_report = MODULE.build_dynamic_sequence_report
build_ground_metric_report = MODULE.build_ground_metric_report
build_rgb_metric_report = MODULE.build_rgb_metric_report
build_visual_metric_report = MODULE.build_visual_metric_report
default_thresholds = MODULE.default_thresholds
ensure_visual_review_template = MODULE.ensure_visual_review_template
load_replay_rows = MODULE.load_replay_rows
locate_reference_frames = MODULE.locate_reference_frames
read_json = MODULE.read_json
render_skeleton_or_geom = MODULE.render_skeleton_or_geom
resolve_package_files = MODULE.resolve_package_files
sha256_file = MODULE.sha256_file
thresholds_from_args = MODULE.thresholds_from_args
validate_package_files = MODULE.validate_package_files
validate_framework_renderer_provenance = MODULE.validate_framework_renderer_provenance
validate_visual_review = MODULE.validate_visual_review
write_comparison_markdown = MODULE.write_comparison_markdown
write_comparison_sheet = MODULE.write_comparison_sheet
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
    parser.add_argument("--scene-contract", type=Path, help="Explicit strict scene_contract_v3.json for true-mesh replay.")
    parser.add_argument("--visual-review", type=Path, help="Required human visual_review.json for final true-mesh pass.")
    parser.add_argument("--review-only", action="store_true", help="Validate a signed review against existing evidence without rerendering.")
    parser.add_argument("--framework-api", action="store_true", help="Require EvihAnimation Actor + RenderPipeline capture.")
    parser.add_argument(
        "--framework-results-only",
        action="store_true",
        help="Consume an existing Framework capture and rebuild metrics, comparison, evidence, and manifest.",
    )
    parser.add_argument(
        "--framework-python",
        type=Path,
        default=Path("/root/miniconda3/envs/evihanimation-mimickit-bridge/bin/python"),
    )
    parser.add_argument("--mean-silhouette-iou-min", type=float, default=default_thresholds()["mean_silhouette_iou_min"])
    parser.add_argument("--p10-silhouette-iou-min", type=float, default=default_thresholds()["p10_silhouette_iou_min"])
    parser.add_argument("--mean-centroid-error-max", type=float, default=default_thresholds()["mean_centroid_error_max"])
    parser.add_argument("--p95-centroid-error-max", type=float, default=default_thresholds()["p95_centroid_error_max"])
    parser.add_argument("--bbox-area-ratio-p10-min", type=float, default=default_thresholds()["bbox_area_ratio_p10_min"])
    parser.add_argument("--bbox-area-ratio-p90-max", type=float, default=default_thresholds()["bbox_area_ratio_p90_max"])
    return parser


def frame_map(directory: Path) -> dict[int, Path]:
    return {
        int(path.stem.split("_")[-1]): path
        for path in directory.glob("frame_*.png")
    }


def frame_set_sha256(frames: dict[int, Path]) -> str:
    return MODULE._frame_set_sha256(frames)


def first_blocker(checks: Sequence[tuple[bool, str]]) -> str:
    return MODULE.first_blocker(checks)


def promote_existing_review(args: argparse.Namespace, out_dir: Path) -> int:
    manifest_path = out_dir / "visual_result_manifest.json"
    aggregate = read_json(manifest_path)
    mesh_report = aggregate.get("mesh_report", {}) if isinstance(aggregate.get("mesh_report"), dict) else {}
    expected_evidence = mesh_report.get("visual_review_evidence", {}) if isinstance(mesh_report.get("visual_review_evidence"), dict) else {}
    review_path = (
        args.visual_review.resolve()
        if args.visual_review
        else out_dir / "evih_mesh_replay" / "visual_review.json"
    )
    review = validate_visual_review(review_path, expected_evidence)
    write_json(out_dir / "evih_framework_replay" / "framework_visual_review_report.json", review)
    mesh_report["visual_review_pass"] = bool(review.get("visual_review_pass"))
    aggregate["mesh_report"] = mesh_report
    aggregate["visual_review_pass"] = bool(review.get("visual_review_pass"))
    blocker = first_blocker(
        [
            (bool(aggregate.get("software_geometry_replay_pass")), "software_geometry_replay_failed"),
            (bool(aggregate.get("framework_api_used")), "evih_framework_api_not_used"),
            (bool(aggregate.get("evih_framework_api_replay_pass")), "evih_framework_api_replay_failed"),
            (bool(aggregate.get("framework_scene_contract_compare_pass")), "framework_scene_contract_compare_failed"),
            (bool(aggregate.get("framework_visual_metric_pass")), "framework_visual_metric_failed"),
            (bool(aggregate.get("framework_scene_visual_metric_pass")), "framework_ground_visual_metric_failed"),
            (bool(aggregate.get("framework_dynamic_sequence_pass")), "framework_dynamic_sequence_failed"),
            (bool(aggregate.get("framework_media_ok")), "framework_media_incomplete"),
            (bool(aggregate.get("framework_renderer_provenance_valid")), "framework_renderer_provenance_invalid"),
            (bool(aggregate.get("visual_review_pass")), str(review.get("blocker") or "visual_review_missing_or_failed")),
        ]
    )
    aggregate["evih_mesh_replay_pass"] = not blocker
    aggregate["blocker"] = blocker
    write_json(manifest_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0 if aggregate["evih_mesh_replay_pass"] else 4


def build_framework_result(
    *,
    args: argparse.Namespace,
    package_dir: Path,
    out_dir: Path,
    mesh_report: JSON,
    run_capture: bool = True,
) -> JSON:
    framework_dir = out_dir / "evih_framework_replay"
    framework_dir.mkdir(parents=True, exist_ok=True)
    report_path = framework_dir / "framework_render_report.json"
    script = REPO_ROOT / "Demos" / "MimicKitReplay" / "framework_capture.py"
    if run_capture and not args.framework_python.is_file():
        return {
            "framework_api_used": False,
            "evih_framework_api_replay_pass": False,
            "blocker": "evih_framework_environment_unavailable",
            "python": str(args.framework_python),
        }
    command = []
    process = None
    if run_capture:
        command = [
            str(args.framework_python),
            str(script),
            "--package-dir",
            str(package_dir),
            "--mesh-asset",
            str(args.mesh_asset.resolve()),
            "--scene-contract",
            str(args.scene_contract.resolve()),
            "--out-dir",
            str(framework_dir),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
        ]
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    framework = read_json(report_path) if report_path.is_file() else {}
    if process is not None:
        framework["command"] = command
        framework["returncode"] = int(process.returncode)
        framework["stdout_tail"] = process.stdout[-4000:]
        framework["stderr_tail"] = process.stderr[-4000:]
    if not framework:
        framework = {
            "framework_api_used": False,
            "evih_framework_api_replay_pass": False,
            "blocker": "evih_framework_capture_failed" if run_capture else "framework_capture_report_missing",
            "command": command,
            "returncode": int(process.returncode) if process is not None else -1,
            "stdout_tail": process.stdout[-4000:] if process is not None else "",
            "stderr_tail": process.stderr[-4000:] if process is not None else "",
        }
        write_json(report_path, framework)
        return framework

    manifest_path = args.mesh_reference_manifest.resolve()
    manifest = read_json(manifest_path)
    frame_ids = list(mesh_report.get("render", {}).get("requested_frame_ids", []))
    source_rgb, source_silhouettes, source_ground_masks = locate_reference_frames(manifest, manifest_path, frame_ids)
    framework_rgb = frame_map(framework_dir / "rgb_frames")
    framework_silhouettes = frame_map(framework_dir / "silhouettes")
    framework_ground_masks = frame_map(framework_dir / "ground_masks")
    thresholds = thresholds_from_args(args)
    visual_metric = build_visual_metric_report(source_silhouettes, framework_silhouettes, thresholds=thresholds)
    ground_metric = build_ground_metric_report(source_ground_masks, framework_ground_masks)
    rgb_metric = build_rgb_metric_report(source_rgb, framework_rgb)
    dynamic = build_dynamic_sequence_report(
        frame_ids=frame_ids,
        source_rgb=source_rgb,
        source_silhouettes=source_silhouettes,
        evih_rgb=framework_rgb,
        evih_silhouettes=framework_silhouettes,
        media=framework.get("media", {}),
    )
    pairs = [(source_rgb[frame], framework_rgb[frame]) for frame in frame_ids if frame in source_rgb and frame in framework_rgb]
    sheet = write_comparison_sheet(pairs, framework_dir / "mimickit_mesh_vs_evih_framework_sheet.png")
    comparison = build_dynamic_comparison_mp4(pairs, framework_dir, fps=args.fps)
    write_json(framework_dir / "framework_visual_metric_report.json", visual_metric)
    write_json(framework_dir / "framework_scene_visual_metric_report.json", ground_metric)
    write_json(framework_dir / "framework_rgb_metric_report.json", rgb_metric)
    write_json(framework_dir / "framework_dynamic_sequence_report.json", dynamic)
    write_json(framework_dir / "framework_comparison_media_report.json", comparison)
    framework.update(
        framework_visual_metric_pass=bool(visual_metric.get("visual_metric_pass")),
        framework_scene_visual_metric_pass=bool(ground_metric.get("scene_visual_metric_pass")),
        framework_dynamic_sequence_pass=bool(dynamic.get("dynamic_sequence_pass")),
        framework_comparison_mp4_pass=bool(comparison.get("ok")),
        framework_comparison_sheet_pass=bool(sheet.get("ok")),
        reports={
            "framework_render": str(report_path),
            "framework_renderer_provenance": str(framework_dir / "framework_renderer_provenance.json"),
            "framework_visual_metric": str(framework_dir / "framework_visual_metric_report.json"),
            "framework_scene_visual_metric": str(framework_dir / "framework_scene_visual_metric_report.json"),
            "framework_rgb_metric": str(framework_dir / "framework_rgb_metric_report.json"),
            "framework_dynamic_sequence": str(framework_dir / "framework_dynamic_sequence_report.json"),
            "framework_comparison_sheet": str(framework_dir / "mimickit_mesh_vs_evih_framework_sheet.png"),
            "framework_comparison_mp4": str(framework_dir / "mimickit_vs_evih_dynamic.mp4"),
        },
        source_rgb=source_rgb,
        framework_rgb=framework_rgb,
    )
    provenance_validation = validate_framework_renderer_provenance(
        framework.get("framework_renderer_provenance", {}),
        expected_paths={
            "framework_capture_script": REPO_ROOT / "Demos" / "MimicKitReplay" / "framework_capture.py",
            "ai4animation_core_module": REPO_ROOT / "ai4animation" / "AI4Animation.py",
            "entity_module": REPO_ROOT / "ai4animation" / "Entity.py",
            "actor_module": REPO_ROOT / "ai4animation" / "Components" / "Actor.py",
            "rigid_node_mesh_module": REPO_ROOT / "ai4animation" / "Standalone" / "RigidNodeMesh.py",
            "standalone_module": REPO_ROOT / "ai4animation" / "Standalone" / "Standalone.py",
            "render_pipeline_module": REPO_ROOT / "ai4animation" / "Standalone" / "RenderPipeline.py",
            "replay_module": REPO_ROOT / "ai4animation" / "Standalone" / "MimicKitSkeletonReplay.py",
            "basic_vertex_shader": REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "basic.vs",
            "grid_shader": REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "grid.fs",
            "mesh_asset": args.mesh_asset.resolve(),
            "pose_dof_replay": package_dir / "visual_replay" / "pose_dof_replay.jsonl",
            "body_world_replay": package_dir / "visual_replay" / "body_world_replay.jsonl",
            "mesh_binding_contract": package_dir / "mesh_binding_contract.json",
            "scene_contract": args.scene_contract.resolve(),
            "rigid_node_transform_report": framework_dir / "framework_rigid_node_transform_report.json",
        },
    )
    framework["framework_renderer_provenance_validation"] = provenance_validation
    framework["framework_renderer_provenance_valid"] = bool(
        provenance_validation.get("framework_renderer_provenance_valid")
    )
    blocker = first_blocker(
        [
            (bool(framework.get("evih_framework_api_replay_pass")), str(framework.get("blocker") or "evih_framework_api_replay_failed")),
            (bool(visual_metric.get("visual_metric_pass")), str(visual_metric.get("blocker") or "framework_silhouette_visual_metric_failed")),
            (bool(ground_metric.get("scene_visual_metric_pass")), str(ground_metric.get("blocker") or "framework_ground_visual_metric_failed")),
            (bool(dynamic.get("dynamic_sequence_pass")), str(dynamic.get("blocker") or "framework_dynamic_sequence_failed")),
            (bool(sheet.get("ok")), str(sheet.get("blocker") or "framework_comparison_sheet_failed")),
            (bool(comparison.get("ok")), str(comparison.get("blocker") or "framework_comparison_mp4_failed")),
            (
                bool(framework.get("framework_renderer_provenance_valid")),
                str(provenance_validation.get("blocker") or "framework_renderer_provenance_invalid"),
            ),
        ]
    )
    framework["evih_framework_api_replay_pass"] = not blocker
    framework["blocker"] = blocker
    write_json(report_path, {key: value for key, value in framework.items() if key not in {"source_rgb", "framework_rgb"}})
    return framework


def apply_framework_result_to_aggregate(
    *,
    args: argparse.Namespace,
    package_dir: Path,
    out_dir: Path,
    aggregate: JSON,
    mesh_report: JSON,
    run_capture: bool,
) -> JSON:
    framework = build_framework_result(
        args=args,
        package_dir=package_dir,
        out_dir=out_dir,
        mesh_report=mesh_report,
        run_capture=run_capture,
    )
    framework_rgb = framework.pop("framework_rgb", {})
    framework.pop("source_rgb", {})
    aggregate["framework_report"] = framework
    aggregate["framework_api_used"] = bool(framework.get("framework_api_used"))
    aggregate["evih_framework_api_replay_pass"] = bool(framework.get("evih_framework_api_replay_pass"))
    aggregate["framework_scene_contract_compare_pass"] = bool(framework.get("framework_scene_contract_compare_pass"))
    aggregate["framework_visual_metric_pass"] = bool(framework.get("framework_visual_metric_pass"))
    aggregate["framework_scene_visual_metric_pass"] = bool(framework.get("framework_scene_visual_metric_pass"))
    aggregate["framework_dynamic_sequence_pass"] = bool(framework.get("framework_dynamic_sequence_pass"))
    aggregate["framework_media_ok"] = bool(framework.get("framework_media_ok"))
    aggregate["framework_renderer_provenance"] = framework.get("framework_renderer_provenance", {})
    aggregate["framework_renderer_provenance_validation"] = framework.get(
        "framework_renderer_provenance_validation", {}
    )
    aggregate["framework_renderer_provenance_valid"] = bool(
        framework.get("framework_renderer_provenance_valid")
    )
    aggregate["framework_comparison_sheet"] = framework.get("reports", {}).get("framework_comparison_sheet", "")
    aggregate["framework_comparison_mp4"] = framework.get("reports", {}).get("framework_comparison_mp4", "")

    review_path = (
        args.visual_review.resolve()
        if args.visual_review
        else out_dir / "evih_mesh_replay" / "visual_review.json"
    )
    combined_evidence = dict(mesh_report.get("visual_review_evidence", {}))
    combined_evidence.update(
        {
            "schema_version": 2,
            "framework_render_report_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_render_report.json"
            ),
            "framework_renderer_provenance_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_renderer_provenance.json"
            ),
            "framework_comparison_sheet_sha256": sha256_file(
                Path(str(aggregate["framework_comparison_sheet"]))
            ),
            "framework_comparison_mp4_sha256": sha256_file(
                Path(str(aggregate["framework_comparison_mp4"]))
            ),
            "framework_rgb_frame_set_sha256": frame_set_sha256(framework_rgb),
            "framework_silhouette_frame_set_sha256": frame_set_sha256(
                frame_map(out_dir / "evih_framework_replay" / "silhouettes")
            ),
            "framework_ground_mask_frame_set_sha256": frame_set_sha256(
                frame_map(out_dir / "evih_framework_replay" / "ground_masks")
            ),
            "framework_scene_contract_compare_report_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_scene_contract_compare_report.json"
            ),
            "framework_applied_scene_contract_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_applied_scene_contract.json"
            ),
            "framework_visual_metric_report_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_visual_metric_report.json"
            ),
            "framework_ground_metric_report_sha256": sha256_file(
                out_dir / "evih_framework_replay" / "framework_scene_visual_metric_report.json"
            ),
        }
    )
    ensure_visual_review_template(review_path, combined_evidence)
    visual_review = validate_visual_review(review_path, combined_evidence)
    write_json(out_dir / "evih_framework_replay" / "framework_visual_review_report.json", visual_review)
    mesh_report["visual_review_evidence"] = combined_evidence
    mesh_report["visual_review_pass"] = bool(visual_review.get("visual_review_pass"))
    aggregate["visual_review_pass"] = bool(visual_review.get("visual_review_pass"))
    aggregate["mesh_report"] = mesh_report
    framework_blocker = first_blocker(
        [
            (bool(aggregate["software_geometry_replay_pass"]), str(mesh_report.get("software_geometry_blocker") or "software_geometry_replay_failed")),
            (bool(aggregate["framework_api_used"]), "evih_framework_api_not_used"),
            (bool(aggregate["evih_framework_api_replay_pass"]), str(framework.get("blocker") or "evih_framework_api_replay_failed")),
            (bool(aggregate["framework_scene_contract_compare_pass"]), "framework_scene_contract_compare_failed"),
            (bool(aggregate["framework_visual_metric_pass"]), "framework_visual_metric_failed"),
            (bool(aggregate["framework_scene_visual_metric_pass"]), "framework_ground_visual_metric_failed"),
            (bool(aggregate["framework_dynamic_sequence_pass"]), "framework_dynamic_sequence_failed"),
            (bool(aggregate["framework_media_ok"]), "framework_media_incomplete"),
            (bool(aggregate["framework_renderer_provenance_valid"]), "framework_renderer_provenance_invalid"),
            (bool(aggregate["visual_review_pass"]), str(visual_review.get("blocker") or "visual_review_missing_or_failed")),
        ]
    )
    aggregate["evih_mesh_replay_pass"] = not framework_blocker
    aggregate["blocker"] = framework_blocker
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_dir = args.package_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        return promote_existing_review(args, out_dir)
    if args.framework_results_only:
        aggregate = read_json(out_dir / "visual_result_manifest.json")
        mesh_report = aggregate.get("mesh_report", {}) if isinstance(aggregate.get("mesh_report"), dict) else {}
        if not aggregate or not mesh_report or not args.mesh_asset or not args.mesh_reference_manifest or not args.scene_contract:
            raise ValueError("framework-results-only requires existing visual result plus mesh asset, manifest, and scene contract")
        aggregate["software_geometry_replay_pass"] = bool(
            aggregate.get("software_geometry_replay_pass")
            or mesh_report.get("software_geometry_replay_pass")
            or all(
                bool(mesh_report.get(field))
                for field in (
                    "data_binding_ok",
                    "mesh_binding_pass",
                    "fk_compare_pass",
                    "scene_contract_compare_pass",
                    "scene_visual_metric_pass",
                    "visual_metric_pass",
                    "dynamic_sequence_pass",
                    "comparison_mp4_pass",
                    "media_ok",
                )
            )
        )
        aggregate = apply_framework_result_to_aggregate(
            args=args,
            package_dir=package_dir,
            out_dir=out_dir,
            aggregate=aggregate,
            mesh_report=mesh_report,
            run_capture=False,
        )
        write_json(out_dir / "visual_result_manifest.json", aggregate)
        print(json.dumps(aggregate, indent=2, ensure_ascii=False))
        return 0 if aggregate["evih_mesh_replay_pass"] else 4
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
        "framework_report": {},
        "software_geometry_replay_pass": False,
        "framework_api_used": False,
        "evih_framework_api_replay_pass": False,
        "framework_scene_contract_compare_pass": False,
        "framework_visual_metric_pass": False,
        "framework_scene_visual_metric_pass": False,
        "framework_dynamic_sequence_pass": False,
        "framework_media_ok": False,
        "framework_renderer_provenance": {},
        "framework_renderer_provenance_validation": {},
        "framework_renderer_provenance_valid": False,
        "evih_mesh_replay_pass": False,
        "data_binding_ok": False,
        "mesh_binding_pass": False,
        "fk_compare_pass": False,
        "scene_contract_compare_pass": False,
        "scene_visual_metric_pass": False,
        "visual_metric_pass": False,
        "dynamic_sequence_pass": False,
        "comparison_mp4_pass": False,
        "visual_review_pass": False,
        "media_ok": False,
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
            aggregate["software_geometry_replay_pass"] = bool(mesh_report.get("software_geometry_replay_pass"))
            for field in (
                "evih_mesh_replay_pass",
                "data_binding_ok",
                "mesh_binding_pass",
                "fk_compare_pass",
                "scene_contract_compare_pass",
                "scene_visual_metric_pass",
                "visual_metric_pass",
                "dynamic_sequence_pass",
                "comparison_mp4_pass",
                "visual_review_pass",
                "media_ok",
            ):
                aggregate[field] = bool(mesh_report.get(field))
            aggregate["source_scene_contract_sha256"] = mesh_report.get("source_scene_contract_sha256", "")
            aggregate["evih_scene_contract_sha256"] = mesh_report.get("evih_scene_contract_sha256", "")
            aggregate["source_hashes"] = mesh_report.get("source_hashes", {})
            aggregate["comparison_sheet"] = mesh_report.get("reports", {}).get("comparison_sheet", "")
            aggregate["comparison_mp4"] = mesh_report.get("reports", {}).get("comparison_mp4", "")
            aggregate["comparison_markdown"] = mesh_report.get("reports", {}).get("comparison_markdown", "")
            aggregate["binding_report"] = mesh_report.get("reports", {}).get("mesh_binding", "")
            aggregate["scene_compare_report"] = mesh_report.get("reports", {}).get("scene_contract_compare", "")
            aggregate["visual_metric_report"] = mesh_report.get("reports", {}).get("visual_metric", "")
            aggregate["scene_visual_metric_report"] = mesh_report.get("reports", {}).get("scene_visual_metric", "")
            aggregate["blocker"] = str(mesh_report.get("blocker", ""))
            if args.framework_api:
                aggregate = apply_framework_result_to_aggregate(
                    args=args,
                    package_dir=package_dir,
                    out_dir=out_dir,
                    aggregate=aggregate,
                    mesh_report=mesh_report,
                    run_capture=True,
                )
            else:
                aggregate["evih_mesh_replay_pass"] = False
                aggregate["blocker"] = "evih_framework_api_not_used"

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
