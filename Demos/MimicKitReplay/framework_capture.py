#!/usr/bin/env python3
"""Capture MimicKit rigid-node replay through EvihAnimation framework APIs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_PATH = REPO_ROOT / "ai4animation" / "Standalone" / "MimicKitSkeletonReplay.py"
SPEC = importlib.util.spec_from_file_location("evih_framework_capture_replay", REPLAY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load replay implementation: {REPLAY_PATH}")
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)
LIGHT_INTENSITY_NORMALIZER = 8000.0


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def framework_environment_report() -> dict:
    required = ("raylib", "pyray", "einops", "pygltflib")
    modules = {}
    ai4animation_spec = importlib.util.find_spec("ai4animation")
    modules["ai4animation"] = {
        "available": ai4animation_spec is not None,
        "file": str(ai4animation_spec.origin if ai4animation_spec else ""),
        "check": "module_spec_before_graphics_context",
    }
    for name in required:
        try:
            module = __import__(name)
            modules[name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "")),
                "file": str(getattr(module, "__file__", "")),
            }
        except Exception as exc:
            modules[name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    display_available = bool(
        platform.system() == "Windows"
        or str(__import__("os").environ.get("DISPLAY", "")).strip()
        or str(__import__("os").environ.get("WAYLAND_DISPLAY", "")).strip()
    )
    passed = all(item["available"] for item in modules.values()) and display_available
    return {
        "schema_version": 1,
        "framework_environment_pass": passed,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "modules": modules,
        "display_available": display_available,
        "blocker": "" if passed else "evih_framework_environment_unavailable",
    }


def source_to_framework_point(values) -> tuple[float, float, float]:
    return float(values[0]), float(values[2]), -float(values[1])


def source_to_framework_direction(values) -> tuple[float, float, float]:
    converted = source_to_framework_point(values)
    length = math.sqrt(sum(value * value for value in converted)) or 1.0
    return tuple(value / length for value in converted)


def source_to_framework_matrix(row_matrix) -> np.ndarray:
    source_column = np.asarray(row_matrix, dtype=np.float64).T
    basis = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return basis @ source_column @ np.linalg.inv(basis)


def camera_sample_for_frame(contract: dict, frame_id: int) -> dict:
    for sample in contract.get("camera_samples", []):
        if isinstance(sample, dict) and int(sample.get("frame", sample.get("frame_id", -1))) == frame_id:
            return sample
    raise ValueError(f"scene contract camera sample missing for frame {frame_id}")


def apply_static_scene_contract(contract: dict) -> dict:
    from ai4animation import AI4Animation
    import pyray as pr

    pipeline = AI4Animation.Standalone.RenderPipeline
    ground = contract.get("ground", {}) if isinstance(contract.get("ground"), dict) else {}
    lights = contract.get("lights", {}) if isinstance(contract.get("lights"), dict) else {}
    distant = lights.get("distant", {}) if isinstance(lights.get("distant"), dict) else {}
    dome = lights.get("dome", {}) if isinstance(lights.get("dome"), dict) else {}

    direction = source_to_framework_direction(distant.get("direction_world", [0.35, -0.35, -1.0]))
    grid_spacing = float(ground["grid_spacing_m"])
    major_grid_spacing = float(ground["major_grid_spacing_m"])
    if grid_spacing <= 0.0 or major_grid_spacing < grid_spacing:
        raise ValueError("invalid Framework ground grid spacing contract")
    pipeline.GridSpacing = grid_spacing
    pipeline.MajorGridSpacing = major_grid_spacing
    pipeline.LightDir = pr.Vector3(*direction)
    pipeline.ShadowLight.position = pr.Vector3(*tuple(-20.0 * value for value in direction))
    pipeline.SunColor = pr.Vector3(*tuple(float(value) for value in distant.get("color_rgb", [1.0, 1.0, 1.0])))
    pipeline.SkyColor = pr.Vector3(*tuple(float(value) for value in dome.get("color_rgb", [0.7, 0.7, 0.7])))
    pipeline.SunStrength = max(0.0, float(distant.get("intensity", 0.0))) / LIGHT_INTENSITY_NORMALIZER
    pipeline.SkyStrength = max(0.0, float(dome.get("intensity", 0.0))) / LIGHT_INTENSITY_NORMALIZER
    ground_rgb = REPLAY._ground_display_color(contract)
    ground_models = [model for model in pipeline.RegisteredModels if model.name == "Ground"]
    for model in ground_models:
        model.color = pr.Color(*ground_rgb, 255)

    return {
        "ground": {
            "kind": ground.get("kind"),
            "semantic_label": ground.get("semantic_label"),
            "color_rgb": ground.get("color_rgb"),
            "display_color_rgb_8bit": list(ground_rgb),
            "grid_spacing_m": ground.get("grid_spacing_m"),
            "major_grid_spacing_m": ground.get("major_grid_spacing_m"),
            "applied_grid_spacing_m": pipeline.GridSpacing,
            "applied_major_grid_spacing_m": pipeline.MajorGridSpacing,
            "registered_model_count": len(ground_models),
        },
        "lights": {
            "distant": {
                "direction_world": distant.get("direction_world"),
                "direction_framework_y_up": list(direction),
                "intensity": distant.get("intensity"),
                "sun_strength": pipeline.SunStrength,
                "intensity_normalizer": LIGHT_INTENSITY_NORMALIZER,
                "color_rgb": distant.get("color_rgb"),
                "casts_shadows": distant.get("casts_shadows"),
            },
            "dome": {
                "intensity": dome.get("intensity"),
                "sky_strength": pipeline.SkyStrength,
                "intensity_normalizer": LIGHT_INTENSITY_NORMALIZER,
                "color_rgb": dome.get("color_rgb"),
            },
        },
        "shadow": {
            "enabled": distant.get("casts_shadows") is True,
            "map_width": int(pipeline.ShadowMap.texture.width),
            "map_height": int(pipeline.ShadowMap.texture.height),
        },
    }


def apply_camera_contract(contract: dict, frame_id: int, width: int, height: int) -> dict:
    from ai4animation import AI4Animation
    import pyray as pr
    import raylib as rl

    sample = camera_sample_for_frame(contract, frame_id)
    eye, target, vertical_fov = REPLAY.camera_for_frame(contract, frame_id, width, height)
    camera = AI4Animation.Standalone.Camera.Camera
    camera.position = pr.Vector3(*source_to_framework_point(eye))
    camera.target = pr.Vector3(*source_to_framework_point(target))
    camera.up = pr.Vector3(0.0, 1.0, 0.0)
    camera.fovy = float(vertical_fov)
    camera.projection = rl.CAMERA_PERSPECTIVE
    near = float(sample["near"])
    far = float(sample["far"])
    rl.rlSetClipPlanes(near, far)
    return {
        "frame": frame_id,
        "eye_source_z_up": list(eye),
        "target_source_z_up": list(target),
        "eye_framework_y_up": list(source_to_framework_point(eye)),
        "target_framework_y_up": list(source_to_framework_point(target)),
        "fov_degrees": float(sample["fov_degrees"]),
        "fov_axis": sample["fov_axis"],
        "vertical_fov_degrees_applied": float(vertical_fov),
        "projection": sample["projection"],
        "near": near,
        "far": far,
    }


def invert_mask(source: Path, out: Path) -> None:
    width, height, pixels = REPLAY.read_png(source)
    inverted = bytes(255 - value for value in pixels)
    REPLAY.write_png(out, width, height, inverted)


def applied_rigid_node_transforms(program, frame_id: int) -> dict:
    samples = []
    for registered, entity in program.Actor.SkinnedMesh.Registered:
        if entity.Name not in {"right_upper_arm", "sword", "right_thigh", "right_foot"}:
            continue
        transform = entity.GetTransform()
        samples.append(
            {
                "node": entity.Name,
                "entity_position": [float(value) for value in transform[:3, 3]],
                "entity_rotation": [[float(value) for value in row] for row in transform[:3, :3]],
                "registered_position": [float(value) for value in registered.position],
                "registered_rotation_axis": [float(value) for value in registered.rotationAxis],
                "registered_rotation_angle_degrees": float(registered.rotationAngle),
                "registered_scale": [float(value) for value in registered.scale],
            }
        )
    return {"frame": frame_id, "samples": samples}


class CaptureProgram:
    def __init__(self, mesh_asset: Path):
        self.mesh_asset = mesh_asset
        self.Actor = None

    def Start(self):
        from ai4animation import Actor, AI4Animation

        self.Actor = AI4Animation.Scene.AddEntity("MimicKitActor").AddComponent(
            Actor,
            str(self.mesh_asset),
            None,
            False,
        )

    def ApplyNodeTransforms(self, node_names, matrices):
        from ai4animation import AI4Animation

        transforms = np.asarray(matrices, dtype=np.float64).copy()
        scales = np.linalg.norm(transforms[:, :3, :3], axis=1)
        safe_scales = np.where(scales > 1e-12, scales, 1.0)
        transforms[:, :3, :3] /= safe_scales[:, np.newaxis, :]
        self.Actor.SetTransforms(transforms, bones=node_names)
        for node_name, scale in zip(node_names, scales):
            entity = self.Actor.NameToEntity.get(node_name)
            if entity is not None:
                AI4Animation.Scene.Scales[entity.Index] = scale
        self.Actor.SyncToScene(bones=node_names, root=False)
        self.Actor.SkinnedMesh.Update()


def capture(args: argparse.Namespace) -> dict:
    global np
    environment = framework_environment_report()
    if not environment["framework_environment_pass"]:
        return {
            "schema_version": 1,
            "framework_api_used": False,
            "evih_framework_api_replay_pass": False,
            "environment": environment,
            "blocker": environment["blocker"],
        }

    import raylib as rl

    package_dir = args.package_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rl.IsWindowReady():
        rl.InitWindow(int(args.width), int(args.height), b"AI4Animation Framework Capture")
    if not rl.IsWindowReady():
        environment["framework_environment_pass"] = False
        environment["blocker"] = "evih_framework_environment_unavailable"
        report = {
            "schema_version": 1,
            "framework_api_used": False,
            "evih_framework_api_replay_pass": False,
            "environment": environment,
            "blocker": environment["blocker"],
        }
        write_json(out_dir / "framework_render_report.json", report)
        return report

    import numpy as np
    from ai4animation import AI4Animation
    rows = REPLAY.load_replay_rows(package_dir / "visual_replay" / "pose_dof_replay.jsonl")
    body_rows = {
        int(row["frame"]): row
        for row in REPLAY.load_replay_rows(package_dir / "visual_replay" / "body_world_replay.jsonl")
    }
    for row in rows:
        body_row = body_rows.get(int(row["frame"]))
        if body_row:
            row["body_pos_m"] = body_row.get("body_pos_m", [])
            row["body_rot_xyzw"] = body_row.get("body_rot_xyzw", [])
    rows_by_frame = {int(row["frame"]): row for row in rows}
    joint_order = REPLAY.read_json(package_dir / "joint_order.json")
    source_rig = REPLAY.read_json(package_dir / "mimickit_source_rig_asset_spec.json")
    binding = REPLAY.validate_mesh_binding(
        args.mesh_asset.resolve(),
        joint_order,
        REPLAY.inspect_glb(args.mesh_asset.resolve()),
        mesh_binding_contract=REPLAY.read_json(package_dir / "mesh_binding_contract.json"),
    )
    contract = REPLAY.read_json(args.scene_contract.resolve())
    frame_ids = list(contract.get("timing", {}).get("expected_frame_ids", []))
    model = REPLAY.load_mesh_model(args.mesh_asset.resolve())
    node_names = [
        str(node.get("name") or f"node_{index}")
        for index, node in enumerate(model.document.get("nodes", []))
    ]
    initial_row = {
        "root_pos_m": [0.0, 0.0, 0.0],
        "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dof_pos": [0.0] * len(rows[0].get("dof_pos", [])),
    }

    program = CaptureProgram(args.mesh_asset.resolve())
    AI4Animation(
        program,
        mode=AI4Animation.Mode.CAPTURE,
        capture_options={
            "width": args.width,
            "height": args.height,
            "hidden": args.hidden,
        },
    )
    pipeline = AI4Animation.Standalone.RenderPipeline
    character_model_names = [
        model.name for model in pipeline.RegisteredModels if model.name.startswith("MimicKitActor/")
    ]
    ground_model_names = [model.name for model in pipeline.RegisteredModels if model.name == "Ground"]
    static_scene = apply_static_scene_contract(contract)

    rgb_dir = out_dir / "rgb_frames"
    silhouette_dir = out_dir / "silhouettes"
    ground_mask_dir = out_dir / "ground_masks"
    raw_pass_dir = out_dir / "framework_passes"
    for directory in (rgb_dir, silhouette_dir, ground_mask_dir, raw_pass_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rendered_ids = []
    applied_camera_samples = []
    rigid_node_transform_samples = {}
    max_pos_error = 0.0
    max_rot_error = 0.0
    for frame_id in frame_ids:
        row = rows_by_frame.get(int(frame_id))
        if row is None:
            continue
        node_world, _, pos_error, rot_error = REPLAY.pose_node_world_matrices(
            model,
            row,
            joint_order,
            source_rig,
            binding,
            initial_row,
        )
        max_pos_error = max(max_pos_error, pos_error)
        max_rot_error = max(max_rot_error, rot_error)
        program.ApplyNodeTransforms(
            node_names,
            [source_to_framework_matrix(matrix) for matrix in node_world],
        )
        if not rigid_node_transform_samples:
            rigid_node_transform_samples = applied_rigid_node_transforms(program, int(frame_id))
            print("FRAMEWORK_RIGID_NODE_TRANSFORMS=" + json.dumps(rigid_node_transform_samples))
        applied_camera_samples.append(
            apply_camera_contract(contract, int(frame_id), args.width, args.height)
        )

        blank = raw_pass_dir / f"blank_{frame_id:06d}.png"
        character = raw_pass_dir / f"character_{frame_id:06d}.png"
        ground = raw_pass_dir / f"ground_{frame_id:06d}.png"
        rgb = rgb_dir / f"frame_{frame_id:06d}.png"
        silhouette = silhouette_dir / f"frame_{frame_id:06d}.png"
        ground_mask = ground_mask_dir / f"frame_{frame_id:06d}.png"
        AI4Animation.Standalone.CaptureFrame(blank, model_names=[])
        AI4Animation.Standalone.CaptureFrame(character, model_names=character_model_names)
        AI4Animation.Standalone.CaptureFrame(ground, model_names=ground_model_names)
        AI4Animation.Standalone.CaptureFrame(rgb)
        AI4Animation.Standalone.CaptureSemanticFrame(
            silhouette,
            character_model_names=character_model_names,
            occluder_model_names=ground_model_names,
        )
        invert_mask(silhouette, ground_mask)
        rendered_ids.append(int(frame_id))

    fps = int(contract.get("timing", {}).get("fps", args.fps) or args.fps)
    rgb_frames = sorted(rgb_dir.glob("frame_*.png"))
    media = REPLAY.create_mp4(rgb_frames, out_dir / "framework_replay.mp4", fps)
    unique_rgb = len({REPLAY.sha256_file(path) for path in rgb_frames})
    unique_silhouette = len({REPLAY.sha256_file(path) for path in silhouette_dir.glob("frame_*.png")})
    unique_ground = len({REPLAY.sha256_file(path) for path in ground_mask_dir.glob("frame_*.png")})
    resolution = contract.get("resolution", {}) if isinstance(contract.get("resolution"), dict) else {}
    timing = contract.get("timing", {}) if isinstance(contract.get("timing"), dict) else {}
    source_contract_hash = str(contract.get("scene_contract_sha256", ""))
    computed_contract_hash = REPLAY.scene_contract_sha256(contract)
    scene_checks = {
        "source_contract_hash_valid": bool(source_contract_hash and source_contract_hash == computed_contract_hash),
        "resolution_match": int(resolution.get("width", 0)) == int(rl.GetScreenWidth()) == args.width
        and int(resolution.get("height", 0)) == int(rl.GetScreenHeight()) == args.height,
        "fps_match": int(timing.get("fps", 0)) == fps,
        "frame_ids_match": rendered_ids == frame_ids,
        "camera_samples_applied": len(applied_camera_samples) == len(frame_ids),
        "ground_contract_applied": (
            static_scene["ground"]["registered_model_count"] == 1
            and math.isclose(
                float(static_scene["ground"]["applied_grid_spacing_m"]),
                float(contract["ground"]["grid_spacing_m"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(static_scene["ground"]["applied_major_grid_spacing_m"]),
                float(contract["ground"]["major_grid_spacing_m"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        "distant_light_contract_applied": (
            bool(static_scene["lights"]["distant"]["direction_world"])
            and math.isclose(
                float(static_scene["lights"]["distant"]["sun_strength"]),
                float(contract["lights"]["distant"]["intensity"]) / LIGHT_INTENSITY_NORMALIZER,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and static_scene["lights"]["distant"]["color_rgb"] == contract["lights"]["distant"]["color_rgb"]
        ),
        "dome_light_contract_applied": (
            math.isclose(
                float(static_scene["lights"]["dome"]["sky_strength"]),
                float(contract["lights"]["dome"]["intensity"]) / LIGHT_INTENSITY_NORMALIZER,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and static_scene["lights"]["dome"]["color_rgb"] == contract["lights"]["dome"]["color_rgb"]
        ),
        "shadow_contract_applied": static_scene["shadow"]["enabled"],
    }
    framework_scene_contract_compare_pass = all(scene_checks.values())
    applied_scene_contract = {
        "schema_version": 1,
        "renderer": "evihanimation_actor_renderpipeline_rigid_node_v1",
        "source_scene_contract": str(args.scene_contract.resolve()),
        "source_scene_contract_sha256": source_contract_hash,
        "computed_source_scene_contract_sha256": computed_contract_hash,
        "resolution": {"width": args.width, "height": args.height},
        "timing": {
            "fps": fps,
            "expected_frame_ids": frame_ids,
            "rendered_frame_ids": rendered_ids,
        },
        **static_scene,
        "camera_samples": applied_camera_samples,
    }
    scene_compare_report = {
        "schema_version": 1,
        "checks": scene_checks,
        "framework_scene_contract_compare_pass": framework_scene_contract_compare_pass,
        "blocker": "" if framework_scene_contract_compare_pass else "framework_scene_contract_compare_failed",
        "applied_scene_contract": str(out_dir / "framework_applied_scene_contract.json"),
    }
    write_json(out_dir / "framework_applied_scene_contract.json", applied_scene_contract)
    write_json(out_dir / "framework_scene_contract_compare_report.json", scene_compare_report)
    write_json(out_dir / "framework_rigid_node_transform_report.json", rigid_node_transform_samples)
    dynamic_pass = min(unique_rgb, unique_silhouette) >= REPLAY.V3_MIN_UNIQUE_DYNAMIC_FRAMES
    media_ok = bool(media.get("ok") and len(rgb_frames) == len(frame_ids))
    passed = bool(
        frame_ids
        and rendered_ids == frame_ids
        and len(character_model_names) > 0
        and len(ground_model_names) == 1
        and framework_scene_contract_compare_pass
        and dynamic_pass
        and media_ok
        and binding.get("mesh_binding_pass") is True
        and max_pos_error <= 1e-6
        and max_rot_error <= 1e-5
    )
    renderer_provenance = {
        "ai4animation_mode": "CAPTURE",
        "actor_component": "ai4animation.Components.Actor.Actor",
        "mesh_component": "ai4animation.Standalone.RigidNodeMesh.RigidNodeMesh",
        "render_pipeline": "ai4animation.Standalone.RenderPipeline.RenderPipeline",
        "capture_passes": ["blank", "character_only", "ground_only", "full_scene"],
        "silhouette_derivation": "renderpipeline_semantic_character_with_ground_depth_occluder",
        "ground_mask_derivation": "complement_of_renderpipeline_semantic_character",
        "mesh_mode": "rigid_node",
        "rigid_node_update_mode": "actor_entity_world_dynamic_vertex_buffer",
        "python_executable": sys.executable,
        "framework_capture_script": str(Path(__file__).resolve()),
        "framework_capture_script_sha256": REPLAY.sha256_file(Path(__file__).resolve()),
        "ai4animation_core_module": str(REPO_ROOT / "ai4animation" / "AI4Animation.py"),
        "ai4animation_core_module_sha256": REPLAY.sha256_file(REPO_ROOT / "ai4animation" / "AI4Animation.py"),
        "entity_module": str(REPO_ROOT / "ai4animation" / "Entity.py"),
        "entity_module_sha256": REPLAY.sha256_file(REPO_ROOT / "ai4animation" / "Entity.py"),
        "actor_module": str(REPO_ROOT / "ai4animation" / "Components" / "Actor.py"),
        "actor_module_sha256": REPLAY.sha256_file(REPO_ROOT / "ai4animation" / "Components" / "Actor.py"),
        "rigid_node_mesh_module": str(REPO_ROOT / "ai4animation" / "Standalone" / "RigidNodeMesh.py"),
        "rigid_node_mesh_module_sha256": REPLAY.sha256_file(
            REPO_ROOT / "ai4animation" / "Standalone" / "RigidNodeMesh.py"
        ),
        "standalone_module": str(REPO_ROOT / "ai4animation" / "Standalone" / "Standalone.py"),
        "standalone_module_sha256": REPLAY.sha256_file(
            REPO_ROOT / "ai4animation" / "Standalone" / "Standalone.py"
        ),
        "render_pipeline_module": str(REPO_ROOT / "ai4animation" / "Standalone" / "RenderPipeline.py"),
        "render_pipeline_module_sha256": REPLAY.sha256_file(
            REPO_ROOT / "ai4animation" / "Standalone" / "RenderPipeline.py"
        ),
        "replay_module": str(REPLAY_PATH),
        "replay_module_sha256": REPLAY.sha256_file(REPLAY_PATH),
        "basic_vertex_shader": str(
            REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "basic.vs"
        ),
        "basic_vertex_shader_sha256": REPLAY.sha256_file(
            REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "basic.vs"
        ),
        "grid_shader": str(REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "grid.fs"),
        "grid_shader_sha256": REPLAY.sha256_file(
            REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "grid.fs"
        ),
        "mesh_asset": str(args.mesh_asset.resolve()),
        "mesh_asset_sha256": REPLAY.sha256_file(args.mesh_asset.resolve()),
        "pose_dof_replay": str(package_dir / "visual_replay" / "pose_dof_replay.jsonl"),
        "pose_dof_replay_sha256": REPLAY.sha256_file(package_dir / "visual_replay" / "pose_dof_replay.jsonl"),
        "body_world_replay": str(package_dir / "visual_replay" / "body_world_replay.jsonl"),
        "body_world_replay_sha256": REPLAY.sha256_file(package_dir / "visual_replay" / "body_world_replay.jsonl"),
        "mesh_binding_contract": str(package_dir / "mesh_binding_contract.json"),
        "mesh_binding_contract_sha256": REPLAY.sha256_file(package_dir / "mesh_binding_contract.json"),
        "scene_contract": str(args.scene_contract.resolve()),
        "scene_contract_sha256": REPLAY.sha256_file(args.scene_contract.resolve()),
        "rigid_node_transform_report": str(out_dir / "framework_rigid_node_transform_report.json"),
        "rigid_node_transform_report_sha256": REPLAY.sha256_file(out_dir / "framework_rigid_node_transform_report.json"),
    }
    provenance_validation = REPLAY.validate_framework_renderer_provenance(
        renderer_provenance,
        expected_paths={
            "framework_capture_script": Path(__file__).resolve(),
            "ai4animation_core_module": REPO_ROOT / "ai4animation" / "AI4Animation.py",
            "entity_module": REPO_ROOT / "ai4animation" / "Entity.py",
            "actor_module": REPO_ROOT / "ai4animation" / "Components" / "Actor.py",
            "rigid_node_mesh_module": REPO_ROOT / "ai4animation" / "Standalone" / "RigidNodeMesh.py",
            "standalone_module": REPO_ROOT / "ai4animation" / "Standalone" / "Standalone.py",
            "render_pipeline_module": REPO_ROOT / "ai4animation" / "Standalone" / "RenderPipeline.py",
            "replay_module": REPLAY_PATH,
            "basic_vertex_shader": REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "basic.vs",
            "grid_shader": REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "grid.fs",
            "mesh_asset": args.mesh_asset.resolve(),
            "pose_dof_replay": package_dir / "visual_replay" / "pose_dof_replay.jsonl",
            "body_world_replay": package_dir / "visual_replay" / "body_world_replay.jsonl",
            "mesh_binding_contract": package_dir / "mesh_binding_contract.json",
            "scene_contract": args.scene_contract.resolve(),
            "rigid_node_transform_report": out_dir / "framework_rigid_node_transform_report.json",
        },
    )
    passed = bool(passed and provenance_validation.get("framework_renderer_provenance_valid"))
    report = {
        "schema_version": 1,
        "framework_api_used": True,
        "renderer": "evihanimation_actor_renderpipeline_rigid_node_v1",
        "framework_renderer_provenance": renderer_provenance,
        "framework_renderer_provenance_validation": provenance_validation,
        "framework_renderer_provenance_valid": bool(
            provenance_validation.get("framework_renderer_provenance_valid")
        ),
        "environment": environment,
        "mesh_binding_pass": bool(binding.get("mesh_binding_pass")),
        "requested_frame_ids": frame_ids,
        "rendered_frame_ids": rendered_ids,
        "character_registered_model_count": len(character_model_names),
        "ground_registered_model_count": len(ground_model_names),
        "rgb_frames": [str(path) for path in rgb_frames],
        "rgb_png_count": len(rgb_frames),
        "silhouette_png_count": len(list(silhouette_dir.glob("frame_*.png"))),
        "ground_mask_png_count": len(list(ground_mask_dir.glob("frame_*.png"))),
        "unique_rgb_frame_count": unique_rgb,
        "unique_silhouette_frame_count": unique_silhouette,
        "unique_ground_mask_frame_count": unique_ground,
        "max_body_pos_error_m": max_pos_error,
        "max_body_rot_error_rad": max_rot_error,
        "framework_scene_contract_compare_pass": framework_scene_contract_compare_pass,
        "framework_scene_contract_compare_report": str(out_dir / "framework_scene_contract_compare_report.json"),
        "framework_applied_scene_contract": str(out_dir / "framework_applied_scene_contract.json"),
        "framework_dynamic_sequence_pass": dynamic_pass,
        "framework_media_ok": media_ok,
        "media": media,
        "evih_framework_api_replay_pass": passed,
        "blocker": "" if passed else "evih_framework_api_replay_failed",
    }
    write_json(out_dir / "framework_renderer_provenance.json", report["framework_renderer_provenance"])
    write_json(out_dir / "framework_render_report.json", report)
    rl.CloseWindow()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--mesh-asset", required=True, type=Path)
    parser.add_argument("--scene-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    if args.environment_only:
        report = framework_environment_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["framework_environment_pass"] else 5
    report = capture(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("evih_framework_api_replay_pass") else 4


if __name__ == "__main__":
    raise SystemExit(main())
