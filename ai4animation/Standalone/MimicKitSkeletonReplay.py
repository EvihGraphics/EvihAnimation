#!/usr/bin/env python3
"""Offline MimicKit skeleton, geom, and strict true-mesh replay helpers.

The true-mesh path intentionally fails closed. It never promotes XML geometry,
an invalid GLB, or a failed MimicKit mesh-reference manifest to mesh parity.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple


JSON = Dict[str, Any]
Vec3 = Tuple[float, float, float]
Mat4 = Tuple[Tuple[float, float, float, float], ...]

COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
REQUIRED_VISUAL_REVIEW_CHECKS = (
    "character",
    "sword",
    "shield",
    "pose",
    "camera",
    "ground",
    "lighting",
    "frame_pairing",
    "no_obvious_penetration_or_drift",
)
V3_REQUIRED_FRAME_IDS = tuple(range(0, 300, 5))
V3_MIN_UNIQUE_DYNAMIC_FRAMES = 6


def read_json(path: Path) -> JSON:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def write_json(path: Path, data: JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(data: JSON) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scene_contract_sha256(contract: JSON) -> str:
    payload = dict(contract)
    payload.pop("scene_contract_sha256", None)
    return stable_json_sha256(payload)


def is_finite_sequence(value: Any, length: int | None = None) -> bool:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def first_blocker(checks: Sequence[tuple[bool, str]]) -> str:
    for passed, blocker in checks:
        if not passed:
            return blocker
    return ""


def _recursive_values(data: Any, key_names: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in key_names:
                values.append(value)
            values.extend(_recursive_values(value, key_names))
    elif isinstance(data, list):
        for value in data:
            values.extend(_recursive_values(value, key_names))
    return values


def _first_existing_path(values: Iterable[Any], bases: Sequence[Path]) -> Path | None:
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        probes = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
        for probe in probes:
            if probe.exists():
                return probe.resolve()
    return None


def load_replay_rows(path: Path) -> list[JSON]:
    rows: list[JSON] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"replay row {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("replay sidecar has no rows")
    return rows


def validate_replay_rows(rows: Sequence[JSON], joint_order: JSON) -> JSON:
    dof_size = int(joint_order.get("dof_size", 0) or 0)
    frames = [int(row.get("frame", -1)) for row in rows]
    finite = all(
        is_finite_sequence(row.get("root_pos_m"), 3)
        and is_finite_sequence(row.get("root_rot_xyzw"), 4)
        and is_finite_sequence(row.get("dof_pos"), dof_size)
        for row in rows
    )
    monotonic = all(current > previous for previous, current in zip(frames, frames[1:]))
    return {
        "row_count": len(rows),
        "dof_size": dof_size,
        "all_rows_finite": finite,
        "frames_monotonic": monotonic,
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "ok": bool(finite and monotonic and dof_size > 0),
    }


def resolve_package_files(package_dir: Path) -> dict[str, Path]:
    return {
        "pose_dof_replay": package_dir / "visual_replay" / "pose_dof_replay.jsonl",
        "body_world_replay": package_dir / "visual_replay" / "body_world_replay.jsonl",
        "body_world_contract": package_dir / "visual_replay" / "body_world_contract.json",
        "pose_dof_meta": package_dir / "visual_replay" / "pose_dof_meta.json",
        "joint_order": package_dir / "joint_order.json",
        "source_rig_asset_spec": package_dir / "mimickit_source_rig_asset_spec.json",
        "visual_alignment_contract": package_dir / "visual_alignment_contract.json",
    }


def validate_package_files(package_dir: Path) -> JSON:
    files = resolve_package_files(package_dir)
    required = ("pose_dof_replay", "body_world_replay", "body_world_contract", "pose_dof_meta", "joint_order", "source_rig_asset_spec", "visual_alignment_contract")
    missing = [name for name in required if not files[name].is_file()]
    data_binding = validate_body_world_replay(files) if not missing else {"data_binding_ok": False, "blocker": "body_world_replay_invalid"}
    return {
        "package_dir": str(package_dir),
        "files": {name: str(path) for name, path in files.items()},
        "hashes": {name: sha256_file(path) for name, path in files.items()},
        "missing": missing,
        "data_binding": data_binding,
        "data_binding_ok": bool(data_binding.get("data_binding_ok")),
        "ok": not missing and bool(data_binding.get("data_binding_ok")),
        "blocker": "" if not missing and data_binding.get("data_binding_ok") else ("mesh_package_sidecars_missing" if missing else "body_world_replay_invalid"),
    }


def validate_body_world_replay(files: dict[str, Path]) -> JSON:
    try:
        pose_rows = load_replay_rows(files["pose_dof_replay"])
        body_rows = load_replay_rows(files["body_world_replay"])
        joint_order = read_json(files["joint_order"])
    except Exception as exc:
        return {"data_binding_ok": False, "blocker": "body_world_replay_invalid", "error": f"{type(exc).__name__}: {exc}"}
    body_order = [str(value) for value in joint_order.get("body_order", [])]
    pose_frames = [int(row.get("frame", -1)) for row in pose_rows]
    body_frames = [int(row.get("frame", -1)) for row in body_rows]
    dimensions_ok = bool(body_order and body_rows) and all(
        row.get("body_order") == body_order
        and is_finite_sequence(row.get("body_pos_m"), len(body_order) * 3)
        and is_finite_sequence(row.get("body_rot_xyzw"), len(body_order) * 4)
        for row in body_rows
    )
    authoritative_source_ok = bool(body_rows) and all(row.get("source") == "canonical_source_rig_fk_v3" for row in body_rows)
    checks = {
        "frame_ids_match": bool(pose_frames) and pose_frames == body_frames,
        "body_order_and_dimensions_match": dimensions_ok,
        "authoritative_source_is_canonical_source_rig_fk_v3": authoritative_source_ok,
    }
    return {
        "schema_version": 1,
        "data_binding_ok": all(checks.values()),
        "checks": checks,
        "row_count": len(body_rows),
        "body_count": len(body_order),
        "blocker": "" if all(checks.values()) else "body_world_replay_invalid",
    }


def _glb_chunks(path: Path) -> tuple[JSON, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("GLB is too small")
    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total_length != len(raw):
        raise ValueError("invalid GLB header")
    document: JSON | None = None
    binary = b""
    offset = 12
    while offset + 8 <= len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            parsed = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("GLB JSON chunk is not an object")
            document = parsed
        elif chunk_type == 0x004E4942:
            binary = chunk
    if document is None:
        raise ValueError("GLB JSON chunk is missing")
    return document, binary


def load_gltf(path: Path) -> tuple[JSON, list[bytes]]:
    if path.suffix.lower() == ".glb":
        document, binary = _glb_chunks(path)
        buffers = [binary]
        for buffer_info in document.get("buffers", [])[1:]:
            uri = buffer_info.get("uri", "") if isinstance(buffer_info, dict) else ""
            buffers.append(_load_buffer_uri(path.parent, uri))
        return document, buffers
    document = read_json(path)
    buffers = [
        _load_buffer_uri(path.parent, info.get("uri", "") if isinstance(info, dict) else "")
        for info in document.get("buffers", [])
    ]
    return document, buffers


def _load_buffer_uri(base: Path, uri: str) -> bytes:
    if uri.startswith("data:") and "," in uri:
        return base64.b64decode(uri.split(",", 1)[1])
    if uri:
        return (base / uri).read_bytes()
    return b""


def read_accessor(document: JSON, buffers: Sequence[bytes], accessor_index: int) -> list[tuple[float, ...]]:
    accessors = document.get("accessors", [])
    views = document.get("bufferViews", [])
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict) or not isinstance(accessor.get("bufferView"), int):
        raise ValueError(f"accessor {accessor_index} has no bufferView")
    view = views[accessor["bufferView"]]
    if not isinstance(view, dict):
        raise ValueError(f"bufferView for accessor {accessor_index} is invalid")
    component_type = int(accessor.get("componentType", 0))
    accessor_type = str(accessor.get("type", ""))
    if component_type not in COMPONENT_FORMATS or accessor_type not in TYPE_COMPONENTS:
        raise ValueError(f"unsupported accessor format: {component_type}/{accessor_type}")
    format_code, component_size = COMPONENT_FORMATS[component_type]
    components = TYPE_COMPONENTS[accessor_type]
    count = int(accessor.get("count", 0))
    item_size = components * component_size
    stride = int(view.get("byteStride", item_size) or item_size)
    offset = int(view.get("byteOffset", 0) or 0) + int(accessor.get("byteOffset", 0) or 0)
    buffer_index = int(view.get("buffer", 0) or 0)
    data = buffers[buffer_index]
    values: list[tuple[float, ...]] = []
    for item_index in range(count):
        start = offset + item_index * stride
        unpacked = struct.unpack_from("<" + format_code * components, data, start)
        values.append(tuple(float(value) for value in unpacked))
    return values


def inspect_glb(path: Path) -> JSON:
    report: JSON = {
        "schema_version": 2,
        "asset_file": str(path),
        "asset_sha256": sha256_file(path),
        "asset_size_bytes": path.stat().st_size if path.is_file() else 0,
        "ok": False,
        "blocker": "",
    }
    if not path.is_file():
        report["blocker"] = "mesh_asset_missing"
        return report
    try:
        document, buffers = load_gltf(path)
    except Exception as exc:
        report.update(blocker="mesh_asset_parse_failed", error=f"{type(exc).__name__}: {exc}")
        return report
    nodes = document.get("nodes", []) if isinstance(document.get("nodes"), list) else []
    meshes = document.get("meshes", []) if isinstance(document.get("meshes"), list) else []
    skins = document.get("skins", []) if isinstance(document.get("skins"), list) else []
    accessors = document.get("accessors", []) if isinstance(document.get("accessors"), list) else []
    node_names = [str(node.get("name", "")) for node in nodes if isinstance(node, dict)]
    mesh_names = [str(mesh.get("name", "")) for mesh in meshes if isinstance(mesh, dict)]
    primitive_count = 0
    vertex_count = 0
    triangle_count = 0
    primitive_errors: list[str] = []
    for mesh_index, mesh in enumerate(meshes):
        primitives = mesh.get("primitives", []) if isinstance(mesh, dict) else []
        if not isinstance(primitives, list):
            continue
        primitive_count += len(primitives)
        for primitive_index, primitive in enumerate(primitives):
            try:
                attributes = primitive.get("attributes", {})
                position_index = attributes.get("POSITION") if isinstance(attributes, dict) else None
                if not isinstance(position_index, int):
                    raise ValueError("POSITION accessor missing")
                position_accessor = accessors[position_index]
                vertex_count += int(position_accessor.get("count", 0))
                if isinstance(primitive.get("indices"), int):
                    index_accessor = accessors[primitive["indices"]]
                    triangle_count += int(index_accessor.get("count", 0)) // 3
                else:
                    triangle_count += int(position_accessor.get("count", 0)) // 3
            except Exception as exc:
                primitive_errors.append(f"mesh={mesh_index} primitive={primitive_index}: {exc}")
    renderable_nodes = [
        index for index, node in enumerate(nodes)
        if isinstance(node, dict) and isinstance(node.get("mesh"), int)
    ]
    names = [normalize_name(name) for name in node_names + mesh_names]
    required_named_nodes = {
        "sword": any("sword" in name for name in names),
        "shield": any("shield" in name for name in names),
    }
    valid_buffers = bool(buffers) and all(len(buffer) > 0 for buffer in buffers)
    report.update(
        node_count=len(nodes),
        mesh_count=len(meshes),
        primitive_count=primitive_count,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        buffer_count=len(buffers),
        valid_buffers=valid_buffers,
        skin_count=len(skins),
        animation_count=len(document.get("animations", []) or []),
        renderable_node_count=len(renderable_nodes),
        renderable_node_indices=renderable_nodes,
        node_names=node_names,
        mesh_names=mesh_names,
        required_named_nodes=required_named_nodes,
        primitive_errors=primitive_errors,
        mesh_mode="skinned" if skins else ("rigid_node" if renderable_nodes else ""),
    )
    report["ok"] = bool(
        valid_buffers
        and nodes
        and meshes
        and primitive_count > 0
        and vertex_count > 0
        and triangle_count > 0
        and renderable_nodes
        and all(required_named_nodes.values())
        and not primitive_errors
    )
    report["blocker"] = "" if report["ok"] else "mesh_asset_structure_invalid"
    return report


def validate_mesh_binding(path: Path, joint_order: JSON, structure: JSON | None = None, **kwargs) -> JSON:
    structure = structure or inspect_glb(path)
    report: JSON = {
        "schema_version": 2,
        "asset_file": str(path),
        "mesh_binding_pass": False,
        "compatibility_fallback_used": False,
        "xml_geom_fallback_used": False,
        "unbound_renderable_nodes": [],
        "missing_required_bodies": [],
        "blocker": "",
    }
    if not structure.get("ok"):
        report["blocker"] = str(structure.get("blocker") or "mesh_asset_structure_invalid")
        return report
    document, _ = load_gltf(path)
    nodes = document.get("nodes", [])
    skins = document.get("skins", [])
    body_names = [str(value) for value in joint_order.get("body_order", [])]
    normalized_bodies = {normalize_name(name): name for name in body_names}

    mesh_binding_contract = kwargs.get("mesh_binding_contract", {})
    contract_nodes = {node.get("node_name", ""): node for node in mesh_binding_contract.get("nodes", [])}
    report["contract_mapping_complete"] = bool(contract_nodes)

    def match_body(name: str) -> str:
        # Use mesh_binding_contract for exact rigid node matching
        contract_node = contract_nodes.get(name)
        if contract_node:
            return str(contract_node.get("body_binding", ""))
        return ""

    mapped_bodies: set[str] = set()
    unbound: list[str] = []
    renderable_bindings: list[JSON] = []
    for node_index in structure.get("renderable_node_indices", []):
        node = nodes[node_index]
        node_name = str(node.get("name", f"node_{node_index}"))
        body = match_body(node_name)
        skin_index = node.get("skin")
        mode = ""
        skin_joint_bodies: list[str] = []
        if isinstance(skin_index, int) and 0 <= skin_index < len(skins):
            mode = "skinned"
            for joint_index in skins[skin_index].get("joints", []):
                if isinstance(joint_index, int) and 0 <= joint_index < len(nodes):
                    matched = match_body(str(nodes[joint_index].get("name", "")))
                    if matched:
                        mapped_bodies.add(matched)
                        skin_joint_bodies.append(matched)
        elif body:
            mode = "rigid_node"
            mapped_bodies.add(body)
        else:
            unbound.append(node_name)
        renderable_bindings.append(
            {
                "node_index": node_index,
                "node_name": node_name,
                "mode": mode,
                "body_name": body,
                "skin_joint_bodies": sorted(set(skin_joint_bodies)),
            }
        )
    missing = [body for body in body_names if body not in mapped_bodies]
    report.update(
        binding_mode=str(structure.get("mesh_mode", "")),
        renderable_bindings=renderable_bindings,
        mapped_bodies=sorted(mapped_bodies),
        missing_required_bodies=missing,
        unbound_renderable_nodes=unbound,
        sword_bound="sword" in {name.lower() for name in mapped_bodies},
        shield_bound="shield" in {name.lower() for name in mapped_bodies},
    )
    report["mesh_binding_pass"] = bool(
        report["contract_mapping_complete"]
        and
        not unbound
        and not missing
        and report["sword_bound"]
        and report["shield_bound"]
        and not report["compatibility_fallback_used"]
        and not report["xml_geom_fallback_used"]
    )
    report["blocker"] = "" if report["mesh_binding_pass"] else "mesh_binding_incomplete"
    return report


def validate_mesh_reference_manifest(manifest: JSON) -> JSON:
    render = manifest.get("render_summary", {})
    if not isinstance(render, dict):
        render = {}
    frame_ids = render.get("frame_ids", manifest.get("frame_ids", []))
    expected_ids = render.get("expected_frame_ids", manifest.get("expected_frame_ids", []))
    png_count = int(render.get("image_count", manifest.get("png_count", 0)) or 0)
    silhouette_count = int(render.get("silhouette_image_count", 0) or 0)
    ground_mask_count = int(render.get("ground_mask_image_count", 0) or 0)
    expected_count = int(render.get("expected_image_count", manifest.get("expected_image_count", 0)) or 0)
    scene_contract = manifest.get("scene_contract_v3", {})
    camera_samples = scene_contract.get("camera_samples", []) if isinstance(scene_contract, dict) else []
    visual_link_sync_ok = bool(camera_samples) and all(
        bool(sample.get("visual_link_sync_ok"))
        and int(sample.get("visual_link_sync_count", 0) or 0) > 0
        and float(sample.get("visual_link_sync_max_pos_error_m", math.inf)) <= 1e-5
        and float(sample.get("visual_link_sync_max_rot_error_rad", math.inf)) <= 1e-5
        for sample in camera_samples
        if isinstance(sample, dict)
    ) and len(camera_samples) == expected_count
    checks = [
        (bool(manifest.get("mesh_reference_pass")), "mimickit_mesh_reference_not_passing"),
        (bool(manifest.get("capture_ok")), "mimickit_capture_gate_failed"),
        (bool(manifest.get("media_ok")), "mimickit_media_gate_failed"),
        (bool(manifest.get("asset_ok")), "mimickit_asset_gate_failed"),
        (bool(manifest.get("package_ok")), "mimickit_package_gate_failed"),
        (str(render.get("visual_kind", "")) == "mesh", "mimickit_reference_not_mesh"),
        (bool(render.get("mesh_detected")), "mimickit_mesh_not_detected"),
        (png_count > 0 and png_count == expected_count, "mimickit_png_sequence_incomplete"),
        (silhouette_count == expected_count, "mimickit_silhouette_sequence_incomplete"),
        (ground_mask_count == expected_count, "mimickit_ground_mask_sequence_incomplete"),
        (list(frame_ids or []) == list(expected_ids or []) and bool(frame_ids), "mimickit_frame_ids_mismatch"),
        (bool(render.get("mp4_ok", manifest.get("mp4_ok"))), "mimickit_mp4_invalid"),
        (not bool(render.get("source_was_ppm_only", manifest.get("source_was_ppm_only"))), "ppm_only_output_rejected"),
        (bool(manifest.get("data_binding_ok")), "body_world_replay_invalid"),
        (bool(manifest.get("dynamic_sequence_ok")), "mimickit_dynamic_sequence_failed"),
        (list(frame_ids or []) == list(V3_REQUIRED_FRAME_IDS), "mimickit_dynamic_frame_ids_invalid"),
        (visual_link_sync_ok, "mimickit_visual_link_sync_failed"),
    ]
    blocker = first_blocker(checks)
    return {
        "ok": not blocker,
        "blocker": blocker,
        "frame_ids": list(frame_ids or []),
        "expected_frame_ids": list(expected_ids or []),
        "png_count": png_count,
        "silhouette_png_count": silhouette_count,
        "ground_mask_png_count": ground_mask_count,
        "expected_png_count": expected_count,
        "dynamic_sequence_ok": bool(manifest.get("dynamic_sequence_ok")),
        "visual_link_sync_ok": visual_link_sync_ok,
        "checks": {name: passed for passed, name in checks},
    }


def extract_scene_contract(manifest: JSON, manifest_path: Path, override: Path | None = None) -> tuple[JSON, str]:
    if override:
        contract = read_json(override)
        return contract, str(override)
    for key in ("scene_contract_v3", "scene_contract_v2", "scene_contract"):
        value = manifest.get(key)
        if isinstance(value, dict):
            return value, f"{manifest_path}#{key}"
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            if candidate.is_file():
                return read_json(candidate), str(candidate)
    values = _recursive_values(manifest, {"scene_contract_file", "scene_contract_v3_file", "scene_contract_v2_file"})
    candidate = _first_existing_path(values, (manifest_path.parent,))
    if candidate:
        return read_json(candidate), str(candidate)
    return {}, ""


def compare_scene_contract(
    contract: JSON,
    *,
    width: int,
    height: int,
    stride: int,
    fps: int,
    frame_ids: Sequence[int],
) -> JSON:
    timing = contract.get("timing", {}) if isinstance(contract.get("timing"), dict) else {}
    resolution = contract.get("resolution", {}) if isinstance(contract.get("resolution"), dict) else {}
    expected_ids = timing.get("expected_frame_ids", contract.get("expected_frame_ids", []))
    contract_width = int(resolution.get("width", contract.get("width", 0)) or 0)
    contract_height = int(resolution.get("height", contract.get("height", 0)) or 0)
    contract_stride = int(timing.get("frame_stride", contract.get("frame_stride", 0)) or 0)
    contract_fps = int(timing.get("fps", contract.get("mp4_fps", 0)) or 0)
    declared_hash = str(contract.get("scene_contract_sha256", ""))
    computed_hash = scene_contract_sha256(contract) if contract else ""
    camera_samples = contract.get("camera_samples", [])
    valid_samples = [
        sample
        for sample in camera_samples
        if isinstance(sample, dict)
        and is_finite_sequence(sample.get("eye_m", sample.get("eye")), 3)
        and is_finite_sequence(sample.get("target_m", sample.get("target")), 3)
        and sample.get("fov_degrees") is not None
        and math.isfinite(float(sample.get("fov_degrees")))
        and sample.get("fov_axis") in {"horizontal", "vertical"}
        and sample.get("projection") == "perspective"
        and math.isfinite(float(sample.get("near", 0.0)))
        and math.isfinite(float(sample.get("far", 0.0)))
        and float(sample.get("near", 0.0)) > 0.0
        and float(sample.get("far", 0.0)) > float(sample.get("near", 0.0))
        and bool(str(sample.get("renderer_version", "")).strip())
        and int(sample.get("capture_settle_updates", 0) or 0) > 0
        and sample.get("visual_link_sync_ok") is True
        and int(sample.get("visual_link_sync_count", 0) or 0) > 0
        and math.isfinite(float(sample.get("visual_link_sync_max_pos_error_m", float("inf"))))
        and math.isfinite(float(sample.get("visual_link_sync_max_rot_error_rad", float("inf"))))
    ] if isinstance(camera_samples, list) else []
    sample_ids = {int(sample.get("frame", sample.get("frame_id", -1))) for sample in valid_samples}
    camera_coverage = bool(frame_ids) and all(int(frame) in sample_ids for frame in frame_ids)
    applied_camera_samples = [
        {
            "frame": int(sample.get("frame", sample.get("frame_id", -1))),
            "eye": list(sample.get("eye_m", sample.get("eye", []))),
            "target": list(sample.get("target_m", sample.get("target", []))),
            "fov_degrees": float(sample["fov_degrees"]),
            "fov_axis": sample["fov_axis"],
            "projection": sample["projection"],
            "near": float(sample["near"]),
            "far": float(sample["far"]),
            "renderer_version": "evih_stdlib_glb_triangle_rasterizer_v3",
        }
        for sample in valid_samples
    ]
    required_scene_fields = all(
        isinstance(contract.get(name), dict)
        for name in ("resolution", "timing", "source", "coordinate_system", "camera", "ground", "lights", "capture")
    ) and bool(str(contract.get("renderer", "")).strip()) and contract.get("color_space") == "srgb"
    lights = contract.get("lights", {}) if isinstance(contract.get("lights"), dict) else {}
    distant_light = lights.get("distant", {}) if isinstance(lights.get("distant"), dict) else {}
    distant_light_strict = (
        is_finite_sequence(distant_light.get("direction_world"), 3)
        and float(distant_light["direction_world"][2]) < 0.0
        and distant_light.get("casts_shadows") is True
    )
    applied_contract = {
        "schema_version": 1,
        "resolution": {"width": width, "height": height},
        "timing": {"frame_stride": stride, "fps": fps, "expected_frame_ids": list(frame_ids)},
        "camera": contract.get("camera", {}),
        "ground": contract.get("ground", {}),
        "lights": contract.get("lights", {}),
        "capture": {
            "mode": "deterministic_offline_rasterizer",
            "settle_updates_per_attempt": [0],
            "requires_visual_link_sync": False,
        },
        "renderer": "evih_stdlib_glb_triangle_rasterizer_v3",
        "renderer_settings": {
            "ground_shadow_opacity": 0.22,
            "ground_shadow_blur_radius_px": 8,
            "ground_shadow_blur_passes": 2,
        },
        "color_space": contract.get("color_space", ""),
        "debug_overlays": False,
        "camera_samples": applied_camera_samples,
    }
    applied_hash = stable_json_sha256(applied_contract)
    checks = {
        "schema_v3_or_newer": int(contract.get("schema_version", 0) or 0) >= 3,
        "declared_hash_valid": bool(declared_hash and declared_hash == computed_hash),
        "required_scene_fields_present": required_scene_fields,
        "distant_light_strict": distant_light_strict,
        "resolution_match": width == contract_width and height == contract_height,
        "stride_match": stride == contract_stride,
        "fps_match": fps == contract_fps,
        "frame_ids_match": list(frame_ids) == list(expected_ids or []),
        "camera_samples_cover_frames": camera_coverage,
        "camera_samples_strict": len(valid_samples) == len(frame_ids),
        "applied_camera_samples_exact": len(applied_camera_samples) == len(frame_ids),
    }
    passed = all(checks.values())
    return {
        "schema_version": 2,
        "source_scene_contract_sha256": declared_hash,
        "evih_scene_contract_sha256": applied_hash,
        "computed_scene_contract_sha256": computed_hash,
        "applied_scene_contract": applied_contract,
        "scene_contract_compare_pass": passed,
        "checks": checks,
        "blocker": "" if passed else "scene_contract_compare_failed",
    }


def compare_consumed_hashes(manifest: JSON, package_report: JSON, mesh_asset: Path, contract: JSON) -> JSON:
    asset_actual = sha256_file(mesh_asset)
    asset_expected_values = _recursive_values(
        manifest,
        {"output_sha256", "asset_sha256", "glb_sha256", "mesh_asset_sha256"},
    )
    asset_expected = next((value for value in asset_expected_values if isinstance(value, str) and len(value) == 64), "")
    sidecar_expected_values = _recursive_values(
        manifest,
        {"pose_dof_replay_sha256", "sidecar_sha256", "replay_sha256"},
    )
    sidecar_expected = next((value for value in sidecar_expected_values if isinstance(value, str) and len(value) == 64), "")
    sidecar_actual = package_report.get("hashes", {}).get("pose_dof_replay", "")
    body_expected_values = _recursive_values(manifest, {"body_world_replay_sha256"})
    body_expected = next((value for value in body_expected_values if isinstance(value, str) and len(value) == 64), "")
    body_actual = package_report.get("hashes", {}).get("body_world_replay", "")
    scene_expected = str(contract.get("scene_contract_sha256", ""))
    scene_actual = scene_contract_sha256(contract) if contract else ""
    entries = {
        "pose_dof_replay": {
            "source_sha256": sidecar_expected or sidecar_actual,
            "evih_consumed_sha256": sidecar_actual,
            "declared_source_hash_present": bool(sidecar_expected),
            "match": bool(sidecar_expected and sidecar_actual == sidecar_expected),
        },
        "body_world_replay": {
            "source_sha256": body_expected,
            "evih_consumed_sha256": body_actual,
            "declared_source_hash_present": bool(body_expected),
            "match": bool(body_expected and body_actual == body_expected),
        },
        "mesh_asset": {
            "source_sha256": asset_expected,
            "evih_consumed_sha256": asset_actual,
            "declared_source_hash_present": bool(asset_expected),
            "match": bool(asset_expected and asset_actual == asset_expected),
        },
        "scene_contract": {
            "source_sha256": scene_expected,
            "evih_consumed_sha256": scene_actual,
            "computed_sha256": scene_actual,
            "declared_source_hash_present": bool(scene_expected),
            "match": bool(scene_expected and scene_expected == scene_actual),
        },
    }
    return {
        "schema_version": 2,
        "hashes": entries,
        "hash_match_pass": all(entry["match"] for entry in entries.values()),
        "blocker": "" if all(entry["match"] for entry in entries.values()) else "source_hash_mismatch",
    }


def identity_matrix() -> Mat4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

def rigid_inverse(matrix: Mat4) -> Mat4:
    # matrix is R * T (row-vector)
    # R is 3x3 top-left, T is matrix[3][0..2]
    # R_inv = R^T
    # T_inv = -T * R^T
    r00, r01, r02 = matrix[0][0], matrix[0][1], matrix[0][2]
    r10, r11, r12 = matrix[1][0], matrix[1][1], matrix[1][2]
    r20, r21, r22 = matrix[2][0], matrix[2][1], matrix[2][2]
    tx, ty, tz = matrix[3][0], matrix[3][1], matrix[3][2]
    
    # R^T
    s2 = r00*r00 + r01*r01 + r02*r02
    inv_s2 = 1.0 / s2 if s2 > 1e-12 else 1.0
    
    i00, i01, i02 = r00 * inv_s2, r10 * inv_s2, r20 * inv_s2
    i10, i11, i12 = r01 * inv_s2, r11 * inv_s2, r21 * inv_s2
    i20, i21, i22 = r02 * inv_s2, r12 * inv_s2, r22 * inv_s2
    
    # -T * R_inv
    itx = -(tx * i00 + ty * i10 + tz * i20)
    ity = -(tx * i01 + ty * i11 + tz * i21)
    itz = -(tx * i02 + ty * i12 + tz * i22)
    
    return (
        (i00, i01, i02, 0.0),
        (i10, i11, i12, 0.0),
        (i20, i21, i22, 0.0),
        (itx, ity, itz, 1.0),
    )


def mat_mul(left: Mat4, right: Mat4) -> Mat4:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4))
        for row in range(4)
    )


def transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    x, y, z = point
    values = (x, y, z, 1.0)
    result = [sum(values[row] * matrix[row][column] for row in range(4)) for column in range(4)]
    w = result[3] or 1.0
    return result[0] / w, result[1] / w, result[2] / w


def quat_matrix(quaternion: Sequence[float]) -> Mat4:
    x, y, z, w = (float(value) for value in quaternion)
    length = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / length, y / length, z / length, w / length
    # All transforms in this renderer use row vectors. This is the transpose
    # of the common column-vector quaternion matrix.
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0.0),
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0.0),
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def quat_normalize_xyzw(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(value) for value in quaternion)
    length = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / length, y / length, z / length, w / length


def quat_mul_xyzw(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = (float(value) for value in left)
    x2, y2, z2, w2 = (float(value) for value in right)
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_rotate_vec(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    x, y, z, w = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def joint_rotation_quaternion(joint: JSON, dof_pos: Sequence[float]) -> tuple[float, float, float, float]:
    start = int(joint.get("dof_index", 0))
    dimension = int(joint.get("dof_dim", 0))
    values = [float(value) for value in dof_pos[start : start + dimension]]
    if dimension == 3 and str(joint.get("joint_type", "")).lower() == "spherical":
        angle = math.sqrt(sum(value * value for value in values))
        if angle > 1e-12:
            scale = math.sin(angle * 0.5) / angle
            return quat_normalize_xyzw((values[0] * scale, values[1] * scale, values[2] * scale, math.cos(angle * 0.5)))
    elif dimension == 1:
        axis = joint.get("axis_xyz", joint.get("axis", [0.0, 1.0, 0.0]))
        if is_finite_sequence(axis, 3):
            ax, ay, az = _normalize((float(axis[0]), float(axis[1]), float(axis[2])))
            half = values[0] * 0.5
            sine = math.sin(half)
            return quat_normalize_xyzw((ax * sine, ay * sine, az * sine, math.cos(half)))
    return 0.0, 0.0, 0.0, 1.0


def trs_matrix(node: JSON) -> Mat4:
    if isinstance(node.get("matrix"), list) and len(node["matrix"]) == 16:
        values = [float(value) for value in node["matrix"]]
        return tuple(tuple(values[row * 4 + column] for column in range(4)) for row in range(4))
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    if node.get("name") == "world":
        print(f"world node scale: {scale}")
    matrix = [list(row) for row in quat_matrix(rotation)]
    for row in range(3):
        for column in range(3):
            matrix[row][column] *= float(scale[row])
    matrix[3][0] = float(translation[0])
    matrix[3][1] = float(translation[1])
    matrix[3][2] = float(translation[2])
    return tuple(tuple(row) for row in matrix)


def root_matrix(row: JSON, _initial_row: JSON) -> Mat4:
    root = row.get("root_pos_m", [0.0, 0.0, 0.0])
    matrix = [list(value) for value in quat_matrix(row.get("root_rot_xyzw", [0.0, 0.0, 0.0, 1.0]))]
    for index in range(3):
        matrix[index][3] = float(root[index])
    return tuple(tuple(value) for value in matrix)


@dataclass
class MeshTriangle:
    points: tuple[Vec3, Vec3, Vec3]
    color: tuple[int, int, int]


@dataclass
class MeshPrimitive:
    node_index: int
    skin_index: int | None
    positions: list[Vec3]
    indices: list[int]
    joint_indices: list[tuple[int, ...]]
    joint_weights: list[tuple[float, ...]]
    color: tuple[int, int, int]


@dataclass
class MeshModel:
    document: JSON
    primitives: list[MeshPrimitive]
    inverse_bind_matrices: list[list[Mat4]]
    parents: dict[int, int]


def _material_color(document: JSON, primitive: JSON) -> tuple[int, int, int]:
    material_index = primitive.get("material")
    materials = document.get("materials", [])
    if not isinstance(material_index, int) or not (0 <= material_index < len(materials)):
        return 220, 220, 220
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {}) if isinstance(material, dict) else {}
    factor = pbr.get("baseColorFactor", [0.86, 0.86, 0.86, 1.0]) if isinstance(pbr, dict) else [0.86, 0.86, 0.86, 1.0]
    return tuple(max(0, min(255, round(float(factor[index]) * 255))) for index in range(3))


def _accessor_matrix(values: Sequence[float]) -> Mat4:
    if len(values) != 16:
        return identity_matrix()
    return tuple(tuple(float(values[row * 4 + column]) for column in range(4)) for row in range(4))


def load_mesh_model(path: Path) -> MeshModel:
    document, buffers = load_gltf(path)
    nodes = document.get("nodes", [])
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        if isinstance(node, dict):
            for child in node.get("children", []):
                if isinstance(child, int):
                    parents[child] = parent_index
    inverse_bind_matrices: list[list[Mat4]] = []
    for skin in document.get("skins", []):
        accessor_index = skin.get("inverseBindMatrices") if isinstance(skin, dict) else None
        if isinstance(accessor_index, int):
            inverse_bind_matrices.append([_accessor_matrix(values) for values in read_accessor(document, buffers, accessor_index)])
        else:
            inverse_bind_matrices.append([identity_matrix() for _ in skin.get("joints", [])])

    primitive_specs: list[MeshPrimitive] = []
    meshes = document.get("meshes", [])
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("mesh"), int):
            continue
        mesh = meshes[node["mesh"]]
        for primitive in mesh.get("primitives", []):
            attributes = primitive["attributes"]
            positions = [tuple(values[:3]) for values in read_accessor(document, buffers, attributes["POSITION"])]
            indices = (
                [int(value[0]) for value in read_accessor(document, buffers, primitive["indices"])]
                if isinstance(primitive.get("indices"), int)
                else list(range(len(positions)))
            )
            joints = (
                [tuple(int(value) for value in values) for values in read_accessor(document, buffers, attributes["JOINTS_0"])]
                if isinstance(attributes.get("JOINTS_0"), int)
                else []
            )
            weights = (
                [tuple(float(value) for value in values) for values in read_accessor(document, buffers, attributes["WEIGHTS_0"])]
                if isinstance(attributes.get("WEIGHTS_0"), int)
                else []
            )
            primitive_specs.append(
                MeshPrimitive(
                    node_index=node_index,
                    skin_index=node.get("skin") if isinstance(node.get("skin"), int) else None,
                    positions=positions,
                    indices=indices,
                    joint_indices=joints,
                    joint_weights=weights,
                    color=_material_color(document, primitive),
                )
            )
    return MeshModel(document=document, primitives=primitive_specs, inverse_bind_matrices=inverse_bind_matrices, parents=parents)


def axis_angle_matrix(axis: Sequence[float], angle: float) -> Mat4:
    x, y, z = _normalize((float(axis[0]), float(axis[1]), float(axis[2])))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    return (
        (cosine + x * x * one_minus, y * x * one_minus + z * sine, z * x * one_minus - y * sine, 0.0),
        (x * y * one_minus - z * sine, cosine + y * y * one_minus, z * y * one_minus + x * sine, 0.0),
        (x * z * one_minus + y * sine, y * z * one_minus - x * sine, cosine + z * z * one_minus, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def joint_rotation_matrix(joint: JSON, dof_pos: Sequence[float]) -> Mat4:
    start = int(joint.get("dof_index", 0))
    dof_dim = int(joint.get("dof_dim", 1))
    end = start + dof_dim
    values = [float(value) for value in dof_pos[start:end]]
    R_z = identity_matrix()
    if str(joint.get("joint_type", "")).lower() == "spherical" and len(values) == 3:
        angle = math.sqrt(sum(value * value for value in values))
        if angle > 1e-12:
            R_z = axis_angle_matrix(values, angle)
    elif len(values) == 1:
        axis = joint.get("axis_xyz", joint.get("axis", [0.0, 1.0, 0.0]))
        if not is_finite_sequence(axis, 3):
            axis = [0.0, 1.0, 0.0]
        R_z = axis_angle_matrix(axis, values[0])
    
    return R_z


def pose_node_world_matrices(model: MeshModel, row: JSON, joint_order: JSON, source_rig: JSON, binding: JSON, initial_row: JSON) -> tuple[list[Mat4], int, float, float]:
    # 1. Compute Body World Transforms from source_rig + row
    flat_bodies = source_rig.get("flat_bodies", [])
    body_by_name = {body["name"]: body for body in flat_bodies}
    joints_by_body = {}
    for joint in joint_order.get("joints", []):
        body_name = joint.get("body_name", "")
        if body_name:
            if body_name not in joints_by_body:
                joints_by_body[body_name] = []
            joints_by_body[body_name].append(joint)
            
    body_name_to_node_idx = {}
    for bind in binding.get("renderable_bindings", []):
        body_name_to_node_idx[bind["body_name"]] = bind["node_index"]

    nodes = model.document.get("nodes", [])
    
    original_cache: dict[int, Mat4] = {}
    def get_original_world(node_index: int) -> Mat4:
        if node_index in original_cache:
            return original_cache[node_index]
        local = trs_matrix(nodes[node_index])
        if node_index in model.parents:
            world = mat_mul(local, get_original_world(model.parents[node_index]))
        else:
            world = local
        original_cache[node_index] = world
        return world

    body_world_cache: dict[tuple[int, str], Mat4] = {}
    body_state_caches: dict[int, dict[str, tuple[Vec3, tuple[float, float, float, float]]]] = {}
    
    def get_body_state(body_name: str, target_row: JSON) -> tuple[Vec3, tuple[float, float, float, float]]:
        state_cache = body_state_caches.setdefault(id(target_row), {})
        if body_name in state_cache:
            return state_cache[body_name]
        body = body_by_name.get(body_name)
        if not body:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
            
        parent_name = body.get("parent", "")
        if not parent_name:
            root_pos = target_row.get("root_pos_m", [0.0, 0.0, 0.0])
            root_rot = target_row.get("root_rot_xyzw", [0.0, 0.0, 0.0, 1.0])
            state = (
                (float(root_pos[0]), float(root_pos[1]), float(root_pos[2])),
                quat_normalize_xyzw(root_rot),
            )
            state_cache[body_name] = state
            return state
            
        parent_pos, parent_rot = get_body_state(parent_name, target_row)
        body_joints = joints_by_body.get(body_name, [])
        primary_joint = body_joints[0] if body_joints else {}
        local_pos = primary_joint.get(
            "bind_local_translation_m",
            body.get("bind_local_translation_m", body.get("pos", [0.0, 0.0, 0.0])),
        )
        local_rot = primary_joint.get(
            "bind_local_rotation_xyzw",
            body.get("bind_local_rotation_xyzw", [0.0, 0.0, 0.0, 1.0]),
        )
        joint_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        for joint in body_joints:
            joint_rot = quat_mul_xyzw(joint_rot, joint_rotation_quaternion(joint, target_row.get("dof_pos", [])))
        offset = quat_rotate_vec(parent_rot, local_pos)
        current_pos = (parent_pos[0] + offset[0], parent_pos[1] + offset[1], parent_pos[2] + offset[2])
        current_rot = quat_mul_xyzw(parent_rot, quat_mul_xyzw(local_rot, joint_rot))
        state = current_pos, current_rot
        state_cache[body_name] = state
        return state

    def get_body_world(body_name: str, target_row: JSON) -> Mat4:
        cache_key = (id(target_row), body_name)
        if cache_key in body_world_cache:
            return body_world_cache[cache_key]
        position, rotation = get_body_state(body_name, target_row)
        values = [list(value) for value in quat_matrix(rotation)]
        values[3][:3] = list(position)
        world = tuple(tuple(value) for value in values)
        body_world_cache[cache_key] = world
        return world

    matrices = [identity_matrix() for _ in range(len(nodes))]
    
    # Use GLB node trs_matrix for unbound nodes

    for node_index in range(len(nodes)):
        matrices[node_index] = get_original_world(node_index)

    mapped_dof_nodes = 0
    renderable_bindings = binding.get("renderable_bindings", [])
    body_names = [str(value) for value in joint_order.get("body_order", [])]
    expected_body_pos = row.get("body_pos_m", [])
    expected_body_rot = row.get("body_rot_xyzw", [])

    def body_world_from_row(body_name: str, target_row: JSON) -> Mat4 | None:
        if body_name not in body_names:
            return None
        index = body_names.index(body_name)
        body_positions = target_row.get("body_pos_m", [])
        body_rotations = target_row.get("body_rot_xyzw", [])
        if index * 3 + 2 >= len(body_positions) or index * 4 + 3 >= len(body_rotations):
            return None
        matrix = [list(value) for value in quat_matrix(body_rotations[index * 4 : index * 4 + 4])]
        matrix[3][:3] = [float(value) for value in body_positions[index * 3 : index * 3 + 3]]
        return tuple(tuple(value) for value in matrix)
    
    for bind in renderable_bindings:
        node_idx = bind["node_index"]
        body_name = bind["body_name"]
        
        computed_world = get_body_world(body_name, row)
        current_body_world = body_world_from_row(body_name, row) or computed_world
        if bind.get("mode") == "rigid_node":
            rest_body_world = body_world_from_row(body_name, initial_row) or get_body_world(body_name, initial_row)
            bind_local = mat_mul(get_original_world(node_idx), rigid_inverse(rest_body_world))
            matrices[node_idx] = mat_mul(bind_local, current_body_world)
        else:
            matrices[node_idx] = current_body_world

        if body_name in joints_by_body and any(int(j.get("dof_dim", 0) or 0) > 0 for j in joints_by_body[body_name]):
            mapped_dof_nodes += 1

    max_pos_error = 0.0
    max_rot_error = 0.0
    if expected_body_pos and expected_body_rot:
        for idx, body in enumerate(body_names):
            if idx * 3 + 2 < len(expected_body_pos) and idx * 4 + 3 < len(expected_body_rot):
                expected = (float(expected_body_pos[idx*3]), float(expected_body_pos[idx*3+1]), float(expected_body_pos[idx*3+2]))
                actual, actual_rotation = get_body_state(body, row)
                dist = math.hypot(math.hypot(expected[0] - actual[0], expected[1] - actual[1]), expected[2] - actual[2])
                max_pos_error = max(max_pos_error, dist)
                expected_rotation = quat_normalize_xyzw(expected_body_rot[idx * 4 : idx * 4 + 4])
                dot = abs(sum(actual_rotation[index] * expected_rotation[index] for index in range(4)))
                max_rot_error = max(max_rot_error, 2.0 * math.acos(max(-1.0, min(1.0, dot))))

    return matrices, mapped_dof_nodes, max_pos_error, max_rot_error


def posed_mesh_triangles(model: MeshModel, row: JSON, joint_order: JSON, source_rig: JSON, binding: JSON, initial_row: JSON) -> tuple[list[MeshTriangle], int, float, float]:
    node_world, mapped_dof_nodes, max_pos_error, max_rot_error = pose_node_world_matrices(model, row, joint_order, source_rig, binding, initial_row)
    skins = model.document.get("skins", [])
    triangles: list[MeshTriangle] = []
    for primitive in model.primitives:
        posed_positions: list[Vec3] = []
        use_skin = (
            primitive.skin_index is not None
            and 0 <= primitive.skin_index < len(skins)
            and len(primitive.joint_indices) == len(primitive.positions)
            and len(primitive.joint_weights) == len(primitive.positions)
        )
        if use_skin:
            skin = skins[primitive.skin_index]
            skin_joints = skin.get("joints", [])
            inverse_binds = model.inverse_bind_matrices[primitive.skin_index]
            for vertex_index, position in enumerate(primitive.positions):
                result = [0.0, 0.0, 0.0]
                total_weight = 0.0
                for skin_joint_index, weight in zip(primitive.joint_indices[vertex_index], primitive.joint_weights[vertex_index]):
                    if weight <= 0.0 or not (0 <= skin_joint_index < len(skin_joints)):
                        continue
                    node_index = int(skin_joints[skin_joint_index])
                    inverse_bind = inverse_binds[skin_joint_index] if skin_joint_index < len(inverse_binds) else identity_matrix()
                    transformed = transform_point(mat_mul(inverse_bind, node_world[node_index]), position)
                    for axis in range(3):
                        result[axis] += transformed[axis] * weight
                    total_weight += weight
                posed_positions.append(tuple(value / total_weight for value in result) if total_weight > 0.0 else position)
        else:
            posed_positions = [transform_point(node_world[primitive.node_index], position) for position in primitive.positions]
        for offset in range(0, len(primitive.indices) - 2, 3):
            triangles.append(
                MeshTriangle(
                    points=tuple(posed_positions[primitive.indices[offset + item]] for item in range(3)),
                    color=primitive.color,
                )
            )
    return triangles, mapped_dof_nodes, max_pos_error, max_rot_error


def _vector_sub(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(_dot(vector, vector)) or 1.0
    return vector[0] / length, vector[1] / length, vector[2] / length


def camera_for_frame(contract: JSON, frame_id: int, width: int, height: int) -> tuple[Vec3, Vec3, float]:
    samples = contract.get("camera_samples", [])
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if int(sample.get("frame", sample.get("frame_id", -1))) == frame_id:
                eye = sample.get("eye_m", sample.get("eye", []))
                target = sample.get("target_m", sample.get("target", []))
                fov = sample.get("fov_degrees")
                if not is_finite_sequence(eye, 3) or not is_finite_sequence(target, 3) or fov is None or not math.isfinite(float(fov)):
                    raise ValueError(f"invalid camera sample for frame {frame_id}")
                axis = str(sample.get("fov_axis", ""))
                if axis not in {"horizontal", "vertical"}:
                    raise ValueError(f"invalid camera fov_axis for frame {frame_id}")
                vfov = float(fov)
                if axis == "horizontal":
                    aspect = float(width) / max(float(height), 1.0)
                    vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(vfov) * 0.5) / aspect))
                return tuple(float(value) for value in eye), tuple(float(value) for value in target), vfov
    raise ValueError(f"scene contract camera sample missing for frame {frame_id}")


def _linear_to_srgb_byte(value: float) -> int:
    value = max(0.0, min(1.0, value))
    srgb = 12.92 * value if value <= 0.0031308 else 1.055 * (value ** (1.0 / 2.4)) - 0.055
    return max(0, min(255, round(srgb * 255.0)))


def _ground_display_color(contract: JSON) -> tuple[int, int, int]:
    ground = contract.get("ground", {}) if isinstance(contract.get("ground"), dict) else {}
    base = ground.get("color_rgb", [0.017, 0.0153, 0.01275])
    if not is_finite_sequence(base, 3):
        base = [0.017, 0.0153, 0.01275]
    lights = contract.get("lights", {}) if isinstance(contract.get("lights"), dict) else {}
    weighted_intensity = 0.0
    for light_name in ("distant", "dome"):
        light = lights.get(light_name, {}) if isinstance(lights.get(light_name), dict) else {}
        color = light.get("color_rgb", [1.0, 1.0, 1.0])
        color_scale = sum(float(value) for value in color) / 3.0 if is_finite_sequence(color, 3) else 1.0
        weighted_intensity += float(light.get("intensity", 0.0) or 0.0) * color_scale
    # MimicKit's Isaac ground shader raises the albedo before the two scene
    # lights are applied. The normalization keeps this deterministic offline.
    albedo_add = float(ground.get("albedo_add", 10.0) or 10.0)
    light_scale = weighted_intensity / 825.0 if weighted_intensity > 0.0 else 2.6
    return tuple(_linear_to_srgb_byte(float(value) * albedo_add * light_scale) for value in base)


def render_ground_scene(
    width: int,
    height: int,
    eye: Vec3,
    target: Vec3,
    vfov: float,
    contract: JSON,
) -> tuple[bytes, bytes]:
    forward = _normalize(_vector_sub(target, eye))
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    scale = 0.5 * height / math.tan(math.radians(vfov) * 0.5)
    mask = bytearray((0, 0, 0) * (width * height))
    pixels = bytearray((8, 11, 17) * (width * height))
    ground = contract.get("ground", {}) if isinstance(contract.get("ground"), dict) else {}
    base_color = _ground_display_color(contract)
    grid_spacing = max(0.01, float(ground.get("grid_spacing_m", 1.0) or 1.0))
    major_spacing = max(grid_spacing, float(ground.get("major_grid_spacing_m", 5.0) or 5.0))
    for y in range(height):
        vertical = (height * 0.5 - (y + 0.5)) / scale
        for x in range(width):
            horizontal = ((x + 0.5) - width * 0.5) / scale
            direction = (
                forward[0] + right[0] * horizontal + up[0] * vertical,
                forward[1] + right[1] * horizontal + up[1] * vertical,
                forward[2] + right[2] * horizontal + up[2] * vertical,
            )
            if direction[2] >= -1e-9:
                continue
            distance = -eye[2] / direction[2]
            if distance <= 0.0:
                continue
            offset = (y * width + x) * 3
            mask[offset : offset + 3] = b"\xff\xff\xff"
            hit_x = eye[0] + direction[0] * distance
            hit_y = eye[1] + direction[1] * distance
            footprint = max(0.012, min(0.12, distance / max(scale, 1.0) * 1.2))
            minor_distance = min(
                abs(hit_x - round(hit_x / grid_spacing) * grid_spacing),
                abs(hit_y - round(hit_y / grid_spacing) * grid_spacing),
            )
            major_distance = min(
                abs(hit_x - round(hit_x / major_spacing) * major_spacing),
                abs(hit_y - round(hit_y / major_spacing) * major_spacing),
            )
            minor_strength = max(0.0, 1.0 - minor_distance / footprint)
            major_strength = max(0.0, 1.0 - major_distance / (footprint * 1.5))
            blend = max(0.42 * minor_strength, 0.62 * major_strength)
            color = tuple(round(channel + (255 - channel) * blend) for channel in base_color)
            pixels[offset : offset + 3] = bytes(color)
    return bytes(mask), bytes(pixels)


def render_ground_mask(width: int, height: int, eye: Vec3, target: Vec3, vfov: float) -> bytes:
    mask, _ = render_ground_scene(width, height, eye, target, vfov, {})
    return mask


def _box_blur_mask(values: Sequence[float], width: int, height: int, radius: int) -> list[float]:
    if radius <= 0:
        return [float(value) for value in values]
    horizontal = [0.0] * (width * height)
    for y in range(height):
        row = y * width
        prefix = [0.0] * (width + 1)
        for x in range(width):
            prefix[x + 1] = prefix[x] + float(values[row + x])
        for x in range(width):
            low = max(0, x - radius)
            high = min(width - 1, x + radius)
            horizontal[row + x] = (prefix[high + 1] - prefix[low]) / (high - low + 1)
    blurred = [0.0] * (width * height)
    for x in range(width):
        prefix = [0.0] * (height + 1)
        for y in range(height):
            prefix[y + 1] = prefix[y] + horizontal[y * width + x]
        for y in range(height):
            low = max(0, y - radius)
            high = min(height - 1, y + radius)
            blurred[y * width + x] = (prefix[high + 1] - prefix[low]) / (high - low + 1)
    return blurred


def apply_ground_shadows(
    ground_rgb: bytes,
    ground_mask: bytes,
    triangles: Sequence[MeshTriangle],
    *,
    width: int,
    height: int,
    eye: Vec3,
    target: Vec3,
    fov: float,
    contract: JSON,
    opacity: float = 0.22,
    blur_radius_px: int = 8,
    blur_passes: int = 2,
) -> bytes:
    lights = contract.get("lights", {}) if isinstance(contract.get("lights"), dict) else {}
    distant = lights.get("distant", {}) if isinstance(lights.get("distant"), dict) else {}
    direction_value = distant.get("direction_world")
    if distant.get("casts_shadows") is not True or not is_finite_sequence(direction_value, 3):
        return ground_rgb
    direction = _normalize(tuple(float(value) for value in direction_value))
    if direction[2] >= -1e-9:
        return ground_rgb
    pixels = bytearray(ground_rgb)
    shadowed: set[int] = set()
    for triangle in triangles:
        ground_points: list[Vec3] = []
        for point in triangle.points:
            distance = -max(0.0, float(point[2])) / direction[2]
            ground_points.append(
                (
                    float(point[0]) + direction[0] * distance,
                    float(point[1]) + direction[1] * distance,
                    0.001,
                )
            )
        projected = [_project(point, eye, target, fov, width, height) for point in ground_points]
        if any(point is None for point in projected):
            continue
        points = [point for point in projected if point is not None]
        min_x = max(0, int(math.floor(min(point[0] for point in points))))
        max_x = min(width - 1, int(math.ceil(max(point[0] for point in points))))
        min_y = max(0, int(math.floor(min(point[1] for point in points))))
        max_y = min(height - 1, int(math.ceil(max(point[1] for point in points))))
        area = _edge((points[0][0], points[0][1]), (points[1][0], points[1][1]), (points[2][0], points[2][1]))
        if abs(area) < 1e-9:
            continue
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pixel = (x + 0.5, y + 0.5)
                w0 = _edge((points[1][0], points[1][1]), (points[2][0], points[2][1]), pixel)
                w1 = _edge((points[2][0], points[2][1]), (points[0][0], points[0][1]), pixel)
                w2 = _edge((points[0][0], points[0][1]), (points[1][0], points[1][1]), pixel)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    index = y * width + x
                    if ground_mask[index * 3] >= 128:
                        shadowed.add(index)
    coverage: list[float] = [0.0] * (width * height)
    for index in shadowed:
        coverage[index] = 1.0
    for _ in range(max(0, blur_passes)):
        coverage = _box_blur_mask(coverage, width, height, max(0, blur_radius_px))
    opacity = max(0.0, min(1.0, opacity))
    for index, shadow_strength in enumerate(coverage):
        if shadow_strength <= 1e-6 or ground_mask[index * 3] < 128:
            continue
        factor = 1.0 - opacity * min(1.0, shadow_strength)
        offset = index * 3
        pixels[offset : offset + 3] = bytes(round(channel * factor) for channel in pixels[offset : offset + 3])
    return bytes(pixels)


def compose_ground_rgb(rgb: bytes, silhouette: bytes, ground_mask: bytes, ground_rgb: bytes | None = None) -> bytes:
    pixels = bytearray(rgb)
    for offset in range(0, len(pixels), 3):
        if ground_mask[offset] >= 128 and silhouette[offset] < 128:
            pixels[offset : offset + 3] = ground_rgb[offset : offset + 3] if ground_rgb is not None else b"\x46\x45\x43"
    return bytes(pixels)


def _project(point: Vec3, eye: Vec3, target: Vec3, fov: float, width: int, height: int) -> tuple[float, float, float] | None:
    forward = _normalize(_vector_sub(target, eye))
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    relative = _vector_sub(point, eye)
    depth = _dot(relative, forward)
    if depth <= 0.01:
        return None
    scale = 0.5 * height / math.tan(math.radians(fov) * 0.5)
    return width * 0.5 + _dot(relative, right) * scale / depth, height * 0.5 - _dot(relative, up) * scale / depth, depth


def _edge(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def clip_polygon_above_ground(points: Sequence[Vec3], ground_z: float = 0.0) -> list[Vec3]:
    output: list[Vec3] = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        current_inside = current[2] >= ground_z
        previous_inside = previous[2] >= ground_z
        if current_inside != previous_inside:
            denominator = current[2] - previous[2]
            ratio = (ground_z - previous[2]) / denominator if abs(denominator) > 1e-12 else 0.0
            output.append(
                (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                    ground_z,
                )
            )
        if current_inside:
            output.append(current)
    return output


def rasterize_triangles(
    triangles: Sequence[MeshTriangle],
    transform: Mat4,
    *,
    width: int,
    height: int,
    eye: Vec3,
    target: Vec3,
    fov: float,
) -> tuple[bytes, bytes]:
    background = (8, 11, 17)
    rgb = bytearray(background * (width * height))
    silhouette = bytearray((0, 0, 0) * (width * height))
    depth_buffer = [float("inf")] * (width * height)
    for triangle in triangles:
        world_polygon = clip_polygon_above_ground([transform_point(transform, point) for point in triangle.points])
        for polygon_index in range(1, len(world_polygon) - 1):
            world_points = (world_polygon[0], world_polygon[polygon_index], world_polygon[polygon_index + 1])
            edge_a = _vector_sub(world_points[1], world_points[0])
            edge_b = _vector_sub(world_points[2], world_points[0])
            normal = _normalize(_cross(edge_a, edge_b))
            light = _normalize((-0.6, -0.8, 1.0))
            illumination = 0.78 + 0.35 * abs(_dot(normal, light))
            shaded_color = tuple(max(0, min(255, round(channel * illumination))) for channel in triangle.color)
            projected = [_project(point, eye, target, fov, width, height) for point in world_points]
            if any(point is None for point in projected):
                continue
            points = [point for point in projected if point is not None]
            min_x = max(0, int(math.floor(min(point[0] for point in points))))
            max_x = min(width - 1, int(math.ceil(max(point[0] for point in points))))
            min_y = max(0, int(math.floor(min(point[1] for point in points))))
            max_y = min(height - 1, int(math.ceil(max(point[1] for point in points))))
            area = _edge((points[0][0], points[0][1]), (points[1][0], points[1][1]), (points[2][0], points[2][1]))
            if abs(area) < 1e-9:
                continue
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    pixel = (x + 0.5, y + 0.5)
                    w0 = _edge((points[1][0], points[1][1]), (points[2][0], points[2][1]), pixel)
                    w1 = _edge((points[2][0], points[2][1]), (points[0][0], points[0][1]), pixel)
                    w2 = _edge((points[0][0], points[0][1]), (points[1][0], points[1][1]), pixel)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        w0, w1, w2 = w0 / area, w1 / area, w2 / area
                        depth = w0 * points[0][2] + w1 * points[1][2] + w2 * points[2][2]
                        index = y * width + x
                        if depth < depth_buffer[index]:
                            depth_buffer[index] = depth
                            offset = index * 3
                            rgb[offset : offset + 3] = bytes(shaded_color)
                            silhouette[offset : offset + 3] = b"\xff\xff\xff"
    return bytes(rgb), bytes(silhouette)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height * 3:
        raise ValueError("RGB pixel payload has unexpected size")
    raw = b"".join(b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 6))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_png(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    width = height = color_type = bit_depth = interlace = 0
    compressed = bytearray()
    while offset + 12 <= len(raw):
        length = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if bit_depth != 8 or interlace != 0 or color_type not in (0, 2, 6):
        raise ValueError(f"unsupported PNG format: bit_depth={bit_depth} color_type={color_type} interlace={interlace}")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = width * channels
    decoded = zlib.decompress(bytes(compressed))
    rows: list[bytearray] = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = decoded[cursor]
        cursor += 1
        current = bytearray(decoded[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = current[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                current[index] = (current[index] + left) & 0xFF
            elif filter_type == 2:
                current[index] = (current[index] + up) & 0xFF
            elif filter_type == 3:
                current[index] = (current[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = left + up - up_left
                pa, pb, pc = abs(predictor - left), abs(predictor - up), abs(predictor - up_left)
                current[index] = (current[index] + (left if pa <= pb and pa <= pc else up if pb <= pc else up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter: {filter_type}")
        rows.append(current)
        previous = current
    rgb = bytearray()
    for row in rows:
        for index in range(0, len(row), channels):
            if channels == 1:
                rgb.extend((row[index], row[index], row[index]))
            else:
                rgb.extend(row[index : index + 3])
    return width, height, bytes(rgb)


def _resize_nearest(width: int, height: int, pixels: bytes, out_width: int, out_height: int) -> bytes:
    output = bytearray(out_width * out_height * 3)
    for y in range(out_height):
        source_y = min(height - 1, int(y * height / out_height))
        for x in range(out_width):
            source_x = min(width - 1, int(x * width / out_width))
            source = (source_y * width + source_x) * 3
            target = (y * out_width + x) * 3
            output[target : target + 3] = pixels[source : source + 3]
    return bytes(output)


def write_comparison_sheet(
    pairs: Sequence[tuple[Path, Path]],
    out_file: Path,
    *,
    sample_count: int = 12,
    thumb_width: int = 320,
    thumb_height: int = 180,
) -> JSON:
    if not pairs:
        return {"ok": False, "blocker": "comparison_pairs_missing", "file": str(out_file)}
    if sample_count <= 1:
        selected = [pairs[0]]
    elif len(pairs) <= sample_count:
        selected = list(pairs)
    else:
        selected = [
            pairs[round(index * (len(pairs) - 1) / (sample_count - 1))]
            for index in range(sample_count)
        ]
    width = thumb_width * 2
    height = thumb_height * len(selected)
    canvas = bytearray((245, 245, 245) * (width * height))
    for row_index, (source_path, evih_path) in enumerate(selected):
        for column, path in enumerate((source_path, evih_path)):
            source_width, source_height, pixels = read_png(path)
            resized = _resize_nearest(source_width, source_height, pixels, thumb_width, thumb_height)
            for y in range(thumb_height):
                target_start = ((row_index * thumb_height + y) * width + column * thumb_width) * 3
                source_start = y * thumb_width * 3
                canvas[target_start : target_start + thumb_width * 3] = resized[source_start : source_start + thumb_width * 3]
    write_png(out_file, width, height, bytes(canvas))
    return {
        "ok": True,
        "file": str(out_file),
        "pair_count": len(selected),
        "selected_frame_ids": [int(source_path.stem.split("_")[-1]) for source_path, _ in selected],
    }


def build_dynamic_comparison_mp4(
    pairs: Sequence[tuple[Path, Path]],
    out_dir: Path,
    *,
    fps: int,
) -> JSON:
    comparison_frames_dir = out_dir / "comparison_frames"
    if comparison_frames_dir.exists():
        shutil.rmtree(comparison_frames_dir)
    comparison_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames: list[Path] = []
    frame_ids: list[int] = []
    try:
        for source_path, evih_path in pairs:
            source_width, source_height, source_pixels = read_png(source_path)
            evih_width, evih_height, evih_pixels = read_png(evih_path)
            if (source_width, source_height) != (evih_width, evih_height):
                raise ValueError("comparison frame dimensions do not match")
            combined_width = source_width + evih_width
            combined = bytearray(combined_width * source_height * 3)
            for y in range(source_height):
                source_start = y * source_width * 3
                evih_start = y * evih_width * 3
                target_start = y * combined_width * 3
                combined[target_start : target_start + source_width * 3] = source_pixels[
                    source_start : source_start + source_width * 3
                ]
                combined[
                    target_start + source_width * 3 : target_start + combined_width * 3
                ] = evih_pixels[evih_start : evih_start + evih_width * 3]
            frame_id = int(source_path.stem.split("_")[-1])
            frame_path = comparison_frames_dir / f"frame_{frame_id:06d}.png"
            write_png(frame_path, combined_width, source_height, bytes(combined))
            output_frames.append(frame_path)
            frame_ids.append(frame_id)
    except Exception as exc:
        return {
            "ok": False,
            "blocker": "comparison_dynamic_frames_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "file": str(out_dir / "mimickit_vs_evih_dynamic.mp4"),
        }
    media = create_mp4(output_frames, out_dir / "mimickit_vs_evih_dynamic.mp4", fps)
    media.update(
        comparison_frame_ids=frame_ids,
        comparison_png_count=len(output_frames),
        unique_decoded_frame_count=decoded_mp4_unique_frame_count(Path(str(media.get("file", "")))),
        source_side="left",
        evih_side="right",
    )
    if media.get("ok") and len(output_frames) != len(V3_REQUIRED_FRAME_IDS):
        media["ok"] = False
        media["blocker"] = "comparison_dynamic_frame_count_invalid"
    if media.get("ok") and int(media.get("unique_decoded_frame_count", 0) or 0) < V3_MIN_UNIQUE_DYNAMIC_FRAMES:
        media["ok"] = False
        media["blocker"] = "comparison_dynamic_mp4_static"
    return media


def write_comparison_markdown(
    out_file: Path,
    *,
    pairs: Sequence[tuple[Path, Path]],
    comparison_sheet: Path,
    visual_metric_report: JSON,
    ground_metric_report: JSON,
    rgb_metric_report: JSON,
    visual_review: JSON,
    dynamic_sequence_report: JSON | None = None,
    comparison_mp4: Path | None = None,
) -> JSON:
    if not pairs or not comparison_sheet.is_file():
        return {"ok": False, "blocker": "comparison_markdown_inputs_missing", "file": str(out_file)}
    visual = visual_metric_report.get("metrics", {}) if isinstance(visual_metric_report.get("metrics"), dict) else {}
    ground = ground_metric_report.get("metrics", {}) if isinstance(ground_metric_report.get("metrics"), dict) else {}
    rgb = rgb_metric_report.get("metrics", {}) if isinstance(rgb_metric_report.get("metrics"), dict) else {}
    dynamic = dynamic_sequence_report or {}
    review = visual_review.get("review", {}) if isinstance(visual_review.get("review"), dict) else {}
    review_checks = review.get("checks", {}) if isinstance(review.get("checks"), dict) else {}
    lines = [
        "# MimicKit vs EvihAnimation True-Mesh Review",
        "",
        f"![Comparison sheet]({comparison_sheet.name})",
        "",
        f"Dynamic side-by-side MP4: [`{comparison_mp4.name}`]({comparison_mp4.name})"
        if comparison_mp4 and comparison_mp4.is_file()
        else "Dynamic side-by-side MP4: **missing**",
        "",
        "## Automatic Evidence",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean silhouette IoU | {float(visual.get('mean_silhouette_iou', 0.0)):.6f} |",
        f"| P10 silhouette IoU | {float(visual.get('p10_silhouette_iou', 0.0)):.6f} |",
        f"| Mean ground-mask IoU | {float(ground.get('mean_ground_mask_iou', 0.0)):.6f} |",
        f"| P10 ground-mask IoU | {float(ground.get('p10_ground_mask_iou', 0.0)):.6f} |",
        f"| RGB normalized MAE, report only | {float(rgb.get('mean_absolute_error_normalized', 1.0)):.6f} |",
        f"| Dynamic sequence pass | {bool(dynamic.get('dynamic_sequence_pass'))} |",
        f"| Dynamic output frames | {int(dynamic.get('required_frame_count', 0) or 0)} |",
        f"| Evih MP4 unique decoded frames | {int(dynamic.get('mp4_unique_frame_count', 0) or 0)} |",
        f"| MimicKit unique RGB/silhouette | {int(dynamic.get('source_rgb_unique_frame_count', 0) or 0)} / {int(dynamic.get('source_silhouette_unique_frame_count', 0) or 0)} |",
        f"| Evih unique RGB/silhouette | {int(dynamic.get('evih_rgb_unique_frame_count', 0) or 0)} / {int(dynamic.get('evih_silhouette_unique_frame_count', 0) or 0)} |",
        "",
        "## Manual Review",
        "",
        f"Final visual review pass: **{bool(visual_review.get('visual_review_pass'))}**",
        f"Review evidence matches current artifacts: **{bool(visual_review.get('evidence_matches'))}**",
        "",
    ]
    lines.extend(
        f"- [{'x' if bool(review_checks.get(name)) else ' '}] {name.replace('_', ' ')}"
        for name in REQUIRED_VISUAL_REVIEW_CHECKS
    )
    lines.extend(["", "## Frame Pairs", "", "| Frame | MimicKit source | Evih replay |", "| ---: | --- | --- |"])
    for source_path, evih_path in pairs:
        frame_name = source_path.stem.split("_")[-1]
        lines.append(f"| {frame_name} | `{source_path}` | `{evih_path}` |")
    lines.extend(
        [
            "",
            "This document is generated evidence. It never changes `visual_review.json` or approves the final gate.",
            "",
        ]
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return {"ok": True, "file": str(out_file), "pair_count": len(pairs)}


def create_mp4(frames: Sequence[Path], out_file: Path, fps: int) -> JSON:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    report: JSON = {"file": str(out_file), "fps": fps, "expected_frame_count": len(frames), "ok": False, "blocker": ""}
    if not frames:
        report["blocker"] = "png_sequence_missing"
        return report
    if not ffmpeg or not ffprobe:
        report["blocker"] = "ffmpeg_or_ffprobe_missing"
        return report
    list_file = out_file.with_suffix(".frames.txt")
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("".join(f"file '{path.resolve()}'\nduration {1.0 / fps:.12f}\n" for path in frames) + f"file '{frames[-1].resolve()}'\n", encoding="utf-8")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-r", str(fps), "-frames:v", str(len(frames)), "-pix_fmt", "yuv420p", str(out_file),
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(out_file),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stream = json.loads(probe.stdout).get("streams", [{}])[0]
    except Exception:
        stream = {}
    frame_count = int(stream.get("nb_read_frames", 0) or 0)
    rate = str(stream.get("avg_frame_rate", "0/1"))
    numerator, denominator = (rate.split("/", 1) + ["1"])[:2]
    actual_fps = float(numerator) / max(float(denominator), 1.0)
    expected_width, expected_height, _ = read_png(frames[0])
    report.update(
        command=command,
        returncode=process.returncode,
        stderr_tail=process.stderr[-2000:],
        ffprobe=stream,
        frame_count=frame_count,
        actual_fps=actual_fps,
        size_bytes=out_file.stat().st_size if out_file.is_file() else 0,
    )
    report["ok"] = bool(
        process.returncode == 0
        and probe.returncode == 0
        and stream.get("codec_name") == "h264"
        and int(stream.get("width", 0) or 0) == expected_width
        and int(stream.get("height", 0) or 0) == expected_height
        and math.isclose(actual_fps, float(fps), rel_tol=0.01, abs_tol=0.01)
        and frame_count == len(frames)
        and report["size_bytes"] > 0
    )
    report["blocker"] = "" if report["ok"] else "mp4_probe_failed"
    return report


def locate_reference_frames(manifest: JSON, manifest_path: Path, frame_ids: Sequence[int]) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    bases = [manifest_path.parent]
    render_values = _recursive_values(manifest, {"render_dir", "frames_dir", "rgb_frames_dir"})
    render_dir = _first_existing_path(render_values, bases)
    rgb_dirs: list[Path] = []
    silhouette_dirs: list[Path] = []
    ground_mask_dirs: list[Path] = []
    if render_dir:
        rgb_dirs.extend((render_dir, render_dir / "frames"))
        silhouette_dirs.extend((render_dir / "silhouettes", render_dir / "silhouette_frames"))
        ground_mask_dirs.extend((render_dir / "ground_masks", render_dir / "ground_mask_frames"))
    silhouette_values = _recursive_values(manifest, {"silhouette_dir", "silhouette_frames_dir"})
    silhouette_candidate = _first_existing_path(silhouette_values, bases)
    if silhouette_candidate:
        silhouette_dirs.insert(0, silhouette_candidate)

    def find_frame(directories: Sequence[Path], frame_id: int) -> Path | None:
        names = (f"frame_{frame_id:06d}.png", f"frame_{frame_id:04d}.png", f"{frame_id:06d}.png")
        for directory in directories:
            for name in names:
                path = directory / name
                if path.is_file():
                    return path
        return None

    rgb = {frame: path for frame in frame_ids if (path := find_frame(rgb_dirs, frame))}
    silhouettes = {frame: path for frame in frame_ids if (path := find_frame(silhouette_dirs, frame))}
    ground_masks = {frame: path for frame in frame_ids if (path := find_frame(ground_mask_dirs, frame))}
    return rgb, silhouettes, ground_masks


def _mask_stats(path: Path) -> tuple[int, int, set[int], tuple[float, float], tuple[int, int, int, int] | None]:
    width, height, pixels = read_png(path)
    mask: set[int] = set()
    xs: list[int] = []
    ys: list[int] = []
    for index in range(width * height):
        offset = index * 3
        if max(pixels[offset : offset + 3]) >= 128:
            mask.add(index)
            xs.append(index % width)
            ys.append(index // width)
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (0.0, 0.0)
    bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else None
    return width, height, mask, centroid, bbox


def build_visual_metric_report(
    source_silhouettes: dict[int, Path],
    evih_silhouettes: dict[int, Path],
    *,
    thresholds: JSON,
) -> JSON:
    frame_ids = sorted(set(source_silhouettes) & set(evih_silhouettes))
    rows: list[JSON] = []
    evih_hashes: set[str] = set()
    for frame_id in frame_ids:
        source_width, source_height, source_mask, source_centroid, source_bbox = _mask_stats(source_silhouettes[frame_id])
        evih_width, evih_height, evih_mask, evih_centroid, evih_bbox = _mask_stats(evih_silhouettes[frame_id])
        if (source_width, source_height) != (evih_width, evih_height):
            continue
        union = len(source_mask | evih_mask)
        intersection = len(source_mask & evih_mask)
        diagonal = math.hypot(source_width, source_height)
        centroid_error = math.hypot(source_centroid[0] - evih_centroid[0], source_centroid[1] - evih_centroid[1]) / diagonal
        source_area = (source_bbox[2] - source_bbox[0] + 1) * (source_bbox[3] - source_bbox[1] + 1) if source_bbox else 0
        evih_area = (evih_bbox[2] - evih_bbox[0] + 1) * (evih_bbox[3] - evih_bbox[1] + 1) if evih_bbox else 0
        evih_hashes.add(sha256_file(evih_silhouettes[frame_id]))
        rows.append(
            {
                "frame_id": frame_id,
                "silhouette_iou": intersection / union if union else 0.0,
                "centroid_error_diagonal_ratio": centroid_error,
                "bbox_area_ratio": evih_area / source_area if source_area else 0.0,
            }
        )
    ious = [row["silhouette_iou"] for row in rows]
    centroids = [row["centroid_error_diagonal_ratio"] for row in rows]
    ratios = [row["bbox_area_ratio"] for row in rows]
    metrics = {
        "mean_silhouette_iou": sum(ious) / len(ious) if ious else 0.0,
        "p10_silhouette_iou": percentile(ious, 0.10),
        "mean_centroid_error_diagonal_ratio": sum(centroids) / len(centroids) if centroids else 1.0,
        "p95_centroid_error_diagonal_ratio": percentile(centroids, 0.95),
        "bbox_area_ratio_p10": percentile(ratios, 0.10),
        "bbox_area_ratio_p90": percentile(ratios, 0.90),
        "motion_visible": len(evih_hashes) > 1,
    }
    checks = {
        "all_requested_frames_paired": bool(source_silhouettes) and len(rows) == len(source_silhouettes) == len(evih_silhouettes),
        "mean_silhouette_iou": metrics["mean_silhouette_iou"] >= float(thresholds["mean_silhouette_iou_min"]),
        "p10_silhouette_iou": metrics["p10_silhouette_iou"] >= float(thresholds["p10_silhouette_iou_min"]),
        "mean_centroid_error": metrics["mean_centroid_error_diagonal_ratio"] <= float(thresholds["mean_centroid_error_max"]),
        "p95_centroid_error": metrics["p95_centroid_error_diagonal_ratio"] <= float(thresholds["p95_centroid_error_max"]),
        "bbox_area_ratio_p10": metrics["bbox_area_ratio_p10"] >= float(thresholds["bbox_area_ratio_p10_min"]),
        "bbox_area_ratio_p90": metrics["bbox_area_ratio_p90"] <= float(thresholds["bbox_area_ratio_p90_max"]),
        "motion_visible": metrics["motion_visible"],
    }
    passed = all(checks.values())
    return {
        "schema_version": 2,
        "requested_thresholds": thresholds,
        "paired_frame_ids": frame_ids,
        "per_frame": rows,
        "metrics": metrics,
        "checks": checks,
        "visual_metric_pass": passed,
        "blocker": "" if passed else "silhouette_visual_metric_failed",
    }


def build_ground_metric_report(source_masks: dict[int, Path], evih_masks: dict[int, Path]) -> JSON:
    frame_ids = sorted(set(source_masks) & set(evih_masks))
    rows: list[JSON] = []
    for frame_id in frame_ids:
        source_width, source_height, source_mask, _, _ = _mask_stats(source_masks[frame_id])
        evih_width, evih_height, evih_mask, _, _ = _mask_stats(evih_masks[frame_id])
        if (source_width, source_height) != (evih_width, evih_height):
            continue
        union = len(source_mask | evih_mask)
        intersection = len(source_mask & evih_mask)
        source_horizon = min((index // source_width for index in source_mask), default=source_height)
        evih_horizon = min((index // evih_width for index in evih_mask), default=evih_height)
        rows.append(
            {
                "frame_id": frame_id,
                "ground_mask_iou": intersection / union if union else 0.0,
                "horizon_error_height_ratio": abs(source_horizon - evih_horizon) / max(source_height, 1),
            }
        )
    ious = [row["ground_mask_iou"] for row in rows]
    horizons = [row["horizon_error_height_ratio"] for row in rows]
    metrics = {
        "mean_ground_mask_iou": sum(ious) / len(ious) if ious else 0.0,
        "p10_ground_mask_iou": percentile(ious, 0.10),
        "max_horizon_error_height_ratio": max(horizons) if horizons else 1.0,
    }
    checks = {
        "all_requested_frames_paired": bool(source_masks) and len(rows) == len(source_masks) == len(evih_masks),
        "mean_ground_mask_iou": metrics["mean_ground_mask_iou"] >= 0.95,
        "p10_ground_mask_iou": metrics["p10_ground_mask_iou"] >= 0.90,
        "horizon_error": metrics["max_horizon_error_height_ratio"] <= 0.02,
    }
    return {
        "schema_version": 1,
        "paired_frame_ids": frame_ids,
        "per_frame": rows,
        "metrics": metrics,
        "checks": checks,
        "scene_visual_metric_pass": all(checks.values()),
        "blocker": "" if all(checks.values()) else "ground_scene_visual_metric_failed",
    }


def build_rgb_metric_report(source_rgb: dict[int, Path], evih_rgb: dict[int, Path]) -> JSON:
    frame_ids = sorted(set(source_rgb) & set(evih_rgb))
    rows: list[JSON] = []
    for frame_id in frame_ids:
        source_width, source_height, source_pixels = read_png(source_rgb[frame_id])
        evih_width, evih_height, evih_pixels = read_png(evih_rgb[frame_id])
        if (source_width, source_height) != (evih_width, evih_height):
            continue
        channel_count = len(source_pixels)
        absolute_error = sum(abs(source_pixels[index] - evih_pixels[index]) for index in range(channel_count))
        pixel_count = max(source_width * source_height, 1)
        rows.append(
            {
                "frame_id": frame_id,
                "mean_absolute_error_255": absolute_error / max(channel_count, 1),
                "mean_absolute_error_normalized": absolute_error / max(channel_count * 255, 1),
                "source_mean_rgb": [
                    sum(source_pixels[channel::3]) / pixel_count
                    for channel in range(3)
                ],
                "evih_mean_rgb": [
                    sum(evih_pixels[channel::3]) / pixel_count
                    for channel in range(3)
                ],
            }
        )
    errors = [float(row["mean_absolute_error_normalized"]) for row in rows]
    return {
        "schema_version": 1,
        "report_only": True,
        "paired_frame_ids": frame_ids,
        "per_frame": rows,
        "metrics": {
            "mean_absolute_error_normalized": sum(errors) / len(errors) if errors else 1.0,
            "p90_absolute_error_normalized": percentile(errors, 0.90),
        },
        "complete": bool(source_rgb) and len(rows) == len(source_rgb) == len(evih_rgb),
        "blocker": "" if source_rgb and len(rows) == len(source_rgb) == len(evih_rgb) else "rgb_similarity_pairs_missing",
    }


def render_true_mesh(
    rows: Sequence[JSON],
    frame_ids: Sequence[int],
    mesh_asset: Path,
    joint_order: JSON,
    contract: JSON,
    out_dir: Path,
    *,
    width: int,
    height: int,
    source_rig: JSON,
    binding: JSON,
) -> JSON:
    model = load_mesh_model(mesh_asset)
    rows_by_frame = {int(row["frame"]): row for row in rows}
    initial_row = {
        "root_pos_m": [0.0, 0.0, 0.0],
        "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0],
        "dof_pos": [0.0] * len(rows[0].get("dof_pos", [])),
    }
    rgb_dir = out_dir / "frames"
    silhouette_dir = out_dir / "silhouettes"
    ground_mask_dir = out_dir / "ground_masks"
    rgb_frames: list[Path] = []
    silhouette_frames: list[Path] = []
    ground_mask_frames: list[Path] = []
    missing_rows: list[int] = []
    mesh_triangle_count = 0
    mapped_dof_node_count = 0
    max_body_pos_error_m = 0.0
    max_body_rot_error_rad = 0.0
    for frame_id in frame_ids:
        row = rows_by_frame.get(frame_id)
        if row is None:
            missing_rows.append(frame_id)
            continue
        triangles, mapped_dof_nodes, frame_pos_error, frame_rot_error = posed_mesh_triangles(model, row, joint_order, source_rig, binding, initial_row)
        max_body_pos_error_m = max(max_body_pos_error_m, frame_pos_error)
        max_body_rot_error_rad = max(max_body_rot_error_rad, frame_rot_error)
        mesh_triangle_count = max(mesh_triangle_count, len(triangles))
        mapped_dof_node_count = max(mapped_dof_node_count, mapped_dof_nodes)
        eye, target, fov = camera_for_frame(contract, frame_id, width, height)
        rgb, silhouette = rasterize_triangles(
            triangles,
            identity_matrix(),
            width=width,
            height=height,
            eye=eye,
            target=target,
            fov=fov,
        )
        ground_mask, ground_rgb = render_ground_scene(width, height, eye, target, fov, contract)
        ground_rgb = apply_ground_shadows(
            ground_rgb,
            ground_mask,
            triangles,
            width=width,
            height=height,
            eye=eye,
            target=target,
            fov=fov,
            contract=contract,
        )
        rgb = compose_ground_rgb(rgb, silhouette, ground_mask, ground_rgb)
        rgb_path = rgb_dir / f"frame_{frame_id:06d}.png"
        silhouette_path = silhouette_dir / f"frame_{frame_id:06d}.png"
        ground_mask_path = ground_mask_dir / f"frame_{frame_id:06d}.png"
        write_png(rgb_path, width, height, rgb)
        write_png(silhouette_path, width, height, silhouette)
        write_png(ground_mask_path, width, height, ground_mask)
        rgb_frames.append(rgb_path)
        silhouette_frames.append(silhouette_path)
        ground_mask_frames.append(ground_mask_path)
    return {
        "renderer": "evih_stdlib_glb_triangle_rasterizer_v3",
        "pose_application_mode": "sidecar_root_and_mapped_joint_dof_glb_pose",
        "dof_pose_applied": mapped_dof_node_count > 0,
        "mapped_dof_node_count": mapped_dof_node_count,
        "mesh_triangle_count": mesh_triangle_count,
        "mesh_primitive_count": len(model.primitives),
        "requested_frame_ids": list(frame_ids),
        "rendered_frame_ids": [int(path.stem.split("_")[-1]) for path in rgb_frames],
        "rgb_frames": [str(path) for path in rgb_frames],
        "silhouette_frames": [str(path) for path in silhouette_frames],
        "ground_mask_frames": [str(path) for path in ground_mask_frames],
        "rgb_png_count": len(rgb_frames),
        "silhouette_png_count": len(silhouette_frames),
        "ground_mask_png_count": len(ground_mask_frames),
        "missing_replay_rows": missing_rows,
        "max_body_pos_error_m": max_body_pos_error_m,
        "max_body_rot_error_rad": max_body_rot_error_rad,
        "fk_compare_pass": max_body_pos_error_m <= 1e-6 and max_body_rot_error_rad <= 1e-5,
        "render_pass": bool(mesh_triangle_count and mapped_dof_node_count > 0 and not missing_rows and len(rgb_frames) == len(frame_ids) and len(silhouette_frames) == len(frame_ids) and len(ground_mask_frames) == len(frame_ids) and max_body_pos_error_m <= 1e-6 and max_body_rot_error_rad <= 1e-5),
    }


def _draw_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int], radius: int = 1) -> None:
    for py in range(max(0, y - radius), min(height, y + radius + 1)):
        for px in range(max(0, x - radius), min(width, x + radius + 1)):
            offset = (py * width + px) * 3
            pixels[offset : offset + 3] = bytes(color)


def _draw_line(pixels: bytearray, width: int, height: int, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        _draw_pixel(pixels, width, height, x, y, color, radius)


def render_skeleton_or_geom(
    rows: Sequence[JSON],
    joint_order: JSON,
    out_dir: Path,
    *,
    frames: int,
    stride: int,
    fps: int,
    width: int,
    height: int,
    geom: bool,
) -> JSON:
    frame_ids = list(range(0, min(frames, len(rows)), max(1, stride)))
    body_order = [str(value) for value in joint_order.get("body_order", [])]
    parents = {"pelvis": ""}
    for joint in joint_order.get("joints", []):
        if isinstance(joint, dict):
            parents[str(joint.get("body_name", ""))] = str(joint.get("parent_body_name", ""))
    output_frames: list[Path] = []
    for frame_id in frame_ids:
        row = rows[frame_id]
        pixels = bytearray((8, 11, 17) * (width * height))
        root = row.get("root_pos_m", [0.0, 0.0, 0.0])
        dofs = row.get("dof_pos", [])
        positions: dict[str, tuple[int, int]] = {}
        for index, body in enumerate(body_order):
            column = index % 5 - 2
            level = index // 5
            wobble = float(dofs[index % len(dofs)]) if dofs else 0.0
            x = round(width * 0.5 + column * width * 0.055 + float(root[0]) * 18 + math.sin(wobble) * 12)
            y = round(height * 0.72 - level * height * 0.11 - float(root[2]) * 12 + math.cos(wobble) * 8)
            positions[body] = (x, y)
        for body, position in positions.items():
            parent = parents.get(body, "")
            if parent in positions:
                _draw_line(pixels, width, height, positions[parent], position, (220, 225, 235), 4 if geom else 1)
            _draw_pixel(pixels, width, height, *position, (240, 190, 70) if body in ("sword", "shield") else (120, 210, 255), 6 if geom else 2)
        path = out_dir / "frames" / f"frame_{frame_id:06d}.png"
        write_png(path, width, height, bytes(pixels))
        output_frames.append(path)
    mp4_file = out_dir / ("character_replay.mp4" if geom else "skeleton_replay.mp4")
    media = create_mp4(output_frames, mp4_file, fps)
    mode = "character_geom" if geom else "skeleton"
    report = {
        "schema_version": 2,
        "mode": mode,
        "mesh_scope": False,
        "frames_loaded": len(rows),
        "source_frame_ids": frame_ids,
        "png_count": len(output_frames),
        "mp4_file": str(mp4_file),
        "mp4_ok": bool(media.get("ok")),
        "motion_visible": len({sha256_file(path) for path in output_frames}) > 1,
        "skeleton_replay_pass": bool(output_frames and media.get("ok")),
        "character_geom_replay_pass": bool(geom and output_frames and media.get("ok")),
        "blocker": "" if output_frames and media.get("ok") else str(media.get("blocker") or "replay_render_failed"),
        "media": media,
    }
    write_json(out_dir / ("character_replay_meta.json" if geom else "skeleton_replay_meta.json"), report)
    return report


def default_thresholds() -> JSON:
    return {
        "mean_silhouette_iou_min": 0.90,
        "p10_silhouette_iou_min": 0.80,
        "mean_centroid_error_max": 0.02,
        "p95_centroid_error_max": 0.04,
        "bbox_area_ratio_p10_min": 0.85,
        "bbox_area_ratio_p90_max": 1.15,
    }


def _frame_set_sha256(frames: dict[int, Path]) -> str:
    if not frames:
        return ""
    return stable_json_sha256(
        {
            "frames": [
                {"frame_id": int(frame_id), "sha256": sha256_file(path)}
                for frame_id, path in sorted(frames.items())
            ]
        }
    )


def decoded_mp4_unique_frame_count(path: Path) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.is_file():
        return 0
    process = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        return 0
    return len(
        {
            line.rsplit(",", 1)[-1].strip()
            for line in process.stdout.splitlines()
            if line and not line.startswith("#") and "," in line
        }
    )


def build_dynamic_sequence_report(
    *,
    frame_ids: Sequence[int],
    source_rgb: dict[int, Path],
    source_silhouettes: dict[int, Path],
    evih_rgb: dict[int, Path],
    evih_silhouettes: dict[int, Path],
    media: JSON,
) -> JSON:
    required_ids = list(V3_REQUIRED_FRAME_IDS)

    def unique_count(frames: dict[int, Path]) -> int:
        return len({sha256_file(path) for path in frames.values() if path.is_file()})

    counts = {
        "source_rgb_unique_frame_count": unique_count(source_rgb),
        "source_silhouette_unique_frame_count": unique_count(source_silhouettes),
        "evih_rgb_unique_frame_count": unique_count(evih_rgb),
        "evih_silhouette_unique_frame_count": unique_count(evih_silhouettes),
    }
    mp4_file = Path(str(media.get("file", "")))
    mp4_unique_frame_count = decoded_mp4_unique_frame_count(mp4_file)
    checks = {
        "requested_frame_ids_exact": list(frame_ids) == required_ids,
        "source_rgb_frame_ids_exact": sorted(source_rgb) == required_ids,
        "source_silhouette_frame_ids_exact": sorted(source_silhouettes) == required_ids,
        "evih_rgb_frame_ids_exact": sorted(evih_rgb) == required_ids,
        "evih_silhouette_frame_ids_exact": sorted(evih_silhouettes) == required_ids,
        "mp4_frame_count_exact": bool(media.get("ok")) and int(media.get("frame_count", 0) or 0) == len(required_ids),
        "mp4_unique_frame_count_sufficient": mp4_unique_frame_count >= V3_MIN_UNIQUE_DYNAMIC_FRAMES,
        **{
            name.replace("_count", "_sufficient"): count >= V3_MIN_UNIQUE_DYNAMIC_FRAMES
            for name, count in counts.items()
        },
    }
    blocker = next((f"dynamic_sequence_failed:{name}" for name, passed in checks.items() if not passed), "")
    return {
        "schema_version": 1,
        "required_frame_ids": required_ids,
        "required_frame_count": len(required_ids),
        "minimum_unique_dynamic_frames": V3_MIN_UNIQUE_DYNAMIC_FRAMES,
        "mp4_unique_frame_count": mp4_unique_frame_count,
        **counts,
        "checks": checks,
        "dynamic_sequence_pass": not blocker,
        "blocker": blocker,
    }


def build_visual_review_evidence(
    *,
    mesh_reference_manifest_path: Path,
    mesh_asset: Path,
    contract: JSON,
    comparison_sheet: Path,
    comparison_mp4: Path,
    source_rgb: dict[int, Path],
    evih_rgb: dict[int, Path],
) -> JSON:
    return {
        "schema_version": 1,
        "mimickit_mesh_reference_manifest_sha256": sha256_file(mesh_reference_manifest_path),
        "mesh_asset_sha256": sha256_file(mesh_asset),
        "source_scene_contract_sha256": scene_contract_sha256(contract) if contract else "",
        "comparison_sheet_sha256": sha256_file(comparison_sheet),
        "comparison_mp4_sha256": sha256_file(comparison_mp4),
        "source_rgb_frame_set_sha256": _frame_set_sha256(source_rgb),
        "evih_rgb_frame_set_sha256": _frame_set_sha256(evih_rgb),
    }


def ensure_visual_review_template(path: Path, evidence: JSON) -> bool:
    if path.is_file():
        try:
            current = read_json(path)
        except Exception:
            return False
        checks = current.get("checks", {}) if isinstance(current.get("checks"), dict) else {}
        current_evidence = current.get("evidence", {}) if isinstance(current.get("evidence"), dict) else {}
        is_unreviewed = (
            not bool(current.get("visual_review_pass"))
            and not str(current.get("reviewer", "")).strip()
            and not any(bool(value) for value in checks.values())
        )
        if current_evidence == evidence:
            return False
        if not is_unreviewed:
            archive = path.with_name(f"{path.stem}.stale.{sha256_file(path)[:12]}{path.suffix}")
            if not archive.is_file():
                write_json(archive, current)
    write_json(
        path,
        {
            "schema_version": 2,
            "visual_review_pass": False,
            "checks": {name: False for name in REQUIRED_VISUAL_REVIEW_CHECKS},
            "reviewer": "",
            "reviewed_at_utc": "",
            "evidence": evidence,
            "notes": "Generated review template. Manual confirmation is required.",
        },
    )
    return True


FRAMEWORK_PROVENANCE_EXACT = {
    "ai4animation_mode": "CAPTURE",
    "actor_component": "ai4animation.Components.Actor.Actor",
    "mesh_component": "ai4animation.Standalone.RigidNodeMesh.RigidNodeMesh",
    "render_pipeline": "ai4animation.Standalone.RenderPipeline.RenderPipeline",
    "silhouette_derivation": "renderpipeline_semantic_character_with_ground_depth_occluder",
    "ground_mask_derivation": "complement_of_renderpipeline_semantic_character",
    "mesh_mode": "rigid_node",
    "rigid_node_update_mode": "actor_entity_world_dynamic_vertex_buffer",
}
FRAMEWORK_REQUIRED_CAPTURE_PASSES = {"blank", "character_only", "ground_only", "full_scene"}
FRAMEWORK_PROVENANCE_ARTIFACTS = (
    "framework_capture_script",
    "ai4animation_core_module",
    "entity_module",
    "actor_module",
    "rigid_node_mesh_module",
    "standalone_module",
    "render_pipeline_module",
    "replay_module",
    "basic_vertex_shader",
    "grid_shader",
    "mesh_asset",
    "pose_dof_replay",
    "body_world_replay",
    "mesh_binding_contract",
    "scene_contract",
    "rigid_node_transform_report",
)


def validate_framework_renderer_provenance(
    provenance: JSON,
    expected_paths: dict[str, Path] | None = None,
) -> JSON:
    checks: JSON = {
        f"{name}_exact": provenance.get(name) == value
        for name, value in FRAMEWORK_PROVENANCE_EXACT.items()
    }
    capture_passes = provenance.get("capture_passes", [])
    checks["capture_passes_exact"] = (
        isinstance(capture_passes, list)
        and set(str(value) for value in capture_passes) == FRAMEWORK_REQUIRED_CAPTURE_PASSES
        and len(capture_passes) == len(FRAMEWORK_REQUIRED_CAPTURE_PASSES)
    )
    python_executable = Path(str(provenance.get("python_executable", "")))
    checks["python_executable_exists"] = python_executable.is_file()
    for name in FRAMEWORK_PROVENANCE_ARTIFACTS:
        value = str(provenance.get(name, "")).strip()
        path = Path(value) if value else None
        declared_hash = str(provenance.get(f"{name}_sha256", "")).strip()
        checks[f"{name}_exists"] = bool(path and path.is_file())
        checks[f"{name}_sha256_matches"] = bool(
            path and path.is_file() and declared_hash and declared_hash == sha256_file(path)
        )
        if expected_paths and name in expected_paths:
            expected = expected_paths[name].resolve()
            checks[f"{name}_matches_expected_path"] = bool(path and path.resolve() == expected)
    passed = bool(provenance) and all(checks.values())
    return {
        "schema_version": 1,
        "checks": checks,
        "framework_renderer_provenance_valid": passed,
        "blocker": "" if passed else "framework_renderer_provenance_invalid",
    }


def validate_visual_review(path: Path | None, expected_evidence: JSON | None = None) -> JSON:
    report: JSON = {
        "schema_version": 3,
        "file": str(path) if path else "",
        "required_checks": list(REQUIRED_VISUAL_REVIEW_CHECKS),
        "expected_evidence": expected_evidence or {},
        "evidence_matches": False,
        "visual_review_pass": False,
        "blocker": "visual_review_missing_or_failed",
    }
    if not path or not path.is_file():
        return report
    try:
        review = read_json(path)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    checks = review.get("checks", {}) if isinstance(review.get("checks"), dict) else {}
    missing_or_failed = [name for name in REQUIRED_VISUAL_REVIEW_CHECKS if not bool(checks.get(name))]
    reviewer_present = bool(str(review.get("reviewer", "")).strip())
    reviewed_at_present = bool(str(review.get("reviewed_at_utc", "")).strip())
    review_evidence = review.get("evidence", {}) if isinstance(review.get("evidence"), dict) else {}
    expected = expected_evidence or {}
    evidence_mismatches = [
        name
        for name, value in expected.items()
        if review_evidence.get(name) != value
    ]
    evidence_matches = bool(expected) and not evidence_mismatches
    report.update(
        review=review,
        missing_or_failed_checks=missing_or_failed,
        reviewer_present=reviewer_present,
        reviewed_at_present=reviewed_at_present,
        evidence_mismatches=evidence_mismatches,
        evidence_matches=evidence_matches,
        visual_review_pass=(
            bool(review.get("visual_review_pass"))
            and not missing_or_failed
            and reviewer_present
            and reviewed_at_present
            and evidence_matches
        ),
    )
    if report["visual_review_pass"]:
        report["blocker"] = ""
    elif (bool(review.get("visual_review_pass")) or reviewer_present or any(bool(value) for value in checks.values())) and not evidence_matches:
        report["blocker"] = "visual_review_evidence_mismatch"
    return report


def build_true_mesh_replay(
    *,
    package_dir: Path,
    mesh_asset: Path,
    mesh_reference_manifest_path: Path,
    out_dir: Path,
    scene_contract_path: Path | None = None,
    visual_review_path: Path | None = None,
    thresholds: JSON | None = None,
) -> JSON:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(mesh_reference_manifest_path)
    reference_gate = validate_mesh_reference_manifest(manifest)
    package_report = validate_package_files(package_dir)
    structure = inspect_glb(mesh_asset)
    joint_order: JSON = {}
    if (package_dir / "joint_order.json").is_file():
        joint_order = read_json(package_dir / "joint_order.json")
    mesh_binding_contract = {}
    if (package_dir / "mesh_binding_contract.json").is_file():
        mesh_binding_contract = read_json(package_dir / "mesh_binding_contract.json")
    binding = validate_mesh_binding(mesh_asset, joint_order, structure, mesh_binding_contract=mesh_binding_contract)
    contract, contract_source = extract_scene_contract(manifest, mesh_reference_manifest_path, scene_contract_path)
    frame_ids = reference_gate.get("frame_ids", [])
    timing = contract.get("timing", {}) if isinstance(contract.get("timing"), dict) else {}
    resolution = contract.get("resolution", {}) if isinstance(contract.get("resolution"), dict) else {}
    width = int(resolution.get("width", contract.get("width", 960)) or 960)
    height = int(resolution.get("height", contract.get("height", 540)) or 540)
    stride = int(timing.get("frame_stride", contract.get("frame_stride", 5)) or 5)
    fps = int(timing.get("fps", contract.get("mp4_fps", 12)) or 12)
    scene_report = compare_scene_contract(contract, width=width, height=height, stride=stride, fps=fps, frame_ids=frame_ids)
    hash_report = compare_consumed_hashes(manifest, package_report, mesh_asset, contract)
    write_json(out_dir / "asset_structure_manifest.json", structure)
    write_json(out_dir / "mesh_binding_report.json", binding)
    write_json(out_dir / "scene_contract_compare_report.json", scene_report)
    write_json(out_dir / "applied_scene_contract.json", scene_report.get("applied_scene_contract", {}))
    write_json(out_dir / "source_hash_compare_report.json", hash_report)
    if contract:
        write_json(out_dir / "scene_contract_v3.json", contract)

    preflight_blocker = first_blocker(
        [
            (bool(reference_gate.get("ok")), str(reference_gate.get("blocker") or "mimickit_mesh_reference_not_passing")),
            (bool(package_report.get("ok")), str(package_report.get("blocker") or "mesh_package_sidecars_missing")),
            (bool(structure.get("ok")), str(structure.get("blocker") or "mesh_asset_structure_invalid")),
            (bool(binding.get("mesh_binding_pass")), str(binding.get("blocker") or "mesh_binding_incomplete")),
            (bool(package_report.get("data_binding_ok")), "body_world_replay_invalid"),
            (bool(scene_report.get("scene_contract_compare_pass")), str(scene_report.get("blocker") or "scene_contract_compare_failed")),
            (bool(hash_report.get("hash_match_pass")), str(hash_report.get("blocker") or "source_hash_mismatch")),
        ]
    )
    base: JSON = {
        "schema_version": 2,
        "mode": "true_mesh",
        "mesh_scope": True,
        "true_mesh_reference_available": bool(reference_gate.get("ok")),
        "mimickit_mesh_reference_manifest": str(mesh_reference_manifest_path),
        "mesh_asset": str(mesh_asset),
        "package_dir": str(package_dir),
        "scene_contract_source": contract_source,
        "source_scene_contract_sha256": scene_report.get("source_scene_contract_sha256", ""),
        "evih_scene_contract_sha256": scene_report.get("evih_scene_contract_sha256", ""),
        "source_hashes": hash_report.get("hashes", {}),
        "data_binding_ok": bool(package_report.get("data_binding_ok")),
        "mesh_binding_pass": bool(binding.get("mesh_binding_pass")),
        "fk_compare_pass": False,
        "scene_contract_compare_pass": bool(scene_report.get("scene_contract_compare_pass")),
        "scene_visual_metric_pass": False,
        "visual_metric_pass": False,
        "dynamic_sequence_pass": False,
        "visual_review_pass": False,
        "media_ok": False,
        "evih_mesh_replay_pass": False,
        "blocker": preflight_blocker,
        "reports": {
            "asset_structure": str(out_dir / "asset_structure_manifest.json"),
            "mesh_binding": str(out_dir / "mesh_binding_report.json"),
            "scene_contract_compare": str(out_dir / "scene_contract_compare_report.json"),
            "applied_scene_contract": str(out_dir / "applied_scene_contract.json"),
            "source_hash_compare": str(out_dir / "source_hash_compare_report.json"),
        },
    }
    if preflight_blocker:
        write_json(out_dir / "visual_asset_manifest.json", base)
        write_json(out_dir / "mesh_replay_meta.json", base)
        return base

    files = resolve_package_files(package_dir)
    rows = load_replay_rows(files["pose_dof_replay"])
    body_rows = {int(row["frame"]): row for row in load_replay_rows(files["body_world_replay"])}
    for row in rows:
        body_row = body_rows.get(int(row["frame"]))
        if body_row:
            row["body_pos_m"] = body_row.get("body_pos_m", [])
            row["body_rot_xyzw"] = body_row.get("body_rot_xyzw", [])
    row_validation = validate_replay_rows(rows, joint_order)
    source_rig = read_json(files["source_rig_asset_spec"])
    render_report = render_true_mesh(rows, frame_ids, mesh_asset, joint_order, contract, out_dir, width=width, height=height, source_rig=source_rig, binding=binding)
    rgb_frames = [Path(value) for value in render_report.get("rgb_frames", [])]
    media = create_mp4(rgb_frames, out_dir / "mesh_replay.mp4", fps)
    source_rgb, source_silhouettes, source_ground_masks = locate_reference_frames(manifest, mesh_reference_manifest_path, frame_ids)
    evih_rgb = {int(path.stem.split("_")[-1]): path for path in (out_dir / "frames").glob("frame_*.png")}
    evih_silhouettes = {int(path.stem.split("_")[-1]): path for path in (out_dir / "silhouettes").glob("frame_*.png")}
    evih_ground_masks = {int(path.stem.split("_")[-1]): path for path in (out_dir / "ground_masks").glob("frame_*.png")}
    dynamic_sequence_report = build_dynamic_sequence_report(
        frame_ids=frame_ids,
        source_rgb=source_rgb,
        source_silhouettes=source_silhouettes,
        evih_rgb=evih_rgb,
        evih_silhouettes=evih_silhouettes,
        media=media,
    )
    metric_report = build_visual_metric_report(source_silhouettes, evih_silhouettes, thresholds=thresholds or default_thresholds())
    ground_metric_report = build_ground_metric_report(source_ground_masks, evih_ground_masks)
    rgb_metric_report = build_rgb_metric_report(source_rgb, evih_rgb)
    comparison_pairs = [(source_rgb[frame], out_dir / "frames" / f"frame_{frame:06d}.png") for frame in frame_ids if frame in source_rgb]
    sheet_report = write_comparison_sheet(comparison_pairs, out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png")
    comparison_media = build_dynamic_comparison_mp4(comparison_pairs, out_dir, fps=fps)
    write_json(out_dir / "comparison_media_report.json", comparison_media)
    if visual_review_path is None:
        visual_review_path = out_dir / "visual_review.json"
    expected_review_evidence = build_visual_review_evidence(
        mesh_reference_manifest_path=mesh_reference_manifest_path,
        mesh_asset=mesh_asset,
        contract=contract,
        comparison_sheet=out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png",
        comparison_mp4=out_dir / "mimickit_vs_evih_dynamic.mp4",
        source_rgb=source_rgb,
        evih_rgb=evih_rgb,
    )
    ensure_visual_review_template(visual_review_path, expected_review_evidence)
    visual_review = validate_visual_review(visual_review_path, expected_review_evidence)
    write_json(out_dir / "visual_metric_report.json", metric_report)
    write_json(out_dir / "scene_visual_metric_report.json", ground_metric_report)
    write_json(out_dir / "rgb_metric_report.json", rgb_metric_report)
    write_json(out_dir / "dynamic_sequence_report.json", dynamic_sequence_report)
    write_json(out_dir / "visual_review_report.json", visual_review)
    markdown_report = write_comparison_markdown(
        out_dir / "comparison_sheet.md",
        pairs=comparison_pairs,
        comparison_sheet=out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png",
        visual_metric_report=metric_report,
        ground_metric_report=ground_metric_report,
        rgb_metric_report=rgb_metric_report,
        visual_review=visual_review,
        dynamic_sequence_report=dynamic_sequence_report,
        comparison_mp4=out_dir / "mimickit_vs_evih_dynamic.mp4",
    )
    base["reports"].update(
        visual_metric=str(out_dir / "visual_metric_report.json"),
        scene_visual_metric=str(out_dir / "scene_visual_metric_report.json"),
        rgb_metric=str(out_dir / "rgb_metric_report.json"),
        dynamic_sequence=str(out_dir / "dynamic_sequence_report.json"),
        comparison_media=str(out_dir / "comparison_media_report.json"),
        visual_review=str(out_dir / "visual_review_report.json"),
        comparison_sheet=str(out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png"),
        comparison_mp4=str(out_dir / "mimickit_vs_evih_dynamic.mp4"),
        comparison_markdown=str(out_dir / "comparison_sheet.md"),
    )
    base.update(
        row_validation=row_validation,
        render=render_report,
        media=media,
        rgb_png_count=int(render_report.get("rgb_png_count", 0)),
        silhouette_png_count=int(render_report.get("silhouette_png_count", 0)),
        ground_mask_png_count=int(render_report.get("ground_mask_png_count", 0)),
        mp4_file=str(out_dir / "mesh_replay.mp4"),
        mp4_ok=bool(media.get("ok")),
        media_ok=bool(
            media.get("ok")
            and render_report.get("rgb_png_count") == len(frame_ids)
            and render_report.get("silhouette_png_count") == len(frame_ids)
            and render_report.get("ground_mask_png_count") == len(frame_ids)
        ),
        fk_compare_pass=bool(render_report.get("fk_compare_pass")),
        scene_visual_metric_pass=bool(ground_metric_report.get("scene_visual_metric_pass")),
        visual_metric_pass=bool(metric_report.get("visual_metric_pass")),
        dynamic_sequence_pass=bool(dynamic_sequence_report.get("dynamic_sequence_pass")),
        visual_review_pass=bool(visual_review.get("visual_review_pass")),
        comparison_sheet_pass=bool(sheet_report.get("ok")),
        comparison_mp4_pass=bool(comparison_media.get("ok")),
        comparison_markdown_pass=bool(markdown_report.get("ok")),
        visual_review_evidence=expected_review_evidence,
    )
    software_geometry_blocker = first_blocker(
        [
            (bool(row_validation.get("ok")), "replay_sidecar_validation_failed"),
            (bool(render_report.get("fk_compare_pass")), "fk_compare_failed"),
            (bool(render_report.get("dof_pose_applied")), "mesh_dof_pose_not_applied"),
            (bool(render_report.get("render_pass")), "evih_mesh_render_failed"),
            (bool(media.get("ok")), str(media.get("blocker") or "evih_mesh_mp4_invalid")),
            (bool(base.get("media_ok")), "evih_mesh_media_incomplete"),
            (bool(dynamic_sequence_report.get("dynamic_sequence_pass")), str(dynamic_sequence_report.get("blocker") or "evih_dynamic_sequence_failed")),
            (bool(ground_metric_report.get("scene_visual_metric_pass")), str(ground_metric_report.get("blocker") or "ground_scene_visual_metric_failed")),
            (bool(metric_report.get("visual_metric_pass")), str(metric_report.get("blocker") or "silhouette_visual_metric_failed")),
            (bool(sheet_report.get("ok")), str(sheet_report.get("blocker") or "comparison_sheet_failed")),
            (bool(comparison_media.get("ok")), str(comparison_media.get("blocker") or "comparison_dynamic_mp4_failed")),
        ]
    )
    base["software_geometry_replay_pass"] = not software_geometry_blocker
    base["software_geometry_blocker"] = software_geometry_blocker
    final_blocker = first_blocker(
        [
            (bool(row_validation.get("ok")), "replay_sidecar_validation_failed"),
            (bool(render_report.get("fk_compare_pass")), "fk_compare_failed"),
            (bool(render_report.get("dof_pose_applied")), "mesh_dof_pose_not_applied"),
            (bool(render_report.get("render_pass")), "evih_mesh_render_failed"),
            (bool(media.get("ok")), str(media.get("blocker") or "evih_mesh_mp4_invalid")),
            (bool(base.get("media_ok")), "evih_mesh_media_incomplete"),
            (bool(dynamic_sequence_report.get("dynamic_sequence_pass")), str(dynamic_sequence_report.get("blocker") or "evih_dynamic_sequence_failed")),
            (bool(source_rgb) and len(source_rgb) == len(frame_ids), "mimickit_rgb_frames_missing"),
            (bool(source_silhouettes) and len(source_silhouettes) == len(frame_ids), "mimickit_silhouette_frames_missing"),
            (bool(source_ground_masks) and len(source_ground_masks) == len(frame_ids), "mimickit_ground_masks_missing"),
            (bool(ground_metric_report.get("scene_visual_metric_pass")), str(ground_metric_report.get("blocker") or "ground_scene_visual_metric_failed")),
            (bool(metric_report.get("visual_metric_pass")), str(metric_report.get("blocker") or "silhouette_visual_metric_failed")),
            (bool(sheet_report.get("ok")), str(sheet_report.get("blocker") or "comparison_sheet_failed")),
            (bool(comparison_media.get("ok")), str(comparison_media.get("blocker") or "comparison_dynamic_mp4_failed")),
            (bool(markdown_report.get("ok")), str(markdown_report.get("blocker") or "comparison_markdown_failed")),
            (bool(visual_review.get("visual_review_pass")), str(visual_review.get("blocker") or "visual_review_missing_or_failed")),
        ]
    )
    base["blocker"] = final_blocker
    base["evih_mesh_replay_pass"] = not final_blocker
    write_json(out_dir / "visual_asset_manifest.json", base)
    write_json(out_dir / "mesh_replay_meta.json", base)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay MimicKit sidecars in EvihAnimation.")
    parser.add_argument("--package-dir", required=True, type=Path, help="MimicKit visual replay package.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output replay directory.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--character-geom", action="store_true", help="Render the preserved mesh_scope=false geom-style replay.")
    parser.add_argument("--true-mesh", action="store_true", help="Enable strict GLB true-mesh replay.")
    parser.add_argument("--mesh-asset", type=Path, help="Exact MimicKit-exported GLB/GLTF asset.")
    parser.add_argument("--mesh-reference-manifest", type=Path, help="Passing MimicKit mesh_reference_manifest.json.")
    parser.add_argument("--scene-contract", type=Path, help="Explicit strict scene_contract_v3.json for true-mesh replay.")
    parser.add_argument("--visual-review", type=Path, help="Required human visual_review.json for final true-mesh pass.")
    parser.add_argument("--mean-silhouette-iou-min", type=float, default=0.90)
    parser.add_argument("--p10-silhouette-iou-min", type=float, default=0.80)
    parser.add_argument("--mean-centroid-error-max", type=float, default=0.02)
    parser.add_argument("--p95-centroid-error-max", type=float, default=0.04)
    parser.add_argument("--bbox-area-ratio-p10-min", type=float, default=0.85)
    parser.add_argument("--bbox-area-ratio-p90-max", type=float, default=1.15)
    return parser


def thresholds_from_args(args: argparse.Namespace) -> JSON:
    return {
        "mean_silhouette_iou_min": args.mean_silhouette_iou_min,
        "p10_silhouette_iou_min": args.p10_silhouette_iou_min,
        "mean_centroid_error_max": args.mean_centroid_error_max,
        "p95_centroid_error_max": args.p95_centroid_error_max,
        "bbox_area_ratio_p10_min": args.bbox_area_ratio_p10_min,
        "bbox_area_ratio_p90_max": args.bbox_area_ratio_p90_max,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_dir = args.package_dir.resolve()
    out_dir = args.out_dir.resolve()
    if args.true_mesh:
        if not args.mesh_asset or not args.mesh_reference_manifest:
            raise SystemExit("--true-mesh requires --mesh-asset and --mesh-reference-manifest")
        report = build_true_mesh_replay(
            package_dir=package_dir,
            mesh_asset=args.mesh_asset.resolve(),
            mesh_reference_manifest_path=args.mesh_reference_manifest.resolve(),
            out_dir=out_dir,
            scene_contract_path=args.scene_contract.resolve() if args.scene_contract else None,
            visual_review_path=args.visual_review.resolve() if args.visual_review else None,
            thresholds=thresholds_from_args(args),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("evih_mesh_replay_pass") else 4
    package_report = validate_package_files(package_dir)
    files = resolve_package_files(package_dir)
    for required in ("pose_dof_replay", "joint_order"):
        if not files[required].is_file():
            print(json.dumps(package_report, indent=2, ensure_ascii=False))
            return 3
    rows = load_replay_rows(files["pose_dof_replay"])
    joint_order = read_json(files["joint_order"])
    report = render_skeleton_or_geom(
        rows,
        joint_order,
        out_dir,
        frames=args.frames,
        stride=args.stride,
        fps=args.fps,
        width=args.width,
        height=args.height,
        geom=args.character_geom,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("skeleton_replay_pass") or report.get("character_geom_replay_pass") else 4


if __name__ == "__main__":
    raise SystemExit(main())
