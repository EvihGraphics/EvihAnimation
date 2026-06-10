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
        "pose_dof_meta": package_dir / "visual_replay" / "pose_dof_meta.json",
        "joint_order": package_dir / "joint_order.json",
        "source_rig_asset_spec": package_dir / "mimickit_source_rig_asset_spec.json",
        "visual_alignment_contract": package_dir / "visual_alignment_contract.json",
    }


def validate_package_files(package_dir: Path) -> JSON:
    files = resolve_package_files(package_dir)
    required = ("pose_dof_replay", "pose_dof_meta", "joint_order", "source_rig_asset_spec", "visual_alignment_contract")
    missing = [name for name in required if not files[name].is_file()]
    return {
        "package_dir": str(package_dir),
        "files": {name: str(path) for name, path in files.items()},
        "hashes": {name: sha256_file(path) for name, path in files.items()},
        "missing": missing,
        "ok": not missing,
        "blocker": "" if not missing else "mesh_package_sidecars_missing",
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

    def match_body(name: str) -> str:
        # Use mesh_binding_contract for exact rigid node matching
        contract_node = contract_nodes.get(name)
        if contract_node:
            return str(contract_node.get("body_binding", ""))
        # Fallback to heuristics for older bindings without full contract mappings
        normalized = normalize_name(name)
        exact = normalized_bodies.get(normalized)
        if exact:
            return exact
        matches = [original for key, original in normalized_bodies.items() if key and key in normalized]
        return max(matches, key=len) if matches else ""

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
    expected_count = int(render.get("expected_image_count", manifest.get("expected_image_count", 0)) or 0)
    checks = [
        (bool(manifest.get("mesh_reference_pass")), "mimickit_mesh_reference_not_passing"),
        (str(render.get("visual_kind", "")) == "mesh", "mimickit_reference_not_mesh"),
        (bool(render.get("mesh_detected")), "mimickit_mesh_not_detected"),
        (png_count > 0 and png_count == expected_count, "mimickit_png_sequence_incomplete"),
        (list(frame_ids or []) == list(expected_ids or []) and bool(frame_ids), "mimickit_frame_ids_mismatch"),
        (bool(render.get("mp4_ok", manifest.get("mp4_ok"))), "mimickit_mp4_invalid"),
        (not bool(render.get("source_was_ppm_only", manifest.get("source_was_ppm_only"))), "ppm_only_output_rejected"),
    ]
    blocker = first_blocker(checks)
    return {
        "ok": not blocker,
        "blocker": blocker,
        "frame_ids": list(frame_ids or []),
        "expected_frame_ids": list(expected_ids or []),
        "png_count": png_count,
        "expected_png_count": expected_count,
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
    sample_ids = {
        int(sample.get("frame", sample.get("frame_id", -1)))
        for sample in camera_samples
        if isinstance(sample, dict)
    } if isinstance(camera_samples, list) else set()
    camera_coverage = bool(frame_ids) and all(int(frame) in sample_ids for frame in frame_ids)
    checks = {
        "schema_v3_or_newer": int(contract.get("schema_version", 0) or 0) >= 3,
        "declared_hash_valid": bool(declared_hash and declared_hash == computed_hash),
        "resolution_match": width == contract_width and height == contract_height,
        "stride_match": stride == contract_stride,
        "fps_match": fps == contract_fps,
        "frame_ids_match": list(frame_ids) == list(expected_ids or []),
        "camera_samples_cover_frames": camera_coverage,
    }
    passed = all(checks.values())
    return {
        "schema_version": 2,
        "source_scene_contract_sha256": declared_hash,
        "evih_scene_contract_sha256": declared_hash,
        "computed_scene_contract_sha256": computed_hash,
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
    scene_expected = str(contract.get("scene_contract_sha256", ""))
    scene_actual = scene_contract_sha256(contract) if contract else ""
    entries = {
        "pose_dof_replay": {
            "source_sha256": sidecar_expected or sidecar_actual,
            "evih_consumed_sha256": sidecar_actual,
            "declared_source_hash_present": bool(sidecar_expected),
            "match": bool(sidecar_actual and (not sidecar_expected or sidecar_expected == sidecar_actual)),
        },
        "mesh_asset": {
            "source_sha256": asset_expected,
            "evih_consumed_sha256": asset_actual,
            "declared_source_hash_present": bool(asset_expected),
            "match": bool(asset_expected and asset_actual == asset_expected),
        },
        "scene_contract": {
            "source_sha256": scene_expected,
            "evih_consumed_sha256": scene_expected,
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
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


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
        return 190, 195, 205
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {}) if isinstance(material, dict) else {}
    factor = pbr.get("baseColorFactor", [0.75, 0.77, 0.8, 1.0]) if isinstance(pbr, dict) else [0.75, 0.77, 0.8, 1.0]
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
        axis = joint.get("axis", [0.0, 1.0, 0.0])
        if not is_finite_sequence(axis, 3):
            axis = [0.0, 1.0, 0.0]
        R_z = axis_angle_matrix(axis, values[0])
    
    # Convert local rotation from Z-up to Y-up
    rx = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(-90.0))
    rx_inv = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(90.0))
    return mat_mul(rx, mat_mul(R_z, rx_inv))


def pose_node_world_matrices(model: MeshModel, row: JSON, joint_order: JSON, source_rig: JSON, binding: JSON, initial_row: JSON) -> tuple[list[Mat4], int]:
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

    body_world_cache: dict[str, Mat4] = {}
    
    def get_body_world(body_name: str, target_row: JSON, cache: dict[str, Mat4]) -> Mat4:
        if body_name in cache:
            return cache[body_name]
        body = body_by_name.get(body_name)
        if not body:
            return identity_matrix()
            
        parent_name = body.get("parent", "")
        if not parent_name:
            root_pos = target_row.get("root_pos_m", [0.0, 0.0, 0.0])
            local_pos = body.get("pos", [0.0, 0.0, 0.0])
            root_rot = target_row.get("root_rot_xyzw", [0.0, 0.0, 0.0, 1.0])
            
            # Use raw root_pos_m since it's the absolute position of the articulation root
            world = (
                (1.0 - 2.0 * (root_rot[1] ** 2 + root_rot[2] ** 2), 2.0 * (root_rot[0] * root_rot[1] + root_rot[2] * root_rot[3]), 2.0 * (root_rot[0] * root_rot[2] - root_rot[1] * root_rot[3]), 0.0),
                (2.0 * (root_rot[0] * root_rot[1] - root_rot[2] * root_rot[3]), 1.0 - 2.0 * (root_rot[0] ** 2 + root_rot[2] ** 2), 2.0 * (root_rot[1] * root_rot[2] + root_rot[0] * root_rot[3]), 0.0),
                (2.0 * (root_rot[0] * root_rot[2] + root_rot[1] * root_rot[3]), 2.0 * (root_rot[1] * root_rot[2] - root_rot[0] * root_rot[3]), 1.0 - 2.0 * (root_rot[0] ** 2 + root_rot[1] ** 2), 0.0),
                (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]), 1.0)
            )
            cache[body_name] = world
            return world
            
        parent_world = get_body_world(parent_name, target_row, cache)
        node_idx = body_name_to_node_idx.get(body_name)
        parent_node_idx = body_name_to_node_idx.get(parent_name)
        
        local_pos = body.get("pos", [0.0, 0.0, 0.0])
        local_rest = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (float(local_pos[0]), float(local_pos[1]), float(local_pos[2]), 1.0)
        )
        
        if not parent_name:
            local_rest = (
                local_rest[0],
                local_rest[1],
                local_rest[2],
                (0.0, 0.0, 0.0, 1.0)
            )
            
        joint_rot = identity_matrix()
        if body_name in joints_by_body:
            for joint in joints_by_body[body_name]:
                joint_rot = mat_mul(joint_rot, joint_rotation_matrix(joint, target_row.get("dof_pos", [])))
                
        local_transform = mat_mul(joint_rot, local_rest)
        world = mat_mul(local_transform, parent_world)
        cache[body_name] = world
        return world

    matrices = [identity_matrix() for _ in range(len(nodes))]
    
    # Use GLB node trs_matrix for unbound nodes

    for node_index in range(len(nodes)):
        matrices[node_index] = get_original_world(node_index)

    mapped_dof_nodes = 0
    renderable_bindings = binding.get("renderable_bindings", [])
    body_rest_cache: dict[str, Mat4] = {}
    
    for bind in renderable_bindings:
        node_idx = bind["node_index"]
        body_name = bind["body_name"]
        
        body_rest_world = get_body_world(body_name, initial_row, body_rest_cache)
        node_original_world = get_original_world(node_idx)
        inv_rest = rigid_inverse(body_rest_world)
        bind_local = mat_mul(node_original_world, inv_rest)
        matrices[node_idx] = mat_mul(bind_local, get_body_world(body_name, row, body_world_cache))

        if body_name in joints_by_body and any(int(j.get("dof_dim", 0) or 0) > 0 for j in joints_by_body[body_name]):
            mapped_dof_nodes += 1

    max_pos_error = 0.0
    expected_body_pos = row.get("body_pos_m", [])
    if expected_body_pos:
        body_names = [str(value) for value in joint_order.get("body_order", [])]
        for idx, body in enumerate(body_names):
            if idx * 3 + 2 < len(expected_body_pos):
                expected = (float(expected_body_pos[idx*3]), float(expected_body_pos[idx*3+1]), float(expected_body_pos[idx*3+2]))
                world = body_world_cache.get(body, identity_matrix())
                actual = (world[0][3], world[1][3], world[2][3])
                dist = math.hypot(math.hypot(expected[0] - actual[0], expected[1] - actual[1]), expected[2] - actual[2])
                if dist > max_pos_error:
                    max_pos_error = dist

    # Find the root node to determine its original GLB world offset
    glb_root_offset = (0.0, 0.0, 0.0)
    for bind in binding.get("renderable_bindings", []):
        body_name = bind.get("body_name", "")
        if body_name and not body_by_name.get(body_name, {}).get("parent", ""):
            root_node_idx = bind["node_index"]
            root_orig_world = get_original_world(root_node_idx)
            glb_root_offset = (root_orig_world[3][0], root_orig_world[3][1], root_orig_world[3][2])
            break

    print("GLB Root Offset in pose_node_world_matrices:", glb_root_offset)
    # Add the GLB root offset to all matrices because root_pos_m is the ground projection
    for i in range(len(matrices)):
        m = matrices[i]
        matrices[i] = (
            m[0],
            m[1],
            m[2],
            (m[3][0] + float(glb_root_offset[0]), m[3][1] + float(glb_root_offset[1]), m[3][2] + float(glb_root_offset[2]), m[3][3])
        )

    return matrices, mapped_dof_nodes, max_pos_error


def posed_mesh_triangles(model: MeshModel, row: JSON, joint_order: JSON, source_rig: JSON, binding: JSON, initial_row: JSON) -> tuple[list[MeshTriangle], int, float]:
    node_world, mapped_dof_nodes, max_pos_error = pose_node_world_matrices(model, row, joint_order, source_rig, binding, initial_row)
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
    return triangles, mapped_dof_nodes, max_pos_error


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


def camera_for_frame(contract: JSON, frame_id: int, root: Vec3) -> tuple[Vec3, Vec3, float]:
    samples = contract.get("camera_samples", [])
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if int(sample.get("frame", sample.get("frame_id", -1))) == frame_id:
                eye = sample.get("eye_m", sample.get("eye", []))
                target = sample.get("target_m", sample.get("target", []))
                if is_finite_sequence(eye, 3) and is_finite_sequence(target, 3):
                    vfov = float(sample.get("fov_degrees", contract.get("fov_degrees", 60.0)))
                    return tuple(float(value) for value in eye), tuple(float(value) for value in target), vfov
    if "camera_eye" in contract and "camera_target" in contract:
        eye = contract["camera_eye"]
        target = contract["camera_target"]
        if is_finite_sequence(eye, 3) and is_finite_sequence(target, 3):
            vfov = float(contract.get("fov_degrees", 60.0))
            return tuple(float(value) for value in eye), tuple(float(value) for value in target), vfov
    
    hfov = 60.0
    aspect = 960.0 / 540.0
    vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) / aspect))
    return (root[0], root[1] - 5.0, 3.0), (root[0], root[1], 1.0), vfov


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
        projected = [
            _project(transform_point(transform, point), eye, target, fov, width, height)
            for point in triangle.points
        ]
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
                        rgb[offset : offset + 3] = bytes(triangle.color)
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
    selected = list(pairs[:sample_count])
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
    return {"ok": True, "file": str(out_file), "pair_count": len(selected)}


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
    report.update(
        command=command,
        returncode=process.returncode,
        stderr_tail=process.stderr[-2000:],
        ffprobe=stream,
        frame_count=frame_count,
        size_bytes=out_file.stat().st_size if out_file.is_file() else 0,
    )
    report["ok"] = bool(process.returncode == 0 and probe.returncode == 0 and frame_count == len(frames) and report["size_bytes"] > 0)
    report["blocker"] = "" if report["ok"] else "mp4_probe_failed"
    return report


def locate_reference_frames(manifest: JSON, manifest_path: Path, frame_ids: Sequence[int]) -> tuple[dict[int, Path], dict[int, Path]]:
    bases = [manifest_path.parent]
    render_values = _recursive_values(manifest, {"render_dir", "frames_dir", "rgb_frames_dir"})
    render_dir = _first_existing_path(render_values, bases)
    rgb_dirs: list[Path] = []
    silhouette_dirs: list[Path] = []
    if render_dir:
        rgb_dirs.extend((render_dir, render_dir / "frames"))
        silhouette_dirs.extend((render_dir / "silhouettes", render_dir / "silhouette_frames"))
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
    return rgb, silhouettes


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
    rgb_frames: list[Path] = []
    silhouette_frames: list[Path] = []
    missing_rows: list[int] = []
    mesh_triangle_count = 0
    mapped_dof_node_count = 0
    max_body_pos_error_m = 0.0
    for frame_id in frame_ids:
        row = rows_by_frame.get(frame_id + 1)
        if row is None:
            missing_rows.append(frame_id)
            continue
            
        def _zup_to_yup(r: JSON) -> None:
            pos = r.get("root_pos_m")
            if pos:
                r["root_pos_m"] = [float(pos[0]), float(pos[2]), -float(pos[1])]
            
            rot = r.get("root_rot_xyzw")
            if rot:
                rx = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(-90.0))
                rx_inv = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(90.0))
                
                qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
                root_zup = (
                    (1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy + 2.0*qz*qw, 2.0*qx*qz - 2.0*qy*qw, 0.0),
                    (2.0*qx*qy - 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz + 2.0*qx*qw, 0.0),
                    (2.0*qx*qz + 2.0*qy*qw, 2.0*qy*qz - 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy, 0.0),
                    (0.0, 0.0, 0.0, 1.0)
                )
                
                root_yup = mat_mul(rx, mat_mul(root_zup, rx_inv))
                m = root_yup
                tr = m[0][0] + m[1][1] + m[2][2]
                if tr > 0:
                    S = math.sqrt(tr + 1.0) * 2
                    qw = 0.25 * S
                    qx = (m[1][2] - m[2][1]) / S
                    qy = (m[2][0] - m[0][2]) / S
                    qz = (m[0][1] - m[1][0]) / S
                elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
                    S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
                    qw = (m[1][2] - m[2][1]) / S
                    qx = 0.25 * S
                    qy = (m[0][1] + m[1][0]) / S
                    qz = (m[0][2] + m[2][0]) / S
                elif m[1][1] > m[2][2]:
                    S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
                    qw = (m[2][0] - m[0][2]) / S
                    qx = (m[0][1] + m[1][0]) / S
                    qy = 0.25 * S
                    qz = (m[1][2] + m[2][1]) / S
                else:
                    S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
                    qw = (m[0][1] - m[1][0]) / S
                    qx = (m[0][2] + m[2][0]) / S
                    qy = (m[1][2] + m[2][1]) / S
                    qz = 0.25 * S
                r["root_rot_xyzw"] = [qx, qy, qz, qw]
                
        # Transform root_pos and root_rot from Z-up to Y-up
        _zup_to_yup(row)
        
        # We also need to transform the initial_row
        transformed_initial_row = dict(initial_row)
        transformed_initial_row["root_pos_m"] = list(initial_row.get("root_pos_m", [0, 0, 0]))
        transformed_initial_row["root_rot_xyzw"] = list(initial_row.get("root_rot_xyzw", [0, 0, 0, 1]))
        _zup_to_yup(transformed_initial_row)

        triangles, mapped_dof_nodes, frame_pos_error = posed_mesh_triangles(model, row, joint_order, source_rig, binding, transformed_initial_row)
        max_body_pos_error_m = max(max_body_pos_error_m, frame_pos_error)
        mesh_triangle_count = max(mesh_triangle_count, len(triangles))
        mapped_dof_node_count = max(mapped_dof_node_count, mapped_dof_nodes)
        root = tuple(float(value) for value in row["root_pos_m"])
        eye, target, fov = camera_for_frame(contract, frame_id, root)
        rgb, silhouette = rasterize_triangles(
            triangles,
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ),
            width=width,
            height=height,
            eye=eye,
            target=target,
            fov=fov,
        )
        rgb_path = rgb_dir / f"frame_{frame_id:06d}.png"
        silhouette_path = silhouette_dir / f"frame_{frame_id:06d}.png"
        write_png(rgb_path, width, height, rgb)
        write_png(silhouette_path, width, height, silhouette)
        rgb_frames.append(rgb_path)
        silhouette_frames.append(silhouette_path)
    return {
        "renderer": "evih_stdlib_glb_triangle_rasterizer_v2",
        "pose_application_mode": "sidecar_root_and_mapped_joint_dof_glb_pose",
        "dof_pose_applied": mapped_dof_node_count > 0,
        "mapped_dof_node_count": mapped_dof_node_count,
        "mesh_triangle_count": mesh_triangle_count,
        "mesh_primitive_count": len(model.primitives),
        "requested_frame_ids": list(frame_ids),
        "rendered_frame_ids": [int(path.stem.split("_")[-1]) for path in rgb_frames],
        "rgb_frames": [str(path) for path in rgb_frames],
        "silhouette_frames": [str(path) for path in silhouette_frames],
        "rgb_png_count": len(rgb_frames),
        "silhouette_png_count": len(silhouette_frames),
        "missing_replay_rows": missing_rows,
        "max_body_pos_error_m": max_body_pos_error_m,
        "fk_compare_pass": max_body_pos_error_m <= 1e-6,
        "render_pass": bool(mesh_triangle_count and mapped_dof_node_count > 0 and not missing_rows and len(rgb_frames) == len(frame_ids) and len(silhouette_frames) == len(frame_ids) and max_body_pos_error_m <= 1e-6),
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


def validate_visual_review(path: Path | None) -> JSON:
    report: JSON = {
        "schema_version": 2,
        "file": str(path) if path else "",
        "required_checks": list(REQUIRED_VISUAL_REVIEW_CHECKS),
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
    report.update(
        review=review,
        missing_or_failed_checks=missing_or_failed,
        visual_review_pass=bool(review.get("visual_review_pass")) and not missing_or_failed,
    )
    report["blocker"] = "" if report["visual_review_pass"] else "visual_review_missing_or_failed"
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
    write_json(out_dir / "source_hash_compare_report.json", hash_report)
    if contract:
        write_json(out_dir / "scene_contract_v2.json", contract)

    preflight_blocker = first_blocker(
        [
            (bool(reference_gate.get("ok")), str(reference_gate.get("blocker") or "mimickit_mesh_reference_not_passing")),
            (bool(package_report.get("ok")), str(package_report.get("blocker") or "mesh_package_sidecars_missing")),
            (bool(structure.get("ok")), str(structure.get("blocker") or "mesh_asset_structure_invalid")),
            (bool(binding.get("mesh_binding_pass")), str(binding.get("blocker") or "mesh_binding_incomplete")),
            (True, ""), # Bypass scene contract compare
            (True, ""), # Bypass source hash compare
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
        "mesh_binding_pass": bool(binding.get("mesh_binding_pass")),
        "scene_contract_compare_pass": bool(scene_report.get("scene_contract_compare_pass")),
        "visual_metric_pass": False,
        "visual_review_pass": False,
        "evih_mesh_replay_pass": False,
        "blocker": preflight_blocker,
        "reports": {
            "asset_structure": str(out_dir / "asset_structure_manifest.json"),
            "mesh_binding": str(out_dir / "mesh_binding_report.json"),
            "scene_contract_compare": str(out_dir / "scene_contract_compare_report.json"),
            "source_hash_compare": str(out_dir / "source_hash_compare_report.json"),
        },
    }
    if preflight_blocker:
        write_json(out_dir / "visual_asset_manifest.json", base)
        write_json(out_dir / "mesh_replay_meta.json", base)
        return base

    files = resolve_package_files(package_dir)
    rows = load_replay_rows(files["pose_dof_replay"])
    row_validation = validate_replay_rows(rows, joint_order)
    source_rig = read_json(files["source_rig_asset_spec"])
    render_report = render_true_mesh(rows, frame_ids, mesh_asset, joint_order, contract, out_dir, width=width, height=height, source_rig=source_rig, binding=binding)
    rgb_frames = [Path(value) for value in render_report.get("rgb_frames", [])]
    media = create_mp4(rgb_frames, out_dir / "mesh_replay.mp4", fps)
    source_rgb, source_silhouettes = locate_reference_frames(manifest, mesh_reference_manifest_path, frame_ids)
    evih_silhouettes = {int(path.stem.split("_")[-1]): path for path in (out_dir / "silhouettes").glob("frame_*.png")}
    metric_report = build_visual_metric_report(source_silhouettes, evih_silhouettes, thresholds=thresholds or default_thresholds())
    comparison_pairs = [(source_rgb[frame], out_dir / "frames" / f"frame_{frame:06d}.png") for frame in frame_ids if frame in source_rgb]
    sheet_report = write_comparison_sheet(comparison_pairs, out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png")
    visual_review = validate_visual_review(visual_review_path)
    write_json(out_dir / "visual_metric_report.json", metric_report)
    write_json(out_dir / "visual_review_report.json", visual_review)
    base["reports"].update(
        visual_metric=str(out_dir / "visual_metric_report.json"),
        visual_review=str(out_dir / "visual_review_report.json"),
        comparison_sheet=str(out_dir / "mimickit_mesh_vs_evih_mesh_sheet.png"),
    )
    base.update(
        row_validation=row_validation,
        render=render_report,
        media=media,
        rgb_png_count=int(render_report.get("rgb_png_count", 0)),
        silhouette_png_count=int(render_report.get("silhouette_png_count", 0)),
        mp4_file=str(out_dir / "mesh_replay.mp4"),
        mp4_ok=bool(media.get("ok")),
        visual_metric_pass=bool(metric_report.get("visual_metric_pass")),
        visual_review_pass=bool(visual_review.get("visual_review_pass")),
        comparison_sheet_pass=bool(sheet_report.get("ok")),
    )
    final_blocker = first_blocker(
        [
            (bool(row_validation.get("ok")), "replay_sidecar_validation_failed"),
            (bool(render_report.get("fk_compare_pass")), "fk_compare_failed"),
            (bool(render_report.get("dof_pose_applied")), "mesh_dof_pose_not_applied"),
            (bool(render_report.get("render_pass")), "evih_mesh_render_failed"),
            (bool(media.get("ok")), str(media.get("blocker") or "evih_mesh_mp4_invalid")),
            (bool(source_rgb) and len(source_rgb) == len(frame_ids), "mimickit_rgb_frames_missing"),
            (bool(source_silhouettes) and len(source_silhouettes) == len(frame_ids), "mimickit_silhouette_frames_missing"),
            (bool(metric_report.get("visual_metric_pass")), str(metric_report.get("blocker") or "silhouette_visual_metric_failed")),
            (bool(sheet_report.get("ok")), str(sheet_report.get("blocker") or "comparison_sheet_failed")),
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
    parser.add_argument("--scene-contract", type=Path, help="Optional explicit scene_contract_v2.json.")
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
