# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Render non-skinned GLB mesh nodes through the framework RenderPipeline."""

from array import array
from io import BytesIO

import cffi
import numpy as np
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Transform, Vector3
from pyray import Mesh, load_model_from_mesh
from raylib import (
    LoadImageFromMemory,
    LoadTextureFromImage,
    MATERIAL_MAP_DIFFUSE,
    RAYWHITE,
    SetMaterialTexture,
    UnloadImage,
    UpdateMeshBuffer,
    UploadMesh,
    WHITE,
)

ffi = cffi.FFI()


def _basis_vector(values):
    values = np.asarray(values, dtype=np.float32)
    return np.stack((values[..., 0], values[..., 2], -values[..., 1]), axis=-1)


def _vertex_normals(vertices, triangles, normals):
    vertices = np.asarray(vertices, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape == vertices.shape and np.any(np.linalg.norm(normals, axis=1) > 1e-8):
        return normals

    generated = np.zeros_like(vertices)
    for triangle in np.asarray(triangles, dtype=np.int64).reshape(-1, 3):
        a, b, c = vertices[triangle]
        face_normal = np.cross(b - a, c - a)
        generated[triangle] += face_normal
    lengths = np.linalg.norm(generated, axis=1)
    valid = lengths > 1e-8
    generated[valid] /= lengths[valid, np.newaxis]
    generated[~valid] = (0.0, 0.0, 1.0)
    return generated


def _create_texture(image):
    if image is None:
        return None
    encoded = BytesIO()
    image.convert("RGBA").save(encoded, format="PNG")
    payload = encoded.getvalue()
    raylib_image = LoadImageFromMemory(
        b".png",
        ffi.from_buffer("unsigned char[]", payload),
        len(payload),
    )
    texture = LoadTextureFromImage(raylib_image)
    UnloadImage(raylib_image)
    return texture


def _create_model(mesh):
    vertices = _basis_vector(mesh.Vertices)
    normals = _basis_vector(_vertex_normals(mesh.Vertices, mesh.Triangles, mesh.Normals))
    vertex_count = len(vertices)
    vertex_data = array("f", vertices.flatten())
    normal_data = array("f", normals.flatten())
    triangle_data = array("H", np.asarray(mesh.Triangles, dtype=np.uint16).flatten())
    texcoords = np.asarray(mesh.TexCoords, dtype=np.float32)
    if texcoords.shape != (vertex_count, 2):
        texcoords = np.full((vertex_count, 2), 0.5, dtype=np.float32)
    texcoord_data = array("f", texcoords.flatten())

    raylib_mesh = Mesh()
    raylib_mesh.vertexCount = vertex_count
    raylib_mesh.triangleCount = int(len(triangle_data) / 3)
    raylib_mesh.vertices = ffi.cast("float*", vertex_data.buffer_info()[0])
    raylib_mesh.normals = ffi.cast("float*", normal_data.buffer_info()[0])
    raylib_mesh.texcoords = ffi.cast("float*", texcoord_data.buffer_info()[0])
    raylib_mesh.indices = ffi.cast("unsigned short*", triangle_data.buffer_info()[0])
    raylib_mesh.vaoId = 0
    UploadMesh(ffi.addressof(raylib_mesh), True)

    model = load_model_from_mesh(raylib_mesh)
    model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE
    texture = _create_texture(getattr(mesh, "Image", None))
    if texture is not None:
        SetMaterialTexture(
            ffi.addressof(model.materials[0]),
            MATERIAL_MAP_DIFFUSE,
            texture,
        )
    return model, texture, vertices, normals


class RigidNodeMesh:
    def __init__(self, actor, model):
        self.Actor = actor
        self.Model = model
        self.Models = []
        self.Registered = []
        self.Geometry = []
        self.Textures = []
        primitives = iter(model.Meshes)
        mesh_primitives = {}
        for mesh_index, mesh in enumerate(model._glb.meshes):
            mesh_primitives[mesh_index] = [next(primitives) for _ in mesh.primitives]

        for node_index, node in enumerate(model._glb.nodes):
            if node.mesh is None:
                continue
            node_name = node.name or f"node_{node_index}"
            entity = actor.NameToEntity.get(node_name)
            if entity is None:
                continue
            for primitive_index, mesh in enumerate(mesh_primitives.get(node.mesh, [])):
                raylib_model, texture, vertices, normals = _create_model(mesh)
                registered = AI4Animation.Standalone.RenderPipeline.RegisterModel(
                    name=f"{actor.Entity.Name}/{node_name}/{primitive_index}",
                    model=raylib_model,
                    skinned_mesh=None,
                    color=RAYWHITE,
                )
                self.Models.append(raylib_model)
                self.Registered.append((registered, entity))
                self.Geometry.append((registered, entity, vertices, normals))
                if texture is not None:
                    self.Textures.append(texture)
        self.Update()

    def SetColor(self, color):
        for registered, _ in self.Registered:
            registered.color = color

    def Register(self):
        pass

    def Unregister(self):
        pass

    def Update(self):
        for registered, entity, vertices, normals in self.Geometry:
            transform = entity.GetTransform()
            position = Transform.GetPosition(transform)
            rotation = Transform.GetRotation(transform)
            scale = entity.GetScale()
            world_vertices = (rotation @ (vertices * scale).T).T + position
            world_normals = (rotation @ normals.T).T
            lengths = np.linalg.norm(world_normals, axis=1)
            valid = lengths > 1e-8
            world_normals[valid] /= lengths[valid, np.newaxis]
            world_normals[~valid] = (0.0, 1.0, 0.0)
            vertex_data = array("f", np.asarray(world_vertices, dtype=np.float32).flatten())
            normal_data = array("f", np.asarray(world_normals, dtype=np.float32).flatten())
            UpdateMeshBuffer(
                registered.model.meshes[0],
                0,
                ffi.cast("void*", vertex_data.buffer_info()[0]),
                len(vertex_data) * vertex_data.itemsize,
                0,
            )
            UpdateMeshBuffer(
                registered.model.meshes[0],
                2,
                ffi.cast("void*", normal_data.buffer_info()[0]),
                len(normal_data) * normal_data.itemsize,
                0,
            )
            registered.position = Vector3.ToRayLib(Vector3.Zero())
            registered.rotationAxis = Vector3.ToRayLib(Vector3.UnitY())
            registered.rotationAngle = 0.0
            registered.scale = Vector3.ToRayLib(Vector3.One())
