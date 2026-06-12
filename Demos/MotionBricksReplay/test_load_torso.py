import raylib as rl
rl.InitWindow(800, 600, b"Test")
m = rl.LoadModel(b"meshes/torso_link.obj")
print(f"Vertex count: {m.meshes[0].vertexCount}")
print(f"Triangle count: {m.meshes[0].triangleCount}")
rl.CloseWindow()
