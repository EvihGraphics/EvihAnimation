import raylib as rl
from pyray import Vector3, Camera3D, CAMERA_PERSPECTIVE, Color

def main():
    rl.SetConfigFlags(rl.FLAG_WINDOW_HIDDEN)
    rl.InitWindow(800, 600, b"Test GLB")
    rl.SetTargetFPS(60)

    cam = Camera3D()
    cam.position = Vector3(1.0, 1.0, 1.0)
    cam.target = Vector3(0.0, 0.0, 0.0)
    cam.up = Vector3(0.0, 1.0, 0.0)
    cam.fovy = 45.0
    cam.projection = CAMERA_PERSPECTIVE

    model = rl.LoadModel(b"meshes/torso_link.glb")
    
    rl.rlDisableBackfaceCulling()

    rl.BeginDrawing()
    rl.ClearBackground(Color(200, 200, 200, 255))
    rl.BeginMode3D(cam)
    
    rl.DrawModel(model, Vector3(0,0,0), 1.0, Color(255, 0, 0, 255))
    rl.DrawGrid(10, 1.0)
    
    rl.EndMode3D()
    rl.EndDrawing()
    
    rl.TakeScreenshot(b"glb_test_render.png")

    rl.UnloadModel(model)
    rl.CloseWindow()

if __name__ == "__main__":
    main()
