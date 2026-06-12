import raylib as rl
from pyray import Vector3, Camera3D, CAMERA_PERSPECTIVE, Color

def main():
    rl.InitWindow(800, 600, b"Test Torso")
    rl.SetTargetFPS(60)

    cam = Camera3D()
    cam.position = Vector3(1.0, 1.0, 1.0)
    cam.target = Vector3(0.0, 0.0, 0.0)
    cam.up = Vector3(0.0, 1.0, 0.0)
    cam.fovy = 45.0
    cam.projection = CAMERA_PERSPECTIVE

    model = rl.LoadModel(b"meshes/torso_link.obj")
    
    # We must explicitly disable face culling just in case
    rl.rlDisableBackfaceCulling()

    while not rl.WindowShouldClose():
        rl.BeginDrawing()
        rl.ClearBackground(Color(200, 200, 200, 255))
        rl.BeginMode3D(cam)
        
        rl.DrawModel(model, Vector3(0,0,0), 1.0, Color(255, 0, 0, 255))
        rl.DrawGrid(10, 1.0)
        
        rl.EndMode3D()
        
        # take screenshot on first frame
        if rl.GetTime() > 0.1:
            rl.TakeScreenshot(b"torso_test_render.png")
            break
            
        rl.EndDrawing()

    rl.UnloadModel(model)
    rl.CloseWindow()

if __name__ == "__main__":
    main()
