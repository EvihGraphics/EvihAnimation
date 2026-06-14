# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys

import raylib as rl
from ai4animation import Utility
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Rotation, Vector3

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


class Standalone:
    def __init__(self, capture=False, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, hidden=False):
        self.CaptureMode = bool(capture)
        AI4Animation.Standalone = self
        AI4Animation.Draw = Utility.LoadModule(os.path.dirname(__file__) + "/Draw.py")
        AI4Animation.GUI = Utility.LoadModule(os.path.dirname(__file__) + "/GUI.py")
        AI4Animation.Color = self.Color
        flags = 0 if self.CaptureMode else rl.FLAG_MSAA_4X_HINT | rl.FLAG_WINDOW_RESIZABLE
        if hidden:
            flags |= rl.FLAG_WINDOW_HIDDEN
        if flags:
            rl.SetConfigFlags(flags)
        if not rl.IsWindowReady():
            rl.InitWindow(int(width), int(height), Utility.ToBytes("AI4AnimationPy"))
        self.Camera = AI4Animation.Scene.AddEntity("Camera").AddComponent(
            self.LoadModule("Camera").Camera
        )
        self.RenderPipeline = AI4Animation.Scene.AddEntity(
            "RenderPipeline"
        ).AddComponent(
            self.LoadModule("RenderPipeline").RenderPipeline, self.Camera.Camera
        )

        self.Primitives = self.LoadModule("Primitive")

        self.Ground = AI4Animation.Scene.AddEntity("Ground").AddComponent(
            self.LoadModule("Grid").Grid, 25, 50 if self.CaptureMode else 10, self.RenderPipeline, None, None
        )

        if not self.CaptureMode:
            self.Wall1 = AI4Animation.Scene.AddEntity(
                "Wall1", Vector3.Create(0.0, 12.5, -12.5), Rotation.Euler(90.0, 0.0, 0.0)
            ).AddComponent(self.LoadModule("Grid").Grid, 25, 10, self.RenderPipeline)
            self.Wall2 = AI4Animation.Scene.AddEntity(
                "Wall2", Vector3.Create(-12.5, 12.5, 0.0), Rotation.Euler(90.0, 90.0, 0.0)
            ).AddComponent(self.LoadModule("Grid").Grid, 25, 10, self.RenderPipeline)
            self.Wall3 = AI4Animation.Scene.AddEntity(
                "Wall3", Vector3.Create(12.5, 12.5, 0.0), Rotation.Euler(90.0, -90.0, 0.0)
            ).AddComponent(self.LoadModule("Grid").Grid, 25, 10, self.RenderPipeline)
            self.Wall4 = AI4Animation.Scene.AddEntity(
                "Wall4", Vector3.Create(0.0, 12.5, 12.5), Rotation.Euler(90.0, 180.0, 0.0)
            ).AddComponent(self.LoadModule("Grid").Grid, 25, 10, self.RenderPipeline)

        self.VideoRecorder = AI4Animation.Scene.AddEntity(
            "Screen Recorder"
        ).AddComponent(self.LoadModule("VideoRecorder").VideoRecorder)

        self.IO = self.LoadModule("InputSystem")

    def WindowPosition(self):
        pos = rl.GetWindowPosition()
        return [int(pos.x), int(pos.y)]

    def ScreenWidth(self):
        return int(rl.GetScreenWidth())

    def ScreenHeight(self):
        return int(rl.GetScreenHeight())

    def ScaleRatio(self):
        ratio_x = self.ScreenWidth() / DEFAULT_WIDTH
        ratio_y = self.ScreenHeight() / DEFAULT_HEIGHT
        ratio = (ratio_x + ratio_y) / 2
        return ratio

    def ToScreen(self, coordinates):
        return (
            int(coordinates[0] * self.ScreenWidth()),
            int(coordinates[1] * self.ScreenHeight()),
        )

    def Run(self):
        while not rl.WindowShouldClose():
            AI4Animation.Update(rl.GetFrameTime())
        self.Exit()

    def Exit(self):
        AI4Animation.Running = False
        self.VideoRecorder.StopRecording()
        self.RenderPipeline.UnloadAll()
        rl.CloseWindow()
        sys.exit()

    def Update(self):
        AI4Animation.__UPDATE__()
        self.Render()

    def Render(self, model_names=None):
        registered = self.RenderPipeline.RegisteredModels
        if model_names is not None:
            allowed = set(model_names)
            self.RenderPipeline.RegisteredModels = [
                model for model in registered if model.name in allowed
            ]
        rl.rlDisableColorBlend()
        rl.BeginDrawing()
        rl.ClearBackground(rl.BLACK)
        self.RenderPipeline.Render(
            (lambda: None) if self.CaptureMode else (lambda: AI4Animation.__DRAW__())
        )
        if not self.CaptureMode:
            rl.rlEnableColorBlend()
            AI4Animation.Draw.Text(
                "FPS " + str(rl.GetFPS()), 0.02, 0.97, 0.02, self.Color.BLACK
            )
            AI4Animation.Draw.Text(
                "Entities: " + str(len(AI4Animation.Scene.Entities)),
                0.02,
                0.95,
                0.02,
                self.Color.BLACK,
            )
            AI4Animation.__GUI__()
        rl.EndDrawing()
        self.RenderPipeline.RegisteredModels = registered

    def CaptureFrame(self, filename, model_names=None):
        self.Render(model_names=model_names)
        path = os.path.relpath(os.path.abspath(str(filename)), os.getcwd())
        rl.TakeScreenshot(Utility.ToBytes(path))

    def CaptureSemanticFrame(self, filename, character_model_names, occluder_model_names):
        rl.rlDisableColorBlend()
        rl.BeginDrawing()
        self.RenderPipeline.RenderSemantic(character_model_names, occluder_model_names)
        rl.EndDrawing()
        path = os.path.relpath(os.path.abspath(str(filename)), os.getcwd())
        rl.TakeScreenshot(Utility.ToBytes(path))

    def SetFramerate(self, fps):  # 0 is unlimited, fps otherwise
        rl.SetTargetFPS(fps)

    def LoadModule(self, name):
        return Utility.LoadModule(os.path.dirname(__file__) + "/" + name + ".py")

    def CreateSkinnedMesh(self, actor, model):
        return self.LoadModule("SkinnedMesh").SkinnedMesh(actor, model)

    def CreateRigidNodeMesh(self, actor, model):
        return self.LoadModule("RigidNodeMesh").RigidNodeMesh(actor, model)

    class Color:
        BLACK = rl.colors.BLACK
        WHITE = rl.colors.WHITE
        RED = rl.colors.RED
        GREEN = rl.colors.GREEN
        BLUE = rl.colors.BLUE
        RAYWHITE = rl.colors.RAYWHITE
        GRAY = rl.colors.GRAY
        LIGHTGRAY = rl.colors.LIGHTGRAY
        SKYBLUE = rl.colors.SKYBLUE
        ORANGE = rl.colors.ORANGE
        MAGENTA = rl.colors.MAGENTA
