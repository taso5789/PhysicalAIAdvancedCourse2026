"""Record only simulator frames and explicitly supplied task text."""
import os
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"red": "赤", "green": "緑", "blue": "青"}
ACTIONS = {"search": "探索", "approach": "接近", "stop": "停止"}


def action_label(action):
    return ACTIONS[action["action"]] + ("：" + COLORS[action["color"]] if "color" in action else "")


def wrap_text(text, font, width):
    lines, line = [], ""
    for char in text:
        if char == "\n" or (line and font.getlength(line + char) > width):
            lines.append(line)
            line = "" if char == "\n" else char
        else:
            line += char
    if line:
        lines.append(line)
    return lines


class VideoRecorder:
    fps = 10

    def __init__(self, sim, path, instruction, plan, provider):
        self.sim = sim
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("--record の保存先は .mp4 にしてください")
        if self.path.exists():
            raise ValueError("録画先が存在します。別のファイル名を指定してください")
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "meiryo.ttc"
        if not font_path.is_file():
            raise ValueError("日本語表示用の Meiryo フォントが見つかりません")
        self.font = ImageFont.truetype(str(font_path), 23)
        self.small = ImageFont.truetype(str(font_path), 19)
        self.instruction = instruction
        self.plan = plan
        self.provider = provider
        self.index = None
        self.label = "準備"
        self.frames = 0
        self.writer = None
        sim.model.vis.global_.offwidth = max(800, sim.model.vis.global_.offwidth)
        sim.model.vis.global_.offheight = max(540, sim.model.vis.global_.offheight)
        self.renderer = mujoco.Renderer(sim.model, height=540, width=800)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.5, 0, 0]
        self.camera.distance = 8
        self.camera.azimuth = 135
        self.camera.elevation = -55
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.writer = imageio_ffmpeg.write_frames(str(self.path), (1280, 720), fps=self.fps,
                                                     codec="libx264", pix_fmt_out="yuv420p",
                                                     output_params=["-movflags", "+faststart"])
            self.writer.send(None)
        except BaseException:
            self.renderer.close()
            raise

    def set_step(self, index, label):
        self.index, self.label = index, label

    def capture(self, state=""):
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        scene = self.renderer.render().copy()
        self.renderer.update_scene(self.sim.data, camera="front")
        front = self.renderer.render().copy()
        canvas = Image.new("RGB", (1280, 720), (20, 25, 34))
        canvas.paste(Image.fromarray(scene), (0, 0))
        canvas.paste(Image.fromarray(front).resize((320, 216)), (960, 504))
        draw = ImageDraw.Draw(canvas)
        draw.text((820, 12), "日本語指示 × 移動ロボット", font=self.font, fill="white")
        provider_label = "Claude の計画" if self.provider == "llm" else "MOCK：AI未使用の確認用計画"
        draw.text((820, 54), provider_label, font=self.small, fill="#6ed4ff")
        for index, action in enumerate(self.plan["actions"]):
            selected = index == self.index
            draw.text((820, 92 + index * 39), f"{'▶' if selected else ' '} {index + 1}. {action_label(action)}",
                      font=self.font, fill="#ffd166" if selected else "white")
        draw.text((20, 550), "指示：", font=self.font, fill="#6ed4ff")
        lines = wrap_text(self.instruction, self.font, 740)
        for index, line in enumerate(lines[:3]):
            if index == 2 and len(lines) > 3:
                line = line[:-1] + "…"
            draw.text((20, 587 + index * 31), line, font=self.font, fill="white")
        draw.text((820, 424), self.label, font=self.font, fill="#ffd166")
        draw.text((820, 460), f"{state}  |  sim {self.sim.data.time:.1f} s", font=self.small, fill="white")
        draw.text((820, 550), "前方カメラ →", font=self.small, fill="white")
        draw.text((820, 600), "通常速度", font=self.small, fill="white")
        pixels = np.asarray(canvas)
        self.writer.send(pixels)
        if self.frames == 0:
            canvas.save(self.path.with_suffix(".preview.png"))
        self.frames += 1

    def card(self, label, seconds=3):
        self.label = label
        for _ in range(round(seconds * self.fps)):
            self.capture("説明カード（静止画）")

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        finally:
            self.renderer.close()

    @property
    def duration(self):
        return self.frames / self.fps
