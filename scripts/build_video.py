"""Assemble rendered demos and Japanese explanation slides without any API call."""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = (1280, 720)
FPS = 10
SLIDES = [
    (10, "日本語指示で動く移動ロボット", [
        "目的：人の指示を理解し、目標を確認して移動する。",
        "Windows 上の MuJoCo で差動二輪ロボットを動かす。",
        "AI は Claude の推論を使用。追加学習は行っていない。",
    ]),
    (12, "システムの構成", [
        "日本語の指示 → Claude → 検証済みの行動計画",
        "実行器 → 探索・接近・停止 → 二輪ロボット",
        "RGB-D カメラ → 色と距離の認識 → 移動制御",
        "画像認識はルールベース。AI の役割は行動計画。",
    ]),
    (12, "工夫した点", [
        "「確認する」は探索、「前で止まる」は接近として実行。",
        "計画は search・approach・stop の許可した動作に限定。",
        "目標の正確な座標を移動制御に使わない。",
        "不正な計画や動作失敗は、残りの実行を中止する。",
    ]),
    (8, "結果と今後の課題", [
        "青いボールを確認し、緑の箱の手前で停止した。",
        "映像は通常速度。待ち時間は録画対象外。",
        "今後：障害物回避・同色物体の区別・結果からの再計画。",
    ]),
]


def clip_info(path):
    if not path.is_file():
        raise ValueError(f"入力動画がありません：{path.name}")
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != SIZE or abs(metadata["fps"] - FPS) > 0.001:
            raise ValueError(f"入力は1280×720・10fpsにしてください：{path.name}")
        count = sum(1 for _ in reader)
        if count == 0:
            raise ValueError(f"入力動画にフレームがありません：{path.name}")
        return {"file": path.name, "frames": count, "seconds": count / FPS}
    finally:
        reader.close()


def slide_pixels(title, lines, duration, font_path):
    image = Image.new("RGB", SIZE, "#141922")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 43)
    body_font = ImageFont.truetype(str(font_path), 29)
    small_font = ImageFont.truetype(str(font_path), 22)
    draw.rectangle((60, 80, 70, 600), fill="#6ed4ff")
    draw.text((100, 85), title, font=title_font, fill="#6ed4ff")
    y = 205
    for text in lines:
        line = ""
        for char in text:
            if line and body_font.getlength(line + char) > 1060:
                draw.text((100, y), line, font=body_font, fill="white")
                y += 43
                line = char
            else:
                line += char
        draw.text((100, y), line, font=body_font, fill="white")
        y += 76
    if y > 630:
        raise ValueError("スライドの文字量が多すぎます")
    draw.text((100, 650), f"PAI 最終課題　|　説明スライド {duration} 秒", font=small_font, fill="#aeb9ca")
    return np.asarray(image)


def build(demo, output, stop_demo=None):
    demo, output = Path(demo), Path(output)
    stop_demo = Path(stop_demo) if stop_demo else None
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("保存先には存在しない .mp4 ファイルを指定してください")
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "meiryo.ttc"
    if not font.is_file():
        raise ValueError("日本語表示用の Meiryo フォントが見つかりません")
    clips = [clip_info(demo)]
    if stop_demo:
        clips.append(clip_info(stop_demo))
    total_frames = sum(clip["frames"] for clip in clips) + sum(slide[0] * FPS for slide in SLIDES)
    if stop_demo:
        total_frames += 6 * FPS
    duration = total_frames / FPS
    if not 60 <= duration <= 180:
        raise ValueError(f"結合後は {duration:.1f} 秒です。提出の1〜3分に収まる素材を使ってください")
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = output.parent / f"{output.stem}_slides"
    previews.mkdir(exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), SIZE, fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", output_params=["-movflags", "+faststart"])
    written = 0

    def write_slide(number, slide):
        nonlocal written
        seconds, title, lines = slide
        pixels = slide_pixels(title, lines, seconds, font)
        Image.fromarray(pixels).save(previews / f"{number:02d}.png")
        for _ in range(seconds * FPS):
            writer.send(pixels)
            written += 1

    def write_clip(path):
        nonlocal written
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            next(reader)
            for frame in reader:
                writer.send(frame)
                written += 1
        finally:
            reader.close()

    writer.send(None)
    try:
        write_slide(1, SLIDES[0])
        write_slide(2, SLIDES[1])
        write_clip(demo)
        if stop_demo:
            write_slide(3, (6, "未対応の指示では停止", [
                "次の映像：『コーヒーを持ってきて』への応答。",
                "このロボットは把持・運搬に対応していない。",
                "停止のみの計画を実行する様子を確認する。",
            ]))
            write_clip(stop_demo)
        write_slide(4, SLIDES[2])
        write_slide(5, SLIDES[3])
    finally:
        writer.close()
    if written != total_frames:
        raise RuntimeError("入力動画が処理中に変わった可能性があります。出力を確認してください")
    manifest = {"video": output.name, "size": list(SIZE), "fps": FPS,
                "frames": written, "seconds": written / FPS, "clips": clips,
                "stop_demo_included": stop_demo is not None, "audio": False}
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("outputs/demo.mp4"))
    parser.add_argument("--stop-demo", type=Path, help="LLM の計画で録画した、コーヒー指示の停止デモ（任意）")
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.mp4"))
    args = parser.parse_args()
    try:
        result = build(args.demo, args.output, args.stop_demo)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"作成：{args.output} / {result['seconds']:.1f}秒 / 1280×720 / 10fps")
