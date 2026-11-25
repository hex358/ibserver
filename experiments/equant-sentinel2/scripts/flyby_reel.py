"""Один ролик из вариантов flyby_variants.py / flyby_video.py — от самого достоверного к самому понятному — с заставками.

  python scripts/flyby_reel.py ISS_20261118T222808
"""
import subprocess, sys, tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
D = ROOT / "results/flyby"
FONT, FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
obj = "МКС" if name.startswith("ISS") else name.split("_")[0]
CLIPS = [
    ("v1_raw", "1 · Сырые данные Sentinel-2", "Самое достоверное: настоящие пиксели 10 м, без модели.\nКаждая полоса сканирует своей линией в свой момент — поэтому объект на снимке\nраспадается на разноцветные пятна."),
    ("v2_scanner", "2 · Модель в геометрии сканера", "Как записала бы каждая полоса с пикселем 0.6 м: модель снята сканером Sentinel-2,\nотсюда растяжение, перекос и разнос цветов. Геометрия — точная, детали — из модели."),
    ("flyby", "3 · Кадровая камера Sentinel-2 без искажений", "Та же точка и тот же момент, но обычная камера: реальный масштаб и поворот,\nсевер вверху. Объект ближе к камере, чем земля, поэтому выглядит крупнее (λ/R)."),
    ("v4_tracking", "4 · Сопровождение", "Камера ведёт объект, как фотограф ведёт птицу: объект резкий,\nземля уносится вдоль следа."),
    ("v5_flyaround", "5 · Облёт в стиле фото NASA", "Самое понятное: камера в сотнях метров летит вместе с объектом.\nЗемля и облака — из той же съёмки Sentinel-2. Это реконструкция, не фото."),
]


def card(title, text, path, W=1920, H=1080):
    img = Image.new("RGB", (W, H), (8, 11, 16)); dr = ImageDraw.Draw(img)
    f0, f1, f2 = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONTB, 64), ImageFont.truetype(FONT, 34)
    dr.text((140, 300), f"{obj} · 18.11.2026 22:28 UTC · Ocellus" if name.startswith("ISS") else f"{obj} · Ocellus", font=f0, fill=(140, 160, 175))
    dr.text((140, 360), title, font=f1, fill=(240, 244, 248))
    for k, ln in enumerate(text.split("\n")):
        dr.text((140, 480 + 52 * k), ln, font=f2, fill=(190, 202, 212))
    img.save(path)


def main():
    tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-0/-home-user/a0bc3b1a-f0f9-53c5-9f40-08051c32dfae/scratchpad"))
    parts = []
    for k, (suf, title, text) in enumerate(CLIPS):
        clip = D / f"{name}_{suf}.mp4"
        if not clip.exists():
            print("нет", clip.name); continue
        png = tmp / f"c{k}.png"; card(title, text, png)
        cm = tmp / f"c{k}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-t", "3.5", "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(cm)], check=True)
        parts += [cm, clip]
    lst = tmp / "list.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = D / f"{name}_reel.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", "30", str(out)], check=True)
    print("OK", out)


if __name__ == "__main__":
    main()
