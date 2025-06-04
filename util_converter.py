import numpy as np
from PIL import Image
from pympler.asizeof import asizeof
import lzma
import struct, os
from functools import partial

#frames_path = "playeropt.png"

class Color:
    def __init__(self, color_tuple: tuple, void: bool = False):
        self.is_void = void
        self.r, self.g, self.b, self.a = (
            color_tuple + (255,) * (4 - len(color_tuple))
        )[:4]

    def to_tuple(self):
        return (self.r, self.g, self.b, self.a)
    def all(self, other: "Color"):
        return (
            self.is_void == other.is_void
            and self.r == other.r and self.g == other.g
            and self.b == other.b and self.a == other.a
        )

TRANSPARENT = Color((0, 0, 0, 0))
VOID = Color((0, 0, 0, 0), void=True)

dims: tuple = (1, 1)
data = []#Color(p) for p in img.getdata()]
frame_width = 16
frame_height = 27
offset_x, offset_y = 0, 0

def _getpixel(x: int, y: int):
    return data[y * dims[0] + x] if y * dims[0] + x < len(data) else TRANSPARENT

palette: dict[tuple[int], int] = {}
palette_max = 0

def segment(width: int, height: int, _getpixel_callable):
    global palette_max
    visited, rects = set(), []

    for y in range(height):
        for x in range(width):
            if (x, y) in visited:
                continue
            color = _getpixel_callable(x, y)
            if color.is_void:
                continue

            rect_w = 1
            while (
                x + rect_w < width
                and (x + rect_w, y) not in visited
                and _getpixel_callable(x + rect_w, y).all(color)
            ):
                rect_w += 1

            rect_h, expandable = 1, True
            while expandable and (y + rect_h < height):
                for dx in range(rect_w):
                    if (
                        (x + dx, y + rect_h) in visited
                        or not _getpixel_callable(x + dx, y + rect_h).all(color)
                    ):
                        expandable = False
                        break
                if expandable:
                    rect_h += 1

            for dy in range(rect_h):
                for dx in range(rect_w):
                    visited.add((x + dx, y + dy))

            tpl = color.to_tuple()
            if tpl not in palette:
                palette[tpl] = palette_max
                palette_max += 1
                if palette_max >= 254:
                    raise Exception("COLOR_PALETTE_MAX_EXCEEDED")

            rects.append(((x, y), (rect_w, rect_h), palette[tpl]))
    return rects

def _getpixel_dummy(frame_dict: dict, x: int, y: int):
    return frame_dict.get((x, y), VOID)

# def img_finish(diff_pixels: dict, frames_out: dict, frame_idx: int):
#     rects = segment(frame_width, frame_height, partial(_getpixel_dummy, diff_pixels))
#     frames_out[frame_idx] = rects

def finite_compress():
    overall, prev_pixels, pixbuf = {}, {}, {}
    frame_idx = -1

    for global_x in range(dims[0]):
        local_x = global_x % frame_width
        if local_x == 0:
            pixbuf.clear()
            frame_idx += 1

        for y in range(offset_y, frame_height):
            curr = _getpixel(global_x, y)
            if not prev_pixels.get((local_x, y), TRANSPARENT).all(curr):
                pixbuf[(local_x, y)] = curr
                prev_pixels[(local_x, y)] = curr

        if local_x == frame_width - 1:#diff_pixels: dict, frames_out: dict, frame_idx: int
            rects = segment(frame_width, frame_height, partial(_getpixel_dummy, pixbuf))
            overall[frame_idx] = rects
            #img_finish(pixbuf, overall, frame_idx)
    overall["palette"] = {v: k for k, v in palette.items()}


    out = bytearray()
    out.append(len(overall) - 1)

    for frame_id in sorted(k for k in overall if isinstance(k, int)):
        commands = overall[frame_id]
        out.append(len(commands))
        for ((x, y), (w, h), p_idx) in commands:
            out.extend(struct.pack("BBBBB", x, y, w, h, p_idx))

    pal = overall["palette"]
    out.append(len(pal))
    for i in range(len(pal)):
        out.extend(struct.pack("BBBB", *pal[i]))

    return lzma.compress(bytes(out))

def retrieve(comp_bytes: bytes):
    raw = lzma.decompress(comp_bytes)
    pos = 0

    num_frames = raw[pos]; pos += 1
    frames = []
    for _ in range(num_frames):
        num_cmds = raw[pos]; pos += 1
        cmds = []
        for __ in range(num_cmds):
            x, y, w, h, p_idx = struct.unpack("BBBBB", raw[pos : pos + 5])
            pos += 5
            cmds.append(((x, y), (w, h), p_idx))
        frames.append(cmds)

    num_palette = raw[pos]; pos += 1
    pal = {}
    for i in range(num_palette):
        r, g, b, a = struct.unpack("BBBB", raw[pos : pos + 4])
        pos += 4
        pal[i] = (r, g, b, a)

    #os.makedirs("out", exist_ok=True)
    prev = {}
    sheet = Image.new("RGBA", dims)

    for f_idx, cmds in enumerate(frames):
        new_img = Image.new("RGBA", (frame_width, frame_height))
        for c, col in prev.items():
            new_img.putpixel(c, col)

        for ((x, y), (w, h), p_idx) in cmds:
            col = pal[p_idx]
            for dx in range(w):
                for dy in range(h):
                    coord = (x + dx, y + dy)
                    new_img.putpixel(coord, col)
                    prev[coord] = col
        sheet.paste(new_img, (offset_x + f_idx * frame_width, offset_y))
        if dims[0]-1 > offset_x + f_idx * frame_width and f_idx == len(frames)-1 and offset_x + f_idx * frame_width + frame_width + 1 < dims[0]:
            sheet.putpixel((offset_x + f_idx * frame_width + frame_width + 1, 0), (200,200,200,200))

   # sheet.save("out/sheet.png")
    return {"data": [tuple(p) for p in sheet.getdata()], "dims": dims}

def perform_compress(_data, _dims: tuple) -> dict:
    global data
    global dims
    _data = [Color(p) for p in _data]
    data = _data
    dims = _dims
    return finite_compress()

def perform_decompress(input: dict, _dims: tuple):
    global dims
    dims = _dims
    return retrieve(input)