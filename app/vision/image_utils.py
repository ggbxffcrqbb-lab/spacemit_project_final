from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


NV12_DUMP_PATTERN = re.compile(r"_(?P<width>\d+)x(?P<height>\d+)_s(?P<stride>\d+)\.nv12$")


def load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def save_rgb_image(rgb: np.ndarray, image_path: Path) -> Path:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(image_path)
    return image_path


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[..., ::-1])


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[..., ::-1])


def resize_long_edge(rgb: np.ndarray, max_long_edge: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return rgb
    scale = max_long_edge / float(long_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(rgb, mode="RGB")
    return np.asarray(image.resize(new_size, Image.Resampling.BILINEAR))


def parse_nv12_dump_path(nv12_path: Path) -> tuple[int, int, int]:
    match = NV12_DUMP_PATTERN.search(nv12_path.name)
    if not match:
        raise ValueError(f"无法从 NV12 文件名解析尺寸: {nv12_path}")
    return (
        int(match.group("width")),
        int(match.group("height")),
        int(match.group("stride")),
    )


def load_nv12_as_rgb(nv12_path: Path) -> np.ndarray:
    width, height, stride = parse_nv12_dump_path(nv12_path)
    data = np.fromfile(nv12_path, dtype=np.uint8)
    expected_size = stride * height * 3 // 2
    if data.size < expected_size:
        raise ValueError(
            f"NV12 数据长度不足: got={data.size}, expected_at_least={expected_size}, file={nv12_path}"
        )

    y_plane = data[: stride * height].reshape((height, stride))[:, :width].astype(np.float32)
    uv_plane = data[stride * height : stride * height + stride * height // 2].reshape(
        (height // 2, stride)
    )[:, :width]

    u_plane = uv_plane[:, 0::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)
    v_plane = uv_plane[:, 1::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)

    y = y_plane
    u = u_plane[:height, :width] - 128.0
    v = v_plane[:height, :width] - 128.0

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def nv12_bytes_to_rgb(
    data: bytes | bytearray | memoryview | np.ndarray,
    *,
    width: int,
    height: int,
    stride: int | None = None,
) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    stride = stride or width
    expected_size = stride * height * 3 // 2
    if array.size < expected_size:
        raise ValueError(
            f"NV12 鏁版嵁闀垮害涓嶈冻: got={array.size}, expected_at_least={expected_size}, "
            f"width={width}, height={height}, stride={stride}"
        )

    y_plane = array[: stride * height].reshape((height, stride))[:, :width].astype(np.float32)
    uv_plane = array[stride * height : stride * height + stride * height // 2].reshape(
        (height // 2, stride)
    )[:, :width]

    u_plane = uv_plane[:, 0::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)
    v_plane = uv_plane[:, 1::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)

    y = y_plane
    u = u_plane[:height, :width] - 128.0
    v = v_plane[:height, :width] - 128.0

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)
