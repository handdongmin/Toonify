import os

import cv2
import numpy as np


INPUT_PATH = "input/image.jpg"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cartoon.jpg")
SHOW_PREVIEW = os.environ.get("SHOW_PREVIEW", "1").lower() not in {"0", "false", "no"}

MAX_PROCESSING_SIDE = 960
PALETTE_COLORS = 12
DETAIL_GAIN = 0.95
LINE_STRENGTH = 0.48


def resize_for_processing(image, max_side=MAX_PROCESSING_SIDE):
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_side:
        return image.copy(), 1.0

    scale = max_side / float(longest_side)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def quantize_palette(image, palette_colors=PALETTE_COLORS):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)
    ab = np.dstack((a_channel, b_channel)).reshape((-1, 2)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        24,
        0.2,
    )
    _, labels, centers = cv2.kmeans(
        ab,
        palette_colors,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    palette_ab = centers[labels.flatten()].reshape((image.shape[0], image.shape[1], 2))
    chroma_distance = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)

    color_weight = np.clip((chroma_distance - 6.0) / 24.0, 0.0, 1.0)[:, :, None]
    neutral_highlight = np.clip((l_channel - 150.0) / 70.0, 0.0, 1.0)
    neutral_highlight *= np.clip((18.0 - chroma_distance) / 18.0, 0.0, 1.0)

    blended_ab = np.dstack((a_channel, b_channel)) * (1.0 - color_weight) + palette_ab * color_weight
    blended_ab = blended_ab * (1.0 - neutral_highlight[:, :, None]) + 128.0 * neutral_highlight[:, :, None]

    quantized_lab = cv2.merge(
        (
            l_channel.astype(np.uint8),
            np.clip(blended_ab[:, :, 0], 0, 255).astype(np.uint8),
            np.clip(blended_ab[:, :, 1], 0, 255).astype(np.uint8),
        )
    )
    return cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)


def build_color_layer(image):
    processed, scale = resize_for_processing(image)

    smooth = cv2.edgePreservingFilter(processed, flags=1, sigma_s=55, sigma_r=0.28)
    smooth = cv2.pyrMeanShiftFiltering(smooth, sp=14, sr=24)
    palette = quantize_palette(smooth)

    if scale != 1.0:
        palette = cv2.resize(
            palette,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.edgePreservingFilter(palette, flags=1, sigma_s=24, sigma_r=0.16)


def restore_tone_and_texture(color_layer, reference, detail_gain=DETAIL_GAIN):
    base_lab = cv2.cvtColor(color_layer, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    base_l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(base_lab[:, :, 0].astype(np.uint8)).astype(np.float32)
    ref_l = ref_lab[:, :, 0]

    detail = ref_l - cv2.GaussianBlur(ref_l, (0, 0), 1.1)
    detail_weight = cv2.GaussianBlur(np.abs(detail), (0, 0), 1.5)
    detail_weight = cv2.normalize(detail_weight, None, 0.0, 1.0, cv2.NORM_MINMAX)

    restored_l = 0.76 * base_l + 0.24 * ref_l
    restored_l += detail * detail_gain * (0.2 + 0.8 * detail_weight)
    restored_l += np.clip(ref_l - base_l, 0.0, 255.0) * 0.18 + 3.0

    base_lab[:, :, 0] = np.clip(restored_l, 0, 255)
    return cv2.cvtColor(base_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def build_line_map(reference):
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    canny = cv2.Canny(gray, 45, 120)
    canny = cv2.dilate(canny, np.ones((2, 2), np.uint8), iterations=1)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian = cv2.convertScaleAbs(np.abs(laplacian))
    _, laplacian = cv2.threshold(laplacian, 24, 255, cv2.THRESH_BINARY)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )
    adaptive = cv2.convertScaleAbs(adaptive, alpha=0.28)

    line_map = np.maximum.reduce([canny, laplacian, adaptive])
    line_map = cv2.morphologyEx(line_map, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.GaussianBlur(line_map, (3, 3), 0)


def composite_lines(color_layer, line_map, reference, line_strength=LINE_STRENGTH):
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = ref_hsv[:, :, 1] / 255.0
    value = ref_hsv[:, :, 2] / 255.0

    line_alpha = (line_map.astype(np.float32) / 255.0) ** 0.9
    white_protect = np.clip((value - 0.72) / 0.28, 0.0, 1.0) * np.clip((0.18 - saturation) / 0.18, 0.0, 1.0)
    darkening = line_strength * (1.0 - 0.55 * white_protect)

    shaded = color_layer.astype(np.float32) * (1.0 - line_alpha[:, :, None] * darkening[:, :, None])
    return np.clip(shaded, 0, 255).astype(np.uint8)


def preserve_neutral_highlights(cartoon, reference):
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(ref_lab)
    chroma_distance = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)

    mask = np.clip((l_channel - 150.0) / 70.0, 0.0, 1.0)
    mask *= np.clip((18.0 - chroma_distance) / 18.0, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.6)[:, :, None]

    blend = 0.42 * mask
    mixed = cartoon.astype(np.float32) * (1.0 - blend) + reference.astype(np.float32) * blend
    return np.clip(mixed, 0, 255).astype(np.uint8)


def match_reference_brightness(cartoon, reference):
    cartoon_lab = cv2.cvtColor(cartoon, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    cartoon_l = cartoon_lab[:, :, 0]
    ref_l = ref_lab[:, :, 0]
    delta = np.clip(ref_l - cartoon_l, 0.0, 255.0)

    lift = delta * 0.34
    lift += max(ref_l.mean() - cartoon_l.mean(), 0.0) * 0.55

    cartoon_lab[:, :, 0] = np.clip(cartoon_l + lift, 0, 255)
    return cv2.cvtColor(cartoon_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_vibrance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gain = np.clip((saturation - 30.0) / 110.0, 0.0, 1.0)
    hsv[:, :, 1] = np.clip(saturation + gain * (255.0 - saturation) * 0.06, 0, 255)

    white_mask = np.clip((value - 170.0) / 55.0, 0.0, 1.0) * np.clip((24.0 - saturation) / 24.0, 0.0, 1.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - 0.6 * white_mask), 0, 255)
    hsv[:, :, 2] = np.clip(value * 1.02 + 2.0, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def render_cartoon(image):
    color_layer = build_color_layer(image)
    toned_layer = restore_tone_and_texture(color_layer, image)
    line_map = build_line_map(image)

    cartoon = composite_lines(toned_layer, line_map, image)
    cartoon = preserve_neutral_highlights(cartoon, image)
    cartoon = match_reference_brightness(cartoon, image)
    return apply_vibrance(cartoon)


def preview_image(title, image):
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Preview window could not be opened in this environment.")


def main():
    image = cv2.imread(INPUT_PATH)
    if image is None:
        print(f"Failed to load input image: {INPUT_PATH}")
        raise SystemExit(1)

    cartoon = render_cartoon(image)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, cartoon)
    print(f"Saved: {OUTPUT_PATH}")

    if SHOW_PREVIEW:
        preview_image("Cartoon Render", cartoon)


if __name__ == "__main__":
    main()
