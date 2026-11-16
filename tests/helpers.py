import numpy as np


def rectangles(n, shape=(32, 64), seed=0):
    """Fotos sintéticas pequenas: retângulo claro em fundo escuro com ruído, e sua máscara."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    images = rng.integers(0, 60, (n, rows, cols, 3), dtype=np.uint8)
    masks = np.zeros((n, rows, cols), dtype=bool)
    for i in range(n):
        r0, c0 = rng.integers(2, rows // 3), rng.integers(2, cols // 3)
        h, w = rng.integers(rows // 4, rows // 2), rng.integers(cols // 4, cols // 2)
        masks[i, r0:r0 + h, c0:c0 + w] = True
        images[i][masks[i]] += 150
    return images, masks
