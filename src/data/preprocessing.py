import numpy as np

def normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(float)
    min_value = np.min(image)
    image -= min_value
    max_value = np.max(image) + eps
    image /= max_value
    return image
