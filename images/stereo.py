from sys import argv

from numpy import (
    arange,
    argmin,
    array,
    full,
    inf,
    newaxis,
    zeros,
)
from numpy.linalg import norm
from PIL import Image


if __name__ == "__main__":
    assert len(argv) == 6

    left_image = array(Image.open(argv[1]), dtype=float)
    right_image = array(Image.open(argv[2]), dtype=float)

    assert left_image.shape == right_image.shape

    max_disparity = int(argv[3])

    assert max_disparity > 0

    smoothness = float(argv[4])

    assert smoothness >= 0

    output_image_name = argv[5]

    height = left_image.shape[0]
    width = left_image.shape[1]

    cache = full((*left_image.shape[:2], max_disparity + 1), inf)

    cache[:, 0, 0] = norm(left_image[:, 0] - right_image[:, 0], axis=-1)
    for x in range(1, width):
        print(f"Column {x}")
        real_max_disparity = min(max_disparity, x)
        disparities = arange(real_max_disparity + 1)
        cache[:, x, disparities] = (
            abs(left_image[:, x, newaxis] - right_image[:, x - disparities])
            + (
                cache[:, x - 1, disparities, newaxis]
                + smoothness * abs(disparities[:, newaxis] - disparities[newaxis, :])[newaxis, ...]
            ).min(axis=-2)
        )

    disparity_map = zeros(left_image.shape[:2], dtype=int)

    disparity_map[:, -1] = argmin(cache[:, -1], axis=-1)
    for x in range(2, width):
        real_max_disparity = min(max_disparity, x)
        disparities = arange(real_max_disparity + 1)
        print(f"Backward pass: column {x}")
        disparity_map[:, -x] = argmin(
            cache[:, -x, disparities]
            + abs(
                disparity_map[:, -x + 1, newaxis]
                - disparities[newaxis, :]
            )
        , axis=-1)

    disparity_map_normalized = (
        255 * disparity_map.astype(float) / disparity_map.max()
    )
    Image.fromarray(
        disparity_map_normalized.astype("uint8")
    ).save(output_image_name)
