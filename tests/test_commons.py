import numpy as np

from fishweight.commons import find_bighest_cluster
from fishweight.segmentation import handcraft  # noqa: F401  (o import precisa funcionar)


def test_find_bighest_cluster_keeps_only_largest_blob():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:15, 5:15] = True    # 100 px
    mask[40:43, 40:43] = True  # 9 px
    result = find_bighest_cluster(mask)
    assert result.sum() == 100
    assert not result[40:43, 40:43].any()
