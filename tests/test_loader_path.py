"""load_generic_dataset(loader_path=...) imports a custom loader, as dataset specs in the README describe."""

import numpy as np
import pytest

from utils.tf_data_preprocessing import load_generic_dataset


def test_loader_path_imports_and_calls_the_custom_loader():
    x_train, y_train, x_val, y_val = load_generic_dataset(
        "my_custom_dataset", loader_path="tests.synthetic_data:load_blobs", subset_size=None, flatten=True, one_hot=True
    )
    assert x_train.shape == (900, 16) and y_train.shape == (900, 4)
    assert x_val.shape == (300, 16) and y_val.shape == (300, 4)
    assert np.allclose(y_train.sum(axis=1), 1.0)


def test_unknown_dataset_without_loader_path_still_errors():
    with pytest.raises(ValueError, match="loader_path"):
        load_generic_dataset("my_custom_dataset")


@pytest.mark.parametrize("bad", ["no_colon", ":func", "mod:"])
def test_bad_loader_path_is_rejected(bad):
    with pytest.raises(ValueError, match="loader_path"):
        load_generic_dataset("anything", loader_path=bad)
