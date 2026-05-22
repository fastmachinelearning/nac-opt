"""
hls4ml LHC jet tagging dataset loader.

Downloads hls4ml_lhc_jets_hlf from OpenML on first use; subsequent calls load
from the sklearn cache (~/.scikit_learn_data). Mirrors the preprocessing applied
in the hls4ml tutorial part1_getting_started.ipynb:

    LabelEncoder (g,q,t,w,z -> 0..4)
    train_test_split (80/20, stratified)
    StandardScaler (fit on train, transform both)
    to_categorical (5 classes)
"""

import numpy as np
from tensorflow.keras.utils import to_categorical


# Alphabetical LabelEncoder order (g < q < t < w < z)
JET_CLASSES = ["g", "q", "t", "w", "z"]

# The 16 high-level jet-substructure observables returned by OpenML
JET_FEATURE_NAMES = [
    "zlogz",
    "c1_b0_mmdt",
    "c1_b1_mmdt",
    "c1_b2_mmdt",
    "c2_b1_mmdt",
    "c2_b2_mmdt",
    "d2_b1_mmdt",
    "d2_b2_mmdt",
    "d2_a1_b1_mmdt",
    "d2_a1_b2_mmdt",
    "m2_b1_mmdt",
    "m2_b2_mmdt",
    "n2_b1_mmdt",
    "n2_b2_mmdt",
    "mass_mmdt",
    "multiplicity",
]


def load_and_preprocess_hls4ml_jets(
    subset_size=None,
    val_subset_size=None,
    normalize=True,
    flatten=True,
    one_hot=True,
    num_classes=5,
    val_split=0.2,
    random_state=42,
    **_ignored,
):
    """
    Download and preprocess the hls4ml LHC jet tagging dataset from OpenML.

    The dataset (hls4ml_lhc_jets_hlf) contains 830 000 jet events described
    by 16 high-level substructure observables.  The classification task is to
    identify which of five particle types (g, q, t, W, Z) produced the jet.

    On first call sklearn downloads the dataset (~30 MB compressed) and caches
    it under ~/.scikit_learn_data.  Subsequent calls load from the cache.

    Parameters
    ----------
    subset_size : int or None
        If set, subsample ONLY the training split to this many examples.
    val_subset_size : int or None
        If set, stratified-subsample the test/validation split to roughly this
        many examples. The full test split is ~166k rows; used as-is it is
        validated on every training epoch (in both global and local search),
        which dominates wall-clock time. A few thousand stratified rows give an
        equivalent validation signal far more cheaply.
    normalize : bool
        Apply StandardScaler (fit on train, transform both splits).
    flatten : bool
        If False, add a trailing channel dimension (batch, 16, 1).
        required for Conv/ConvAttn block searches.
    one_hot : bool
        Encode labels as one-hot vectors of length `num_classes`.
    num_classes : int
        Number of output classes (default 5).
    val_split : float
        Fraction of data to reserve as the test/validation split.
    random_state : int
        Random seed for train/test split and (if subset_size is set)
        training-subset selection.

    Returns
    -------
    x_train, y_train, x_test, y_test : numpy arrays
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print("Fetching hls4ml_lhc_jets_hlf from OpenML (cached after first download)...")
    data = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=True, parser="auto")

    X = data["data"].to_numpy(dtype=np.float32)  # (830000, 16)
    y_raw = data["target"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)  # g,q,t,w,z -> 0,1,2,3,4

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )

    # Stratified subsample of the (large) test split, on integer labels before
    # one-hot encoding. Keeps roughly val_subset_size rows, balanced per class.
    if val_subset_size is not None and val_subset_size < len(x_test):
        rng = np.random.RandomState(random_state)
        per_class = max(1, val_subset_size // num_classes)
        sel = np.concatenate(
            [rng.permutation(np.where(y_test == c)[0])[:per_class] for c in np.unique(y_test)]
        )
        rng.shuffle(sel)
        x_test, y_test = x_test[sel], y_test[sel]

    if normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

    if not flatten:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

    if subset_size is not None:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(x_train))
        x_train = x_train[idx[:subset_size]]
        y_train = y_train[idx[:subset_size]]

    print("hls4ml jet tagging data loaded and preprocessed:")
    print(f"  x_train shape : {x_train.shape},  x_test shape : {x_test.shape}")
    print(f"  y_train shape : {y_train.shape},  y_test shape : {y_test.shape}")
    print(f"  classes       : {list(le.classes_)}")

    return x_train, y_train, x_test, y_test
