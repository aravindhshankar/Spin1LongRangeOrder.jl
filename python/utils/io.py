import h5py
from tenpy.tools import hdf5_io


def _normalize_filename(filename):
    """Ensure filename ends with .h5 exactly once."""
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    filename = filename.strip()

    if not filename.lower().endswith(".h5"):
        filename += ".h5"

    return filename


def save_mps_with_metadata(filename=None, mps=None, metadata=None):
    """
    Save an optional MPS and optional metadata.

    Parameters
    ----------
    mps : MPS or None
    filename : str
    metadata : dict or None
    """
    if filename is None:
        raise ValueError("filename must be provided")

    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        raise TypeError(
            f"metadata must be a dict, got {type(metadata).__name__}"
        )

    filename = _normalize_filename(filename)

    with h5py.File(filename, "w") as f:

        if mps is not None:
            hdf5_io.save_to_hdf5(f, mps, "mps")

        if metadata:
            hdf5_io.save_to_hdf5(f, metadata, "metadata")

    return filename


def load_mps_with_metadata(filename):
    """
    Load an optional MPS and optional metadata.

    Returns
    -------
    mps : MPS or None
    metadata : dict
    """
    filename = _normalize_filename(filename)

    with h5py.File(filename, "r") as f:

        mps = (
            hdf5_io.load_from_hdf5(f, "mps")
            if "mps" in f
            else None
        )

        metadata = (
            hdf5_io.load_from_hdf5(f, "metadata")
            if "metadata" in f
            else {}
        )

    return mps, metadata