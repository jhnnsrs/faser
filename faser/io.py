import json
import xarray as xr
import zarr as zr
import numpy as np


def save(generator, config, path, as_zarr=False, **kwargs):
    """Save a generator to a zarr file.

    Parameters
    ----------
    generator : generator
        The generator to save.
    path : str
        The path to the zarr file.
    **kwargs
        Additional keyword arguments to pass to xarray.to_zarr.

    """

    data = generator(config).generate()

    if as_zarr:
        with open(path, "wb") as f:
            np.save(f, data)
        return path

    else:
        store = zr.DirectoryStore(path, compression=None)

        return (
            xr.DataArray(
                data,
                dims=list("xyz"),
                attrs={"config": config.json(), "generator": generator.__name__},
            )
            .to_dataset(name="data")
            .to_zarr(store, mode="w", compute=True)
        )


def load(path):
    if True:
        with open(path, "rb") as f:
            data = np.load(f)
        return data

    return xr.open_zarr(path)["data"]
