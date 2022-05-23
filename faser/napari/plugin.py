from napari_plugin_engine import napari_hook_implementation
from .gui import generate_psf


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return generate_psf, {"area": "right", "name": "Faser"}
