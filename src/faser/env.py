import os


FASER_PATH = os.path.dirname(os.path.realpath(__file__))
ASSETS_PATH = os.path.join(FASER_PATH, "assets")


def get_asset_file(file, darkMode=False):
    return os.path.join(ASSETS_PATH, file)
