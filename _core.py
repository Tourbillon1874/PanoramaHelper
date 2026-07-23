from .nodes import PanoramaPreprocess, PanoramaPostprocess

NODE_CLASS_MAPPINGS = {
    "PanoramaPreprocess": PanoramaPreprocess,
    "PanoramaPostprocess": PanoramaPostprocess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaPreprocess": "PanoramaPreprocess",
    "PanoramaPostprocess": "PanoramaPostprocess",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']