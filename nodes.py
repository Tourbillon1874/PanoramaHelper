from .utils import (
    cubemap_batch_to_equirect,
    tile_right,
    crop_image,
    resize_image,
    adain_color_match,
    blend_with_mask,
    fix_left_right_seam
)


class PanoramaPreprocess:

    BASE_CROP_W = 4242
    BASE_CROP_H = 2048

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cubemap_batch": ("IMAGE",),
                "output_height": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 1}),
                "resize_mode": (["lanczos", "bicubic", "bilinear"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PanoramaHelper"

    def process(self, cubemap_batch, output_height, resize_mode):
        equirect = cubemap_batch_to_equirect(cubemap_batch, out_h=2048, out_w=4096, sample_mode='bicubic')
        tiled = tile_right(equirect)
        cropped = crop_image(tiled, x=0, y=0, w=self.BASE_CROP_W, h=self.BASE_CROP_H)
        scale = output_height / self.BASE_CROP_H
        output_width = int(round(self.BASE_CROP_W * scale))
        result = resize_image(cropped, output_width, output_height, mode=resize_mode)
        return (result,)


class PanoramaPostprocess:

    BASE_HEIGHT = 3584
    BASE_OVERLAP = 256
    BASE_EXPAND = -120
    BASE_BLUR_RADIUS = 48.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "sky_ground_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PanoramaHelper"

    def process(self, processed_image, original_image, sky_ground_mask):
        color_matched = adain_color_match(original_image, processed_image)
        mixed = blend_with_mask(processed_image, color_matched, 1.0 - sky_ground_mask)
        img_h = processed_image.shape[1]
        scale = img_h / self.BASE_HEIGHT
        overlap_width = int(round(self.BASE_OVERLAP * scale))
        expand = int(round(self.BASE_EXPAND * scale))
        blur_radius = self.BASE_BLUR_RADIUS * scale
        result = fix_left_right_seam(mixed, overlap_width=overlap_width, expand=expand, blur_radius=blur_radius)
        return (result,)