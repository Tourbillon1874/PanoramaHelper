from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import math


def cubemap_batch_to_equirect(cubemap_batch, out_h=2048, out_w=4096, sample_mode='bicubic'):
    cubemap = cubemap_batch.permute(0, 3, 1, 2).float()
    face_h, face_w = cubemap.shape[2], cubemap.shape[3]
    lon = torch.linspace(-math.pi, math.pi, out_w, device=cubemap.device)
    lat = torch.linspace(-math.pi/2, math.pi/2, out_h, device=cubemap.device)
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')
    x = torch.cos(lat_grid) * torch.sin(lon_grid)
    y = torch.sin(lat_grid)
    z = torch.cos(lat_grid) * torch.cos(lon_grid)
    abs_x, abs_y, abs_z = torch.abs(x), torch.abs(y), torch.abs(z)
    mask_front = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
    mask_back = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)
    mask_right = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    mask_left = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
    mask_up = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    mask_down = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    face_idx = torch.zeros_like(x, dtype=torch.long)

    sc, tc = x / z.clamp(min=1e-8), -y / z.clamp(min=1e-8)
    u = torch.where(mask_front, (sc + 1) / 2, u)
    v = torch.where(mask_front, (tc + 1) / 2, v)
    face_idx = torch.where(mask_front, 0, face_idx)

    sc, tc = -x / (-z).clamp(min=1e-8), -y / (-z).clamp(min=1e-8)
    u = torch.where(mask_back, (sc + 1) / 2, u)
    v = torch.where(mask_back, (tc + 1) / 2, v)
    face_idx = torch.where(mask_back, 2, face_idx)

    sc, tc = -z / x.clamp(min=1e-8), -y / x.clamp(min=1e-8)
    u = torch.where(mask_right, (sc + 1) / 2, u)
    v = torch.where(mask_right, (tc + 1) / 2, v)
    face_idx = torch.where(mask_right, 1, face_idx)

    sc, tc = z / (-x).clamp(min=1e-8), -y / (-x).clamp(min=1e-8)
    u = torch.where(mask_left, (sc + 1) / 2, u)
    v = torch.where(mask_left, (tc + 1) / 2, v)
    face_idx = torch.where(mask_left, 3, face_idx)

    sc, tc = x / y.clamp(min=1e-8), z / y.clamp(min=1e-8)
    u = torch.where(mask_up, (sc + 1) / 2, u)
    v = torch.where(mask_up, (tc + 1) / 2, v)
    face_idx = torch.where(mask_up, 4, face_idx)

    sc, tc = x / (-y).clamp(min=1e-8), -z / (-y).clamp(min=1e-8)
    u = torch.where(mask_down, (sc + 1) / 2, u)
    v = torch.where(mask_down, (tc + 1) / 2, v)
    face_idx = torch.where(mask_down, 5, face_idx)

    grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).unsqueeze(0)

    output = torch.zeros(1, 3, out_h, out_w, device=cubemap.device, dtype=cubemap.dtype)
    for i in range(6):
        mask = (face_idx == i).unsqueeze(0).unsqueeze(0)
        if mask.sum() == 0:
            continue
        sampled = F.grid_sample(
            cubemap[i:i+1], grid,
            mode=sample_mode, padding_mode='border', align_corners=False
        )
        output = torch.where(mask, sampled, output)

    result = output.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    result = result.flip(dims=[1])
    return result


def tile_right(image):
    return torch.cat([image, image], dim=2)


def crop_image(image, x, y, w, h):
    return image[:, y:y+h, x:x+w, :]


def resize_image(image, target_w, target_h, mode='lanczos'):
    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    if mode != 'lanczos':
        img = image.permute(0, 3, 1, 2)
        resized = F.interpolate(img, size=(target_h, target_w), mode=mode, align_corners=False)
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    result_list = []
    image_np = (image.cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(B):
        pil_img = Image.fromarray(image_np[i])
        resized_pil = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
        resized_np = np.array(resized_pil).astype(np.float32) / 255.0
        result_list.append(torch.from_numpy(resized_np))
    
    result = torch.stack(result_list, dim=0).to(device=device, dtype=dtype)
    return result.clamp(0.0, 1.0)


def adain_color_match(target, reference):

    t = target.permute(0, 3, 1, 2)
    r = reference.permute(0, 3, 1, 2)

    t_mean = t.mean(dim=[2,3], keepdim=True)
    t_std = t.std(dim=[2,3], keepdim=True) + 1e-8
    r_mean = r.mean(dim=[2,3], keepdim=True)
    r_std = r.std(dim=[2,3], keepdim=True) + 1e-8

    matched = r_std * (t - t_mean) / t_std + r_mean
    matched = matched.permute(0, 2, 3, 1)
    return torch.clamp(matched, 0.0, 1.0)


def blend_with_mask(background, layer, mask):

    mask = mask.unsqueeze(-1)
    return background * (1 - mask) + layer * mask


def fix_left_right_seam(image, overlap_width=256, expand=-120, blur_radius=48.0):

    B, H, W, C = image.shape
    std_width = W - overlap_width

    main_img = image[:, :, :std_width, :]
    edge_strip = image[:, :, std_width:, :]

    pad_right = std_width - overlap_width
    layer_img = F.pad(edge_strip, (0, 0, 0, pad_right, 0, 0), mode='constant', value=0.0)

    mask = torch.zeros(B, H, std_width, 1, device=image.device, dtype=image.dtype)
    mask[:, :, :overlap_width, :] = 1.0

    shrink = int(abs(expand))
    if expand < 0 and shrink < overlap_width:
        mask[:, :, overlap_width - shrink:overlap_width, :] = 0.0

    if blur_radius > 0:
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255.0).astype(np.uint8)
        blurred_list = []
        for i in range(B):
            pil_mask = Image.fromarray(mask_np[i], mode='L')
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(blur_radius))
            blurred_np = np.array(pil_mask).astype(np.float32) / 255.0
            blurred_list.append(torch.from_numpy(blurred_np).unsqueeze(-1))
        mask = torch.stack(blurred_list, dim=0).to(device=image.device, dtype=image.dtype)

    result = main_img * (1 - mask) + layer_img * mask
    return result.clamp(0.0, 1.0)