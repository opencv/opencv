import statistics


def estimate_final_panorama_dimensions(cameras, warper, img_handler):
    medium_to_final_ratio = img_handler.get_medium_to_final_ratio()

    panorama_scale_determined_on_medium_img = \
        estimate_panorama_scale(cameras)

    panorama_scale = (panorama_scale_determined_on_medium_img *
                      medium_to_final_ratio)
    panorama_corners = []
    panorama_sizes = []

    for size, camera in zip(img_handler.img_sizes, cameras):
        width, height = img_handler.final_scaler.get_scaled_img_size(size)
        roi = warper.warp_roi(width, height, camera, panorama_scale, medium_to_final_ratio)
        panorama_corners.append(roi[0:2])
        panorama_sizes.append(roi[2:4])

    return panorama_scale, panorama_corners, panorama_sizes


def estimate_panorama_scale(cameras):
    focals = [cam.focal for cam in cameras]
    panorama_scale = statistics.median(focals)
    return panorama_scale
