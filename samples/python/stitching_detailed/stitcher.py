from types import SimpleNamespace

import cv2 as cv
import numpy as np

from . import stitcher_choices as choices
from .panorama_part import PanoramaPart
from .image_to_megapix_scaler import ImageToMegapixScaler
from .image_registration import ImageRegistration
from .warper import Warper


class Stitcher:

    def __init__(self, img_names, **kwargs):
        self.img_names = img_names
        self.args = SimpleNamespace(**kwargs)

    def downscale_images(self, images, megapix):
        scaler = ImageToMegapixScaler(megapix)
        images = [scaler.set_scale_and_downscale(img) for img in images]
        return images, scaler.scale

        # work_images, work_scale = self.downscale_images(images,
        #                                                 self.args.work_megapix)

    def stitch(self):
        args = self.args
        img_names = self.img_names

        work_megapix_scaler = ImageToMegapixScaler(args.work_megapix)
        seam_megapix_scaler = ImageToMegapixScaler(args.seam_megapix)

        conf_thresh = args.conf_thresh
        if args.save_graph is None:
            save_graph = False
        else:
            save_graph = True

        full_img_sizes = []
        work_imgs = []
        images = []
        is_compose_scale_set = False

        for name in img_names:
            image = PanoramaPart(name)
            image.read_image()
            full_img_sizes.append(image.full_img_size)
            work_img = work_megapix_scaler.set_scale_and_downscale(image.full_img)
            work_imgs.append(work_img)
            seam_img = seam_megapix_scaler.set_scale_and_downscale(image.full_img)
            images.append(seam_img)

        seam_work_aspect = (work_megapix_scaler.scale /
                            seam_megapix_scaler.scale )

        image_registration = get_image_registration_object(args)
        features = image_registration.find_features(work_imgs)
        p = image_registration.match_features(features)

        if save_graph:
            with open(args.save_graph, 'w') as fh:
                fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

        indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(img_names[indices[i, 0]])
            img_subset.append(images[indices[i, 0]])
            full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(img_names)
        if num_images < 2:
            print("Need more images")
            exit()

        cameras = image_registration.estimate_camera_parameters(features, p)
        cameras = image_registration.adjust_camera_parameters(features, p, cameras)

        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

        cameras = image_registration.perform_wave_correction(cameras)

# =============================================================================
# COMPOSITION PART
# =============================================================================

        compose_megapix = args.compose_megapix
        warp_type = args.warp
        blend_type = args.blend
        blend_strength = args.blend_strength
        result_name = args.output
        if args.timelapse is not None:
            timelapse = True
            if args.timelapse == "as_is":
                timelapse_type = cv.detail.Timelapser_AS_IS
            elif args.timelapse == "crop":
                timelapse_type = cv.detail.Timelapser_CROP
            else:
                print("Bad timelapse method")
                exit()
        else:
            timelapse = False

        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = Warper(warp_type, warped_image_scale * seam_work_aspect)
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            corner, image_wp = warper.warp_image(images[idx], cameras[idx], swa)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp_image(masks[idx], cameras[idx], swa)
            masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        compensator = get_compensator(args)
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = choices.SEAM_FIND_CHOICES[args.seam]
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = None
        # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(img_names):
            full_img = cv.imread(name)
            if not is_compose_scale_set:
                if compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_megapix_scaler.scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                          int(round(full_img_sizes[i][1] * compose_scale)))
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img
            _img_size = (img.shape[1], img.shape[0])
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None and not timelapse:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            elif timelapser is None and timelapse:
                timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
                timelapser.initialize(corners, sizes)
            if timelapse:
                ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                timelapser.process(image_warped_s, ma_tones, corners[idx])
                pos_s = img_names[idx].rfind("/")
                if pos_s == -1:
                    fixed_file_name = "fixed_" + img_names[idx]
                else:
                    fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
                cv.imwrite(fixed_file_name, timelapser.getDst())
            else:
                blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        if not timelapse:
            result = None
            result_mask = None
            self.result, result_mask = blender.blend(result, result_mask)
            cv.imwrite(result_name, self.result)
            zoom_x = 600.0 / self.result.shape[1]
            dst = cv.normalize(src=self.result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            self.dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)


        print("Done")


def get_image_registration_object(args):
    return ImageRegistration(args.features,
                             args.matcher,
                             args.rangewidth,
                             args.try_cuda,
                             args.match_conf,
                             args.estimator,
                             args.ba,
                             args.ba_refine_mask,
                             args.wave_correct)

def get_compensator(args):
    expos_comp_type = choices.EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator