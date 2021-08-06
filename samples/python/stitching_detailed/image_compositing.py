from stitching_detailed.warper import Warper
from stitching_detailed.exposure_error_compensator import ExposureErrorCompensator  # noqa
from stitching_detailed.seam_finder import SeamFinder
from stitching_detailed.blender import Blender
from stitching_detailed.timelapser import Timelapser


class ImageCompositing:
    def __init__(self,
                 warper=Warper(),
                 seam_finder=SeamFinder(),
                 compensator=ExposureErrorCompensator(),
                 blender=Blender(),
                 timelapser=Timelapser()):

        self.warper = warper
        self.seam_finder = seam_finder
        self.compensator = compensator
        self.blender = blender
        self.timelapser = timelapser
