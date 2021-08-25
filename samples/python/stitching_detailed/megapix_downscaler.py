from .megapix_scaler import MegapixScaler


class MegapixDownscaler(MegapixScaler):

    @staticmethod
    def force_downscale(scale):
        return min(1.0, scale)

    def get_scale_by_img_size(self, img_size):
        return self.force_downscale(
            super().get_scale_by_img_size(img_size)
            )

    def set_scale_by_img_size(self, img_size):
        scale = super()._get_scale_by_resolution(img_size[0] * img_size[1])
        self.scale = self.force_downscale(scale)
        self.is_scale_set = True
