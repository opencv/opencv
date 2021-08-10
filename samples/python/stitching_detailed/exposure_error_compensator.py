from collections import OrderedDict
import cv2 as cv


class ExposureErrorCompensator:

    COMPENSATOR_CHOICES = OrderedDict()
    COMPENSATOR_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS  # noqa
    COMPENSATOR_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
    COMPENSATOR_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
    COMPENSATOR_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS  # noqa
    COMPENSATOR_CHOICES['no'] = cv.detail.ExposureCompensator_NO

    DEFAULT_COMPENSATOR = list(COMPENSATOR_CHOICES.keys())[0]
    DEFAULT_NR_FEEDS = 1
    DEFAULT_BLOCK_SIZE = 32

    def __init__(self,
                 compensator=DEFAULT_COMPENSATOR,
                 nr_feeds=DEFAULT_NR_FEEDS,
                 block_size=DEFAULT_BLOCK_SIZE):

        if compensator == 'channel':
            self.compensator = cv.detail_ChannelsCompensator(nr_feeds)
        elif compensator == 'channel_blocks':
            self.compensator = cv.detail_BlocksChannelsCompensator(
                block_size, block_size, nr_feeds
                )
        else:
            self.compensator = cv.detail.ExposureCompensator_createDefault(
                ExposureErrorCompensator.COMPENSATOR_CHOICES[compensator]
                )

    def feed(self, *args):
        """https://docs.opencv.org/master/d2/d37/classcv_1_1detail_1_1ExposureCompensator.html#ae6b0cc69a7bc53818ddea53eddb6bdba"""  # noqa
        self.compensator.feed(*args)

    def apply(self, *args):
        """https://docs.opencv.org/master/d2/d37/classcv_1_1detail_1_1ExposureCompensator.html#a473eaf1e585804c08d77c91e004f93aa"""  # noqa
        return self.compensator.apply(*args)
