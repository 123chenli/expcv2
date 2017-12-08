import cv2
import numpy
import utils


def recolorRC(src, dst):
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


def blend(foregroundSrc, backgraoundSrc, dst, alphaMask):
    maxAlpha = numpy.iinfo(alphaMask.dtype).max
    normalizedAlphaMask = (1.0 / maxAlpha) * alphaMask
    normalizedInverseAlphaMask = numpy.ones_like(normalizedAlphaMask)
    normalizedInverseAlphaMask[:] = normalizedInverseAlphaMask - normalizedAlphaMask
    foregroundChannels = cv2.split(foregroundSrc)
    backgraoundChannels = cv2.split(backgraoundSrc)
    numChannels = len(foregroundChannels)
    i = 0
    while i < numChannels:
        backgraoundChannels[i][:] = normalizedAlphaMask * foregroundChannels[i]
        + normalizedInverseAlphaMask * backgraoundChannels
    i += 1
    cv2.merge(backgraoundChannels, dst)


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=numpy.uint0):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        srcFlatView = utils.flatView(src)
        dstFlatView = utils.flatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)


class BGRFuncFilter(object):
    def __init__(self, vFunc=None, bFunc=None, gFunc=None,
                 rFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositerFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositerFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositerFunc(rFunc, vFunc)
        )

    def apply(self, src, dst):
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCuveFunc(vPoints),
                               utils.createCuveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(self,
                                bPoints=[(0, 20), (255, 235)],
                                gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
                                rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
                                dtype=dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype=dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
            gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
            rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
            dtype=dtype
        )


class BGRVelviaCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
            bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
            dtype=dtype
        )


class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)