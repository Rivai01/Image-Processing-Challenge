import cv2 as cv
import numpy as np

class CircleFinder:
    """Class to find the center and concentric circular rings of images generated 
    for the Metrology Algorithms Challenge."""

    def __init__(self, img):
        """
        Initialize a`CircleFinder` object.

        :param img: the image to be analyzed
        """
        self.img = img                      # set image
        self.center = self.get_center()     # find the image's center
        self._init_dist_map()               # initialize distance map
        self._init_radial_profile()         # find radial patterns
    
    def get_center(self):
        """
        Returns the coordinates of the center of `self.img`.
        
        Currently finds the brightest pixel, which is susceptible to noise and contamination, and
        specific to the Metrology Algorithms Challenge images.
        """
        try:                    # check if the center has already been initialized
            return self.center
        except AttributeError:  # need to initialize the center
            return cv.minMaxLoc(self.img)[3]
        
    def get_img(self):
        """Returns the image being analyzed."""
        return self.img
    
    def get_avg(self):
        """
        Return an array containing the average pixel values at each radius.
        
        ``self.get_avg()[i]`` is the average of the values at a distance of `i` pixels from the center.
        """
        return self.avg
    
    def get_stdev(self):
        """
        Return an array containing the standard deviation of the pixel values at each radius.
        
        ``self.get_stdev()[i]`` is the standard deviation of the values at a distance of `i` pixels from the center.
        """
        return self.stdev

    def get_ring_pixels(self, radius, width=1):
        """
        Return a 1D array containing the pixels at radius ``radius`` from ``self.center``.

        :param radius: The (major) radius at which to select pixels
        :param width: The minor radius of the annulus from which to select pixels around ``radius``
        """
        mask = np.abs(self.map - radius) <= width / 2
        return self.img[mask]
    
    def _init_dist_map(self):
        """
        Create and store a radial distance map over the image, centered at ``self.center``.
        """
        h, w = self.img.shape[:2]
        cx, cy = self.center

        y, x = np.ogrid[:h, :w]
        xdist = x - cx
        ydist = y - cy
        self.map = np.sqrt(xdist * xdist + ydist * ydist)

    def _init_radial_profile(self):
        """
        Create and store the average and standard deviation of all pixel values at each radius.
        """
        radius = 0
        avg = []
        stdev = []
        while True:
            ring = self.get_ring_pixels(radius, 1)
            if len(ring) == 0:
                break
            avg.append(np.mean(ring))
            stdev.append(np.std(ring))
            radius += 1
        self.avg, self.stdev =  np.array(avg), np.array(stdev)
