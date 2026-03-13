import cv2 as cv
import numpy as np
from scipy.signal import argrelmin

class CircleFinder:
    """Class to find the center and concentric circular rings of images generated 
    for the Metrology Algorithms Challenge."""

    def __init__(self, img):
        """
        Initialize a`CircleFinder` object.

        :param img: the grayscale image to be analyzed
        """
        if img.ndim != 2:
            raise ValueError("Expected img to be a grayscale image!")
        self.img = img                      # set image
        self.center = self.get_center()     # find the image's center
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
    
    def get_circles(self):
        try:
            return self.radii 
        except AttributeError:
            self.radii = argrelmin(self.avg, order=5)[0]
            return self.radii
    
    def marked_img(self):
        new_img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        radii = self.get_circles()
        for radius in radii:
            cv.circle(new_img, self.center, radius, (0, 255, 0), 3)
        cv.circle(new_img, self.center, 10, (255, 0, 0), -1)
        return new_img

    def _init_radial_profile(self):
        """
        Create and store the average and standard deviation of all pixel values at each radius.
        """
        h, w = self.img.shape
        y, x = np.ogrid[:h, :w]
        cx, cy = self.center
        xdist = x - cx
        ydist = y - cy
        # radii in pixels
        r = np.sqrt(xdist * xdist + ydist * ydist).astype(np.int32)

        r_flat = r.ravel()
        img_flat = self.img.ravel()
        _, inv, counts = np.unique(r_flat, return_inverse=True, return_counts=True)
        # sum of values per radius
        sum_vals = np.bincount(inv, weights=img_flat)
        # sum of squares per radius
        sum_sq = np.bincount(inv, weights=np.square(img_flat))
        avg = sum_vals / counts
        stdev = np.sqrt(sum_sq / counts - np.square(avg))
        
        self.avg = avg
        self.stdev = stdev
