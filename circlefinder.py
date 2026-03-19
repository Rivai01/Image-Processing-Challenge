"""
Backend code for detection.

Run ``find_circles.py`` from the command line if you are looking to run this on an image.

By Levi Hauser, 20-March-2026
"""

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

class CircleFinder:
    """Class to find the center and concentric circular rings of images generated 
    for the Metrology Algorithms Challenge. Create an instance for each image to be processed."""

    def __init__(self, img, noisy=False):
        """
        Initializes a`CircleFinder` object.

        :param img: the grayscale image to be analyzed
        :param noisy: uses noise- and contamination-combating techniques; do not use 
        unless the image is noisy
        """
        if img.ndim != 2:
            raise ValueError("Expected img to be a grayscale image!")
        self.img = img                      # set image
        self.noisy = noisy                  # determine whether using noise-combating techniques
        self.center = self.get_center()     # find the image's center
        self._init_radial_profile()         # find radial patterns

        
    def get_img(self):
        """Returns the image being analyzed."""
        return self.img
    
    def get_avg(self):
        """
        Returns an array containing the average pixel values at each radius.
        
        ``self.get_avg()[i]`` is the average of the values at a distance of `i` pixels from the center.
        """
        return self.avg
    
    def get_stdev(self):
        """
        Returns an array containing the standard deviation of the pixel values at each radius. Only available if 
        the object was initialized with ``noisy=True``.
        
        ``self.get_stdev()[i]`` is the standard deviation of the values at a distance of `i` pixels from the center.
        """
        if 'stdev' in self.__dict__:
            return self.stdev
        else:
            raise AttributeError('stdev is only computed if self.noisy=True!')
    
    def get_avg_rem_outliers(self):
        """
        Returns an array containing the average pixel values at each radius, with outliers removed. Only available if 
        the object was initialized with ``noisy=True``.
        
        ``self.get_avg_rem_outliers()[i]`` is the average of the values at a distance of `i` pixels from the center.
        """
        if 'avg_rem_outliers' in self.__dict__:
            return self.avg_rem_outliers
        else:
            raise AttributeError('avg_rem_outliers is only computed if self.noisy=True!')
    
    def get_center(self):
        """
        Returns the coordinates of the center of `self.img`.
        
        Currently finds the brightest pixel, which is susceptible to noise and contamination, and
        specific to the Metrology Algorithms Challenge images.
        """
        if 'center' in self.__dict__:
            return self.center
        else:
            if self.noisy:
                # Take median blur to counteract blotchy contamination
                dst = cv.medianBlur(self.img, 39)

                # Remove background larger than central blur to leave just central Gaussian
                bg = cv.blur(dst, (301, 301))
                dst_sub = cv.subtract(dst, bg)

                # Find brightest point on central Gaussian
                return cv.minMaxLoc(dst_sub)[3]
            else:
                # Don't need to do extra filtering unless the image is noisy
                return cv.minMaxLoc(self.img)[3]
    
    def get_circles(self):
        """Returns an array containing the radii (in pixels) of the detected circles."""
        if 'radii' in self.__dict__:
            return self.radii 
        
        else:
            # find radii of minimum brightness
            # only use average without outliers if noisy; otherwise stdev=0 and all points removed -> signal mismatch
            if self.noisy:
                # distance = 20 and prominence = 3 are tuned parameters for this specific problem
                self.radii, _ = find_peaks(-self.get_avg_rem_outliers(), distance=20, prominence=3)
                # don't count fringes at the center point if they get detected
                if self.radii[0] < 10:
                    self.radii = self.radii[1:]

                # don't count fringes near the farmost corner; too little data to reliably detect
                while self.radii[-1] > len(self.get_avg()) - 25: 
                    self.radii = self.radii[:-1]
            else:
                self.radii, _ = find_peaks(-self.get_avg(), distance=20, prominence=1)

            return self.radii
    
    def marked_img(self):
        """Returns a color image with the center and concentric circles marked."""
        new_img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)

        # draw fringes
        radii = self.get_circles()
        for radius in radii:
            cv.circle(new_img, self.get_center(), radius, (0, 255, 0), 3)

        # draw center
        cv.circle(new_img, self.get_center(), 10, (0, 0, 255), -1)

        return new_img
    
    def quad_regression(self):
        """
        Run a quadratic regression on the fringe radii, and return [`c`, `b`, `a`],
        where the modeled radius `r` at fringe number `n` is `r = an^2 + bn + c`.
        """
        n = np.arange(len(self.get_circles()))
        coef = np.polynomial.polynomial.polyfit(n, self.get_circles(), deg=2)
        return coef

    def _init_radial_profile(self):
        """
        Creates and stores the average and standard deviation of all pixel values at each radius.
        """
        h, w = self.img.shape
        y, x = np.ogrid[:h, :w]
        cx, cy = self.get_center()
        xdist = x - cx
        ydist = y - cy

        # calculate radii (in pixels) of each point in the image
        r = np.sqrt(xdist * xdist + ydist * ydist).astype(np.int32)

        # blur to remove Gaussian noise
        blurred = cv.blur(self.img, (10, 10))

        # flatten for bincount
        r_flat = r.ravel()
        img_flat = blurred.ravel().astype(np.int32)

        # sort pixels by radii
        _, inv, counts = np.unique(r_flat, return_inverse=True, return_counts=True)
        # sum of values per radius
        sum_vals = np.bincount(inv, weights=img_flat)

        # compute average
        avg = sum_vals / counts

        self.avg = avg

        if self.noisy: # only need to remove outliers if noisy
            # sum of squares per radius
            sum_sq = np.bincount(inv, weights=np.square(img_flat))

            # compute standard deviation
            stdev = np.sqrt(np.maximum(0.0, sum_sq / counts - np.square(avg)))

            # recompute average without outliers (to filter out blotchy contamination)
            dev = np.abs(img_flat - avg[inv])
            mask = dev < stdev[inv]
            sum_vals_noout = np.bincount(inv[mask], weights=img_flat[mask], minlength=len(counts))
            counts_noout = np.bincount(inv[mask], minlength=len(counts))
            avg_noout = np.divide(sum_vals_noout, counts_noout, where=counts_noout > 0)
        
            self.stdev = stdev
            self.avg_rem_outliers = avg_noout

