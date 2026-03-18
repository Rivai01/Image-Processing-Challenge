import cv2 as cv
import numpy as np
from scipy.signal import argrelmin

class CircleFinder:
    """Class to find the center and concentric circular rings of images generated 
    for the Metrology Algorithms Challenge."""

    def __init__(self, img):
        """
        Initializes a`CircleFinder` object.

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
        except AttributeError:  # initialize the center
            # Take median blur to counteract blotchy contamination
            dst = cv.medianBlur(self.img, 39)
            # Remove background larger than central blur to leave just central Gaussian
            bg = cv.blur(dst, (301, 301))
            dst_sub = cv.subtract(dst, bg)
            # Find brightest point on central Gaussian
            return cv.minMaxLoc(dst_sub)[3]
        
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
        Returns an array containing the standard deviation of the pixel values at each radius.
        
        ``self.get_stdev()[i]`` is the standard deviation of the values at a distance of `i` pixels from the center.
        """
        return self.stdev
    
    def get_avg_rem_outliers(self):
        """
        Returns an array containing the average pixel values at each radius, with outliers removed.
        
        ``self.get_avg_rem_outliers()[i]`` is the average of the values at a distance of `i` pixels from the center.
        """
        return self.avg_rem_outliers
    

    def get_ring_pixels(self, radius, width=1):
        """
        Returns a 1D array containing the pixels at radius ``radius`` from ``self.center``.

        :param radius: The (major) radius at which to select pixels
        :param width: The minor radius of the annulus from which to select pixels around ``radius``
        """
        mask = np.abs(self.map - radius) <= width / 2
        return self.img[mask]
    
    def get_circles(self):
        """Returns an array containing the radii (in pixels) of the detected circles."""
        try:
            return self.radii 
        except AttributeError:
            # Typical smallest separation between circles is around 20 pixels
            self.radii = argrelmin(self.avg_rem_outliers, order=20)[0] 
            return self.radii
    
    def marked_img(self):
        """Returns a color image with the center and concentric circles marked."""
        new_img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        radii = self.get_circles()
        for radius in radii:
            cv.circle(new_img, self.center, radius, (0, 255, 0), 3)
        cv.circle(new_img, self.center, 10, (255, 0, 0), -1)
        return new_img

    def _init_radial_profile(self):
        """
        Creates and stores the average and standard deviation of all pixel values at each radius.
        """
        h, w = self.img.shape
        y, x = np.ogrid[:h, :w]
        cx, cy = self.center
        xdist = x - cx
        ydist = y - cy
        # radii in pixels
        r = np.sqrt(xdist * xdist + ydist * ydist).astype(np.int32)

        r_flat = r.ravel()
        # Take median blur to counteract blotchy contamination
        img_flat = self.img.ravel().astype(np.int32)
        _, inv, counts = np.unique(r_flat, return_inverse=True, return_counts=True)
        # sum of values per radius
        sum_vals = np.bincount(inv, weights=img_flat)
        # sum of squares per radius
        sum_sq = np.bincount(inv, weights=np.square(img_flat))
        avg = sum_vals / counts
        stdev = np.sqrt(np.maximum(0.0, sum_sq / counts - np.square(avg)))

        # compute average without outliers (to filter out contamination)
        dev = np.abs(img_flat - avg[inv])
        mask = dev < 2 * stdev[inv]
        sum_vals_noout = np.bincount(inv[mask], weights=img_flat[mask], minlength=len(counts))
        counts_noout = np.bincount(inv[mask], minlength=len(counts))
        avg_noout = np.divide(sum_vals_noout, counts_noout, where=counts_noout > 0)

        self.avg = avg
        self.stdev = stdev
        self.avg_rem_outliers = avg_noout

