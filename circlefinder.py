import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

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
    
    def get_center(self):
        """
        Returns the coordinates of the center of `self.img`.
        
        Currently finds the brightest pixel, which is susceptible to noise and contamination, and
        specific to the Metrology Algorithms Challenge images.
        """
        if 'center' in self.__dict__:       # check if already initialized
            return self.center
        
        else:                               # initialize
            # Take median blur to counteract blotchy contamination
            dst = cv.medianBlur(self.img, 39)

            # Remove background larger than central blur to leave just central Gaussian
            bg = cv.blur(dst, (301, 301))
            dst_sub = cv.subtract(dst, bg)

            # Find brightest point on central Gaussian
            return cv.minMaxLoc(dst_sub)[3]
    
    def get_circles(self):
        """Returns an array containing the radii (in pixels) of the detected circles."""
        if 'radii' in self.__dict__:        # check if already initialized
            return self.radii 
        
        else:                               # initialize
            # find radii of minimum brightness
            # distance = 20 and prominence = 3 are tuned parameters for this specific problem
            self.radii, _ = find_peaks(-self.avg_rem_outliers, distance=20, prominence=3)

            # don't count fringes at the center point if they get detected
            if self.radii[0] < 10:
                self.radii = self.radii[1:]

            # don't count fringes near the farmost corner; too little data to reliably detect
            while self.radii[-1] > len(self.avg_rem_outliers) - 25: 
                self.radii = self.radii[:-1]

            return self.radii
    
    def marked_img(self):
        """Returns a color image with the center and concentric circles marked."""
        new_img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)

        # draw fringes
        radii = self.get_circles()
        for radius in radii:
            cv.circle(new_img, self.get_center(), radius, (0, 255, 0), 3)

        # draw center
        cv.circle(new_img, self.get_center(), 10, (255, 0, 0), -1)

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
        # sum of squares per radius
        sum_sq = np.bincount(inv, weights=np.square(img_flat))

        # compute average and standard deviation
        avg = sum_vals / counts
        stdev = np.sqrt(np.maximum(0.0, sum_sq / counts - np.square(avg)))

        # recompute average without outliers (to filter out blotchy contamination)
        dev = np.abs(img_flat - avg[inv])
        mask = dev < stdev[inv]
        sum_vals_noout = np.bincount(inv[mask], weights=img_flat[mask], minlength=len(counts))
        counts_noout = np.bincount(inv[mask], minlength=len(counts))
        avg_noout = np.divide(sum_vals_noout, counts_noout, where=counts_noout > 0)

        # set object attributes
        self.avg = avg
        self.stdev = stdev
        self.avg_rem_outliers = avg_noout

