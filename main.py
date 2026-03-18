from metalg_challenge_compiled import generate_image # type: ignore
import matplotlib.pyplot as plt
from circlefinder import CircleFinder
import time
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img, params = generate_image(n_contam=50, noise_mag=100)
    print("Seed: ", params.get("seed"))

    # time algorithm
    start = time.time()

    finder = CircleFinder(img)
    circles = finder.get_circles()
    coef = finder.quad_regression()

    end = time.time()
    print(f"Regression: r = {coef[2]:.1f}n^2 + {coef[1]:.1f}n + {coef[0]:.1f}")
    print(f"Algorithm run time: {end - start:.4f} seconds")
    
    marked_img = finder.marked_img()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2BGR), cmap='gray')
    ax[0].set_title('Raw image')

    ax[1].imshow(marked_img, cmap='gray')
    ax[1].set_title('Annotated image')

    ax[2].plot(circles, 'o', label='Detected')

    smooth_n = np.linspace(0, len(circles), 100)
    reg = coef[0] + coef[1] * smooth_n + coef[2] * np.square(smooth_n)
    ax[2].plot(smooth_n, reg, label='Regression')
    ax[2].legend()

    ax[2].set_title('Circle radii')
    ax[2].set_xlabel('Circle number')
    ax[2].set_ylabel('Radius (pixels)')

    fig.suptitle('Seed = ' + str(params.get("seed")))

    plt.show()