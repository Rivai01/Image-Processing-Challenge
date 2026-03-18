from metalg_challenge_compiled import generate_image # type: ignore
import matplotlib.pyplot as plt
from circlefinder import CircleFinder
import time
import cv2 as cv

if __name__ == "__main__":
    img, params = generate_image(n_contam=5, noise_mag=10)
    print("Seed: ", params.get("seed"))
    start = time.time()
    finder = CircleFinder(img)
    circles = finder.get_circles()
    end = time.time()
    print(f"Algorithm run time: {end - start:.4f} seconds")
    marked_img = finder.marked_img()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2BGR), cmap='gray')
    ax[0].set_title('Raw image')
    ax[1].imshow(marked_img, cmap='gray')
    ax[1].set_title('Annotated image')
    ax[2].plot(circles, 'o')
    ax[2].set_title('Circle radii')
    ax[2].set_xlabel('Circle number')
    ax[2].set_ylabel('Radius (pixels)')
    # ax[3].plot(finder.get_avg(), label='average')
    # ax[3].plot(finder.get_avg_rem_outliers(), label='average w/o outliers')
    # ax[3].legend()
    # ax[3].set_title('Radial pixel values')
    fig.suptitle('Seed = ' + str(params.get("seed")))
    plt.show()