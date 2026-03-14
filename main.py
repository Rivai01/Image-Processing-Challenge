from metalg_challenge_compiled import generate_image # type: ignore
import matplotlib.pyplot as plt
from circlefinder import CircleFinder
import time

if __name__ == "__main__":
    start = time.time()
    img, params = generate_image(n_contam=5, noise_mag=10)
    finder = CircleFinder(img)
    marked_img = finder.marked_img()
    end = time.time()
    print(f"Time elapsed: {end - start:.4f} seconds")
    fig, ax = plt.subplots(1,3)
    ax[0].plot(finder.get_avg(), label='average')
    ax[0].plot(finder.get_stdev(), label='standard deviation')
    ax[0].set_title('Radial pixel values')
    ax[1].imshow(img, cmap='gray')
    ax[1].set_title('Raw image')
    ax[2].imshow(marked_img, cmap='gray')
    ax[2].set_title('Annotated image')
    plt.show()