from metalg_challenge_compiled import generate_image # type: ignore
import matplotlib.pyplot as plt
from circlefinder import CircleFinder
import time

if __name__ == "__main__":
    img, params = generate_image(n_contam=50, noise_mag=10)
    print("Seed: ", params.get("seed"))
    start = time.time()
    finder = CircleFinder(img)
    marked_img = finder.marked_img()
    end = time.time()
    print(f"Algorithm run time: {end - start:.4f} seconds")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Raw image')
    ax[1].imshow(marked_img, cmap='gray')
    ax[1].set_title('Annotated image')
    # ax[2].plot(finder.get_avg(), label='average')
    # ax[2].plot(finder.get_avg_rem_outliers(), label='average w/o outliers')
    # ax[2].legend()
    # ax[2].set_title('Radial pixel values')
    fig.suptitle('Seed = ' + str(params.get("seed")))
    plt.show()