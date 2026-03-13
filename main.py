from metalg_challenge_compiled import generate_image # type: ignore
import matplotlib.pyplot as plt
from circlefinder import CircleFinder

if __name__ == "__main__":
    img, params = generate_image(n_contam=0, noise_mag=0)
    finder = CircleFinder(img)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(finder.get_avg(), label='average')
    ax[0].plot(finder.get_stdev(), label='standard deviation')
    ax[1].imshow(img, cmap='gray')
    plt.show()