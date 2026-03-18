"""
Command line application to detect central point and fringes of images.

Run ``python find_circles.py --help`` for help.

By Levi Hauser, 20-March-2026
"""

import argparse
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from metalg_challenge_compiled import generate_image  # type: ignore
from circlefinder import CircleFinder


def run_single(n_contam, noise_mag, seed=None, save_path=None, show=True):
    # Generate the image
    img, params = generate_image(
        n_contam=n_contam,
        noise_mag=noise_mag,
        seed=seed
    )

    print("Seed:", params.get("seed"))

    # Run (and time) the detection code
    start = time.time()

    finder = CircleFinder(img)
    circles = finder.get_circles()
    coef = finder.quad_regression()

    end = time.time()

    print(f"Regression: r = {coef[2]:.1f}n^2 + {coef[1]:.1f}n + {coef[0]:.1f}")
    print(f"Detection Runtime: {end - start:.4f} seconds")

    marked_img = finder.marked_img()

    # Make plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    ax[0].set_title('Original image')

    ax[1].imshow(cv.cvtColor(marked_img, cv.COLOR_BGR2RGB))
    ax[1].set_title('Annotated image')

    ax[2].plot(circles, 'o', label='Detected')

    smooth_n = np.linspace(0, len(circles), 100)
    reg = coef[0] + coef[1] * smooth_n + coef[2] * np.square(smooth_n)
    ax[2].plot(smooth_n, reg, label='Regression')
    ax[2].legend()

    ax[2].set_title('Circle radii')
    ax[2].set_xlabel('Circle number')
    ax[2].set_ylabel('Radius (pixels)')

    fig.suptitle(f"Seed = {params.get('seed')}")

    if save_path:
        # Save whole figure as well as individual images. 4 files total.
        plt.savefig(save_path)

        orig_path = save_path.replace(".png", "_original.png")
        cv.imwrite(orig_path, img)

        annotated_path = save_path.replace(".png", "_annotated.png")
        cv.imwrite(annotated_path, marked_img)

        plt.figure(2)
        plt.plot(circles, 'o', label='Detected')
        plt.plot(smooth_n, reg, label='Regression')
        plt.legend()
        plt.title('Circle radii')
        plt.xlabel('Circle number')
        plt.ylabel('Radius (pixels)')
        plt.savefig(save_path.replace(".png", "_regression.png"))
        plt.close()

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Circle detection")

    parser.add_argument("--n-contam", type=int, default=0,
                        help="Number of contaminants")
    parser.add_argument("--noise-mag", type=int, default=0,
                        help="Noise magnitude")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to process")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save output images")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable displaying plots")

    args = parser.parse_args()

    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for i in range(args.num_images):
        print(f"\n--- Image {i+1}/{args.num_images} ---")

        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"result_{i}.png")

        run_single(
            n_contam=args.n_contam,
            noise_mag=args.noise_mag,
            seed=args.seed,
            save_path=save_path,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()