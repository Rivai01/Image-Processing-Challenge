# Metrology Algorithms Coding Challenge

This coding challenge will test your ability to create a data processing algorithm that is robust across a variety of conditions, including noise and contaminations. The challenge involves generating an image using the metalg_challenge.pyc API, and extracting specific features from this image. The image has randomized components, requiring the algorithm to be robust to handle these varying conditions.

## Challenge

The challenge will involve an image that will look something like the following:

![Example image](images/example_image_annotated.png)

This image will be generate by the provided [metalg_challenge_compiled.pyc](metalg_challenge_compiled.pyc) API. Instructions for how to use this API are given below.

### Objectives

1. Detect the center point of the image's waveform
2. Detect the positions of each of the circular "fringes" propagating from the center
3. Model the distribution of these fringes based on their distance away from the center
4. Present the results in the following format:
    - The image with the center position and fringe rings marked with colors (see below for example)
    - The graph of the fringe distribution (X=fringe number, Y=fringe distance from center)
    - Include the **seed** that was used for these specific results, so that it can be confirmed (see below for details on the randomized seed)

You are also expected to demonstrate:
 - Clear code documentation
 - Line of thinking, reasoning, and approach to solve the problem

### Challenge Levels

The API allows for customization of the difficulty of the challenge. The noise and the contaminations can be completely turned off. Therefore, there are 3 challenge levels that can be attempted (plus a Level 0 prerequisite). Levels 2 and 3 are considered **stretch goals** for this challenge.

- Level 0: (Documentation) Outline plan of approach to solving the problem
    - Ideas to try
    - What ideas did/didn't work and why
    - Areas of improvement
- Level 1: Image with no noise or contaminations
- Level 2: Image with noise (magnitude 10)
- Level 3: Image with noise (magnitude 10) and contaminations (minimum 5)

### Tips

- Start with **Level 1** challenge (no noise/contaminations) first, and progress from there once it has been completed.
- Start by querying the image generation multiple times to get a gauge for how the images vary based on the randomized parameters.
- Select a single seed and develop the first-draft algorithm based on that image.
- Test periodically with different images and noise-ON to test how robust the algorithm is to different imaging conditions.

### Example Solution Image

Image with center and fringes annotated.

![Example image solution](images/example_image_solution_annotated.png)

## Environment and API

To ensure compatibility with the provided API, please use Python version 3.11. You can download Python 3.11 from the official website [here](https://www.python.org/downloads/release/python-3119/).

Before starting, set up your environment using the provided `requirements.txt` file. Install dependencies with:

```bash
pip install -r requirements.txt
```

### Using the API

The image generation function is provided in the compiled file `metalg_challenge_compiled.pyc`. To use the API, import the function as follows:

```python
from metalg_challenge_compiled import generate_image
```

#### Function: `generate_image`

**Arguments:**
- `seed` (int, optional): Seed for randomization. If `None`, a random seed is used.
- `n_contam` (int, optional): Number of contaminations to apply to the image.
- `noise_mag` (int, optional): Magnitude of noise to add to the image.

**Returns:**
- `image` (`np.ndarray`): The generated image (grayscale, shape [1000, 1000], dtype uint8).
- `params` (`dict`): Parameters used to generate the image:
    - `x_center` (int): X coordinate of the image center
    - `y_center` (int): Y coordinate of the image center
    - `center_amp` (float): Amplitude of the central Gaussian
    - `center_wid` (float): Width of the central Gaussian
    - `fringe_scale` (float): Scale factor for fringe positions
    - `fringe_amp` (float): Amplitude of the fringes
    - `fringe_wid` (float): Width of the fringes
    - `slope_drop` (int): Intensity drop per width for the down slope
    - `seed` (int): Seed used for randomization

Example usage:

```python
from metalg_challenge_compiled import generate_image

image, params = generate_image(seed=12345, n_contam=5, noise_mag=10)
```

This will generate a new image and return the parameters used for its creation.

