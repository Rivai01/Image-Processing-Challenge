# Metrology Algorithms Image Processing Challenge Solution
### By Levi Hauser

<br>

This is my solution to the 
[Metrology Algorithms Image Processing Challenge](https://github.com/MetrologyAlgorithmsTeam/Image-Processing-Challenge).

*\[Note: This README provides an overall guide to the code. For my thought process and implementation details,
see my [process outline](process_outline.pdf).\]*

<br>
<br>

There are four relevant files:

* [`find_circles.py`](find_circles.py): Command-line interface to generate and process images. Run `python find_circles.py --help` for usage details.

* [`circlefinder.py`](circlefinder.py): Algorithm code for the image processing.

* [`main.py`](main.py): Older file used to generate and process images, with flexibility to modify the code. I recommend using [`find_circles.py`](find_circles.py) instead.

* [`process_outline.pdf`](process_outline.pdf): Detailed discussion of the implementation, sample results, and my thought process.

<br>
<br>

If you're looking for a starting point, I recommend trying out the following commands:

To run a noise-free detection on a random seed:

    python find_circles.py

To run a detection on seed 1 with 5 contaminations and noise magnitude 10, using noise/contamination mitigation:

    python find_circles.py --seed 1000 --n-contam 5 --noise-mag 10 --noisy


