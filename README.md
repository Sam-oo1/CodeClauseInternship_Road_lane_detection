# Road Lane Detection

This project, developed during my internship at CodeClause, focuses on predicting road lanes using opencv.


## Features

Here's the updated list of features with **Image Masking** added:

## Features

- **Grayscale Conversion:** Converts the input image to grayscale.
- **Gaussian Blur:** Applies Gaussian blur to the image to reduce noise and detail.
- **Edge Detection:** Detects edges using the Canny algorithm to highlight the edges in the image.
- **Image Masking:** Applies a mask to filter specific parts of the image, focusing on areas that are relevant for further processing.
- **Region of Interest:** Uses the masked image to focus on a specific region of interest, usually the area where lanes are expected to appear.
- **Lane Detection:** Utilizes the Hough transform on the masked and processed image to detect lines representing lanes.
- **Averaging Lane Lines:** Calculates and averages detected lane lines to form a continuous representation of the lanes.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib
- NumPy

## License

This project is licensed under the [MIT License](LICENSE).
