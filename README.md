# Road Lane Detection

This project, developed during my internship at CodeClause, focuses on predicting road lanes using opencv.


## Features

- **Grayscale Conversion:** Converts the input image to grayscale.
- **Gaussian Blur:** Applies Gaussian blur to the image.
- **Edge Detection:** Detects edges using the Canny algorithm.
- **Region of Interest:** Applies a mask to focus on a specific region of interest.
- **Lane Detection:** Utilizes Hough transform to detect lines representing lanes.
- **Averaging Lane Lines:** Calculates averaged lane lines based on detected lines.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib
- NumPy

## License

This project is licensed under the [MIT License](LICENSE).
