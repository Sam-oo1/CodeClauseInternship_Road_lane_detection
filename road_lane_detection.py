import cv2
import matplotlib.pyplot as plt
import numpy as np


# Create a function to load image
def load_image(image_path):
    """
    Reads and loads image
    :param - image_path (str):  Path of the image file
    :return: numpy.ndarray: The loaded image.
    """
    return cv2.imread(image_path)


# Convert the image into gray scale
def grayscale(image):
    """
    Converts the input image to grayscale.
    :param - image (numpy.ndarray): The input image.
    :return: numpy.ndarray: The grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Apply Gaussian Blur on image
def gaussian_blur(image):
    """
    Applies Gaussian blur to the input image.
    :param - image (numpy.ndarray): The input image.
    :return: numpy.ndarray: The image after applying Gaussian blur.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


# Detect the edges inside image
def edge_detection(image):
    """
    Applies edge detection to the input image using the Canny algorithm.
    :param - image (numpy.ndarray): The input image.
    :return: numpy.ndarray: An image highlighting detected edges.
    """
    edges = cv2.Canny(image, 50, 150)
    return edges


# Create an image mask
def region_of_interest(image):
    """
    Applies a region of interest mask to the input image.
    :param - image (numpy.ndarray): The input image.
    :return: numpy.ndarray: A masked image based on the specified region of interest.
    """
    height, width = image.shape
    triangle = np.array([
        [(100, height), (475, 325), (width, height)]
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


# Draw detected lines
def display_detected_lines(image, lines):
    """
    Draws detected lines on a blank image.
    :param - image (numpy.ndarray): The original image.
    :param - lines (numpy.ndarray or None): Detected lines represented as endpoints (x1, y1, x2, y2).
    :return: numpy.ndarray: An image with detected lines drawn.
    """
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


# Calculate the average lane lines
def average_lines(image, lines):
    """
    Calculates and returns averaged lane lines based on detected lines.
    :param - image (numpy.ndarray): The original image.
    :param - lines (numpy.ndarray or None): Detected lines represented as endpoints (x1, y1, x2, y2).
    :return: numpy.ndarray: Averaged lane lines represented as endpoints.
    """
    left = []
    right = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]
            if slope < 0:
                left.append(make_line(image, (slope, y_int)))
            else:
                right.append(make_line(image, (slope, y_int)))

    # Concatenate left and right arrays into a single array
    lines_array = np.concatenate([left, right], axis=0)

    return lines_array


# Create a line using average slope and y-intercept
def make_line(image, average):
    """
    Creates a line based on average slope and y-intercept.
    :param - image (numpy.ndarray): The original image.
    :param - average (tuple): A tuple representing the average slope and y-intercept.
    :return: numpy.ndarray: A line represented as endpoints (x1, y1, x2, y2).
    """
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2], dtype=int)

# Path of the image file
image_path = '1200px-Road_in_Norway.jpg'

original_image = load_image(image_path)
grey_image = grayscale(original_image)
blurred_image = gaussian_blur(grey_image)
edges_image = edge_detection(blurred_image)
region_image = region_of_interest(edges_image)

lines_detected = cv2.HoughLinesP(region_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_lines(original_image, lines_detected)
detected_lines_image = display_detected_lines(original_image, averaged_lines)
lanes_image = cv2.addWeighted(original_image, 0.8, detected_lines_image, 1, 1)

# List of images and their titles
images = [original_image, edges_image, region_image, detected_lines_image, lanes_image]
titles = ["Original Image", "Edge Detection", "Region of Interest", "Detected Lanes", "Original with Lanes"]

# Plotting all images in a single row with a for loop
fig, axs = plt.subplots(1, 5, figsize=(20, 5))


for i in range(len(images)):
    axs[i].imshow(images[i], cmap='gray' if i == 1 or i == 2 else None)
    axs[i].set_title(titles[i])

plt.show()
