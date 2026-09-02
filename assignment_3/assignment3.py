import cv2
import numpy as np


# Load images
image = cv2.imread("lambo.png")
shapes_image = cv2.imread("shapes-1.png")
template = cv2.imread("shapes_template.jpg")


def sobel_edge_detection(image):
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Sobel edge detection: dx=1, dy=1, ksize=1
    sobel_image = cv2.Sobel(
        blurred_image,
        cv2.CV_64F,
        dx=1,
        dy=1,
        ksize=1
    )

    # Convert to 8-bit image before saving
    sobel_image = cv2.convertScaleAbs(sobel_image)

    cv2.imwrite("sobel.png", sobel_image)


def canny_edge_detection(image, threshold_1, threshold_2):
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Canny edge detection
    canny_image = cv2.Canny(
        blurred_image,
        threshold_1,
        threshold_2
    )

    cv2.imwrite("canny.png", canny_image)


def template_match(image, template):
    # Both image and template must be grayscale for matching
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Template matching
    result = cv2.matchTemplate(
        gray_image,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.9

    # Find all areas with a match above the threshold
    locations = np.where(result >= threshold)

    # Template size
    h, w = gray_template.shape

    # Draw red rectangles around matches
    for point in zip(*locations[::-1]):
        cv2.rectangle(
            image,
            point,
            (point[0] + w, point[1] + h),
            (0, 0, 255),
            2
        )

    cv2.imwrite("template_match.png", image)


def resize(image, scale_factor: int, up_or_down: str):
    resized_image = image.copy()

    if up_or_down == "up":
        # pyrUp doubles the image size
        if scale_factor == 2:
            resized_image = cv2.pyrUp(resized_image)

    elif up_or_down == "down":
        # pyrDown halves the image size
        if scale_factor == 2:
            resized_image = cv2.pyrDown(resized_image)

    cv2.imwrite("resized_" + up_or_down + ".png", resized_image)


# Run all functions
sobel_edge_detection(image)

canny_edge_detection(
    image,
    threshold_1=50,
    threshold_2=50
)

template_match(
    shapes_image,
    template
)

resize(
    image,
    scale_factor=2,
    up_or_down="up"
)

resize(
    image,
    scale_factor=2,
    up_or_down="down"
)