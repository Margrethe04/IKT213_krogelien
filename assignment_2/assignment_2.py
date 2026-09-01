import cv2
import numpy as np

image = cv2.imread("iris-1.png")


def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )

    cv2.imwrite("padding.png", padded_image)

    return padded_image


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]

    cv2.imwrite("cropped.png", cropped_image)

    return cropped_image


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))

    cv2.imwrite("resized.png", resized_image)

    return resized_image


def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]

    cv2.imwrite("copied.png", emptyPictureArray)

    return emptyPictureArray


def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("grayscale.png", gray_image)

    return gray_image


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imwrite("hsv.png", hsv_image)

    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_value = int(image[y, x, c]) + hue

                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0

                emptyPictureArray[y, x, c] = new_value

    cv2.imwrite("hue_shifted.png", emptyPictureArray)

    return emptyPictureArray


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(
        image,
        (15, 15),
        0,
        borderType=cv2.BORDER_DEFAULT
    )

    cv2.imwrite("smoothed.png", smoothed_image)

    return smoothed_image


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    else:
        return image

    cv2.imwrite("rotated_" + str(rotation_angle) + ".png", rotated_image)

    return rotated_image


height, width, channels = image.shape

emptyPictureArray = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

padding(image, 100)

crop(
    image,
    200,
    width - 130,
    200,
    height - 130
)

resize(image, 200, 200)

copy(image, emptyPictureArray)

grayscale(image)

hsv(image)

huePictureArray = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

hue_shifted(image, huePictureArray, 50)

smoothing(image)

rotation(image, 180)