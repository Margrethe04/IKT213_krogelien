import cv2


def print_image_information(image):
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)


def save_camera_information():
    camera = cv2.VideoCapture(0)

    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")

    camera.release()


def main():
    image = cv2.imread("iris-1.jpg")

    print_image_information(image)
    save_camera_information()


main()