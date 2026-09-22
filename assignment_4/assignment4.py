import cv2
import numpy as np


def harris_corner_detection(reference_image):
    image = cv2.imread(reference_image)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {reference_image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    result = image.copy()
    result[corners > 0.01 * corners.max()] = [0, 0, 255]

    cv2.imwrite("harris.png", result)

    return result


def align_images(image_to_align, reference_image, max_features, good_match_precent):
    image1 = cv2.imread(image_to_align)
    image2 = cv2.imread(reference_image)

    if image1 is None or image2 is None:
        raise FileNotFoundError("Could not read one of the images.")

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not find enough features.")

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:max_features]

    print("Good matches found:", len(good_matches))

    if len(good_matches) < 4:
        raise ValueError("Not enough matches to align the images.")

    points1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    points2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        raise ValueError("Could not calculate homography.")

    height, width = image2.shape[:2]

    aligned = cv2.warpPerspective(
        image1,
        homography,
        (width, height)
    )

    cv2.imwrite("aligned.png", aligned)

    match_mask = mask.ravel().tolist()

    matches_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        good_matches,
        None,
        matchesMask=match_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matches_image = cv2.resize(
        matches_image,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite("matches.png", matches_image)

    return aligned, matches_image


if __name__ == "__main__":
    reference_image = "reference_img.png"
    image_to_align = "align_this.jpg"

    harris_corner_detection(reference_image)

    align_images(
        image_to_align,
        reference_image,
        10,
        0.7
    )

    print("Assignment 4 completed.")