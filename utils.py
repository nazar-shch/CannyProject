import cv2


def load_image(path):
    return cv2.imread(path, 0)


def save_image(path, image):
    cv2.imwrite(path, image)


def apply_canny(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)
