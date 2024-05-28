import cv2
import numpy as np
import scipy.ndimage


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(path, image):
    cv2.imwrite(path, image)


def gaussian_blur(image, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def gradient_intensity(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = scipy.ndimage.convolve(image, Kx, mode='reflect')
    Iy = scipy.ndimage.convolve(image, Ky, mode='reflect')

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j + 1]
                    r = G[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = G[i + 1, j - 1]
                    r = G[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = G[i + 1, j]
                    r = G[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = G[i - 1, j - 1]
                    r = G[i + 1, j + 1]

                if (G[i, j] >= q) and (G[i, j] >= r):
                    Z[i, j] = G[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z


def threshold(image, lowThreshold, highThreshold):
    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def apply_canny(image, lowThreshold, highThreshold):
    blurred_image = gaussian_blur(image)
    G, theta = gradient_intensity(blurred_image)
    non_max_img = non_maximum_suppression(G, theta)
    threshold_img, weak, strong = threshold(non_max_img, lowThreshold, highThreshold)
    img_final = hysteresis(threshold_img, weak, strong)
    return img_final
