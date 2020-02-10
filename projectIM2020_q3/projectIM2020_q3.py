import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_DB_images():
    DB_images = []

    for i in range(10):
        # print('000%d.png' %(21 + i))
        DB_images.append(cv2.imread('../projectIM2020_q2/000%d.png' % (21 + i), 0))

    return DB_images


def get_footprints():
    images = []
    for i in range(3):
        images.append(cv2.imread('00021_%d.png' %(1 + i),0))

    return images

def find_footprint(footprint, images):
    img1 = footprint  # queryImage
    for i in range(len(images)):
        img2 = images[i]  # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.imshow(img3, ), plt.show()


if __name__== "__main__":
    DB_images = get_DB_images()
    footprints = get_footprints()

    for foot in footprints:
        find_footprint(foot,DB_images)
