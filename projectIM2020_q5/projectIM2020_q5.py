import cv2
import numpy as np
from matplotlib import pyplot as plt
import random



def get_DB_images():
    DB_images = []

    for i in range(10):
        # print('000%d.png' %(21 + i))
        DB_images.append(cv2.imread('../projectIM2020_q2/000%d.png' % (21 + i), 0))

    return DB_images


def get_footprints():
    images = []
    images.append(cv2.imread('../projectIM2020_q4/00008.jpg' ,0))

    return images

def plot_results(small_pic,big_pic):
    finalImage = np.zeros((big_pic.shape[0],big_pic.shape[1]+small_pic.shape[1]))

    print(finalImage.shape)
    print(small_pic.shape)
    print(big_pic.shape)
    finalImage[0:big_pic.shape[0], 0:big_pic.shape[1]] += big_pic
    finalImage[0:small_pic.shape[0], big_pic.shape[1]:big_pic.shape[1]+small_pic.shape[1]] += small_pic
    plt.imshow(finalImage, 'gray')
    # plt.imshow(footprint,'gray')
    plt.show()

def find_match_footprint(footprint, images):
    img1 = footprint  # queryImage
    maxGreens = 0
    res = []
    for j in range(len(images)):
        img2 = images[j]  # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        pts1 = cv2.KeyPoint.convert(kp1)
        pts2 = cv2.KeyPoint.convert(kp2)


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
        # print(j,draw_params)

        # count the green keypoints (matches)
        cntGreens = 0
        for mask in draw_params['matchesMask']:
            # if mask == [1,0] the keypoint is green
            if mask == [1,0]:
                cntGreens += 1
        if cntGreens > maxGreens:
            maxGreens = cntGreens
            res = img2

        print('cntGreen    ',cntGreens, 'maxGreens    ', maxGreens)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.imshow(img3)
        plt.title((j))
        plt.show()



    plot_results(footprint,res)



if __name__== "__main__":
    DB_images = get_DB_images()

    random.shuffle(DB_images)
    footprints = get_footprints()
    for foot in footprints:
        find_match_footprint(foot,DB_images)
