import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__== "__main__":
    images = []
    blur_images = []
    cimages = []
    # Elliptical Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for i in range(10):
        # print('000%d.png' %(21 + i))
        images.append(cv2.imread('000%d.png' %(21 + i),0))
        blur_images.append(cv2.medianBlur(images[i], 5))
        blur_images[i] = cv2.morphologyEx(blur_images[i],cv2.MORPH_CLOSE,kernel)
        blur_images[i] = cv2.morphologyEx(blur_images[i],cv2.MORPH_OPEN,kernel)

        cimages.append(cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR))
        circles = cv2.HoughCircles(blur_images[i], cv2.HOUGH_GRADIENT, 1, 75,
                                   param1=100, param2=30, minRadius=0, maxRadius=55)

        print(circles)
        #  ------------------  Best Parameters so far ---------------
        # circles = cv2.HoughCircles(blur_images[i], cv2.HOUGH_GRADIENT, 1, 75,
        #                            param1=225, param2=100, minRadius=0, maxRadius=0)
        #  ------------------  Best Parameters so far ---------------

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for j in circles[0, :]:
                # draw the outer circle
                cv2.circle(cimages[i], (j[0], j[1]), j[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(cimages[i], (j[0], j[1]), 2, (0, 0, 255), 3)

        plt.subplot(131), plt.imshow(images[i], cmap='gray')
        plt.subplot(132), plt.imshow(blur_images[i], cmap='gray')
        plt.subplot(133), plt.imshow(cimages[i], cmap='gray')
        plt.show()
# for i in range(10):
    #     plt.imshow(images[i], 'gray')
    #     plt.title('gradient')
    # plt.show()


