import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__== "__main__":
    img = cv2.imread('00008.jpg' , 0)
    blur = cv2.GaussianBlur(img, (15,15),0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Elliptical Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blur = cv2.morphologyEx(blur,cv2.MORPH_CLOSE,kernel)
    blur = cv2.morphologyEx(blur,cv2.MORPH_OPEN,kernel)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 3,
                                   param1=85, param2=55, minRadius=0, maxRadius=0)

    print(circles)
    #  ------------------  Best Parameters so far ---------------
    # circles = cv2.HoughCircles(blur_images[i], cv2.HOUGH_GRADIENT, 1, 75,
    #                            param1=225, param2=100, minRadius=0, maxRadius=0)
    #  ------------------  Best Parameters so far ---------------

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for j in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (j[0], j[1]), j[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (j[0], j[1]), 2, (0, 0, 255), 3)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.subplot(132), plt.imshow(blur, cmap='gray')
    plt.subplot(133), plt.imshow(cimg, cmap='gray')
    plt.show()



