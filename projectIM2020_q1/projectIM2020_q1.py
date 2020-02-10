import cv2
import numpy as np
from matplotlib import pyplot as plt


def sharpen_img(img, ksize=(15,15), sigmaX=0, alpha=1):
    print("sharpen",img,ksize,sigmaX,alpha)
    filter_blured = cv2.GaussianBlur(img,ksize,sigmaX)
    # filter_blured = cv2.medianBlur(A,9)

    print("blured",filter_blured)
    sharpened = img + alpha * (img - filter_blured)
    print("Sharpend",sharpened)

    images = [img, filter_blured, sharpened ]
    titles = ["original", "gaussian blur", "shaprened"]
    # custom_plot(images,titles,3,1)

    return sharpened

def custom_plot(images,titles,rows=2,cols=2):
    for i in range(len(images)):
        print(images[i].shape)
        plt.subplot(cols, rows, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i]), plt.xticks([]), plt.yticks([])

    plt.show()

def corner_harris():
    filename = '3.JPG'

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray,21)
    # kernel = np.ones((9, 9), np.uint8)
    # gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    # gray = 255 - gradient
    plt.imshow(gray,'gray')
    plt.title('gradient')
    plt.show()

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 8, 1, 0.04)

    plt.imshow(dst,'gray')
    plt.title('dst')
    plt.show()

    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)
    plt.imshow(dst,'gray')
    plt.title('dst')
    plt.show()
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [255 ,0, 0]


    plt.imshow(img)
    plt.show()


if __name__== "__main__":
    A = cv2.imread('3.JPG',0)
    B = cv2.imread('12.JPG',0)
    blurA = cv2.medianBlur(A,15)
    blurB = cv2.medianBlur(B,15)
    # plt.imshow(A,'gray')
    # plt.title('A')
    # plt.show()

    corner_harris()

    # sharpened = sharpen_img(blurA,(15,15),3,1)
    # plt.imshow(sharpened, 'gray')
    # plt.title('sharpened')
    # plt.show()
    #
    ret, th1 = cv2.threshold(blurA, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)
    gradient = cv2.morphologyEx(blurA, cv2.MORPH_GRADIENT, kernel)
    # gradient = 255 - gradient


    # plt.imshow(gradient, 'gray')
    # plt.title('gradient')
    # plt.show()
    # th2 =  cv2.adaptiveThreshold(blurA, 127, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # th3 =  cv2.adaptiveThreshold(sharpened, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #
    # plt.imshow(th1,'gray')
    # plt.title('th1')
    # plt.show()
    #
    # plt.imshow(th2,'gray')
    # plt.title('th2')
    # plt.show()
    #
    # plt.imshow(th3,'gray')
    # plt.title('th3')
    # plt.show()


