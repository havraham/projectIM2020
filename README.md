# ProjectIM2020 - final project image processing class

final project in image processing course


# Q1 - finding chessboard and take only the relevant area

This parts contians 2 different mission
1. Find the squares cornes
2. Find the ruler and take only the relevant fragment

For finding the squares i used opencv findContours function
and then i draw the corners.

For finding the ruler i used HoughLine for finding the rulers lines.
I found the relevent edges and cut the image according them.


# Q2 - Finding circles template in footprints

For finding the circles i used HoughCircle function.
I used GaussianBlur and AdaptiveThreshold for
making the circles detection more generic.

# Q3 - Finding the original photo from DB

I used SIFT for find the images keypoints.
and bruteforce Matcher and KNN for features matching.

I had to clean the noise from the inputs with Gaussian blur.
and i had to remove part of the frame because it confused the algorithm


# Q4 - Finding Circles

Same like q2 but more agressive Gaussian Blur and different params


