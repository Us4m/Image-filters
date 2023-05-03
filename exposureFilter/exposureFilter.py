import cv2

# Load image file  
img = cv2.imread( 'original/person.jpg', cv2.IMREAD_COLOR)

# Get user input for exposure level
exposure_level = int(input("Enter exposure level (in positive/negative integers): "))

# Apply exposure on image
exposure_img = cv2.addWeighted(img, 1, img, 0, exposure_level)

# Save modified image
cv2.imwrite('exposure_image.jpg', exposure_img)
