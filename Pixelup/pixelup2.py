import tensorflow as tf
import cv2

# Load the low-resolution image
img = cv2.imread('badpixel.jpg')

# Load the ESRGAN model
model = tf.keras.models.load_model('esrgan_latest_g.h5')

# Preprocess the image
img = img.astype('float32') / 255.0
img = tf.expand_dims(img, axis=0)

# Use the ESRGAN model to generate a high-resolution image
output = model.predict(img)

# Post-process the output image
output = tf.squeeze(output, axis=0)
output = tf.clip_by_value(output, 0, 1)
output = output.numpy() * 255.0
output = output.astype('uint8')

# Save the output image
cv2.imwrite('output_image.jpg', output)
