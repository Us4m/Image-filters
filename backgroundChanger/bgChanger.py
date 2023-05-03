import cv2
from rembg import remove
import requests
from PIL import Image 
from io import BytesIO
import os 

os.makedirs('original', exist_ok=True)
os.makedirs('masked', exist_ok=True)

# now here we remove bg
img_address= 'D:\django task\i\original/person.jpg'
# img_name = 'person.jpg'
img_name = img_address.split('/')[-1]
output_path = 'masked/'+img_name 
output_path

with open(output_path, 'wb') as f:
  input = open('original/'+img_name, 'rb').read()
  subject = remove(input, alpha_matting=True, alpha_matting_foreground_threshold=70)
  f.write(subject)

background_img = Image.open('D:\django task\i\original/background.jpg')
realimage = Image.open('D:\django task\i\original/person.jpg')

background_img = background_img.resize((realimage.width, realimage.height)) 

foreground_img = Image.open('D:\django task\i\masked/person.jpg')
background_img.paste(foreground_img, (0,0), foreground_img)
background_img.save('D:\django task\i\masked/result.png', format='png')