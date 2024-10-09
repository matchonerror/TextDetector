import easyocr
import cv2
import os

#Read the image
img_path = os.path.join('/Users/thanhphu/PycharmProjects/TextDetector/data/ad_panel.png')
img = cv2.imread(img_path)
# preprocessing
# convert image to grayscale
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# create threshold for image
# passing a gaussian blur for sharping the text
blurred=cv2.GaussianBlur(gray_img,(3,3),0,sigmaY=0)
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# #instance text detector
# reader = easyocr.Reader(['en'], gpu=False)
#
# # read the text
# text = reader.readtext(img)
# threshold = 0.5
# # drawing the rectangles for the texts
# for t in text:
#     print(t)
#
#     bbox,text,score = t
#     if score > threshold:
#         cv2.rectangle(img,bbox[0],bbox[2],(0,255,0),2)
#         cv2.putText(img,text,bbox[3],cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,0),2)

# cv2.imwrite('text_on_img_2.png', img)
cv2.imshow('img', binary)
cv2.waitKey(0)