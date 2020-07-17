import cv2
import sys
import os

# Constant
HEIGHT = 100
WIDTH = 100


# Conversion using Haar Cascade Algorithm
def conversion(cascade_file="lbpcascade_animeface.xml"):
  # 1) Create a cascade file filled with the trained classifier from Nagadomi.
  cascade = cv2.CascadeClassifier(cascade_file)
  raw_images = os.listdir('./../dataset/raw')

  # 2) Loop through the dataset/raw folder
  for raw_image in raw_images:
    print(raw_image)
    # 3) Resize the image to be 100 x 100.
    # First, read the image, then convert them into greyscale for easier computing.
    image = cv2.imread(os.path.join('./../dataset/raw', raw_image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Recognize the face in the image.
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(25, 25))

    # Iterate through the face to crop.
    for (x, y, w, h) in faces:
      # crop_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
      crop_image = image[y:y + h, x:x + w]
      filename = os.path.basename(raw_image).split('.')[0]
      cv2.imwrite(os.path.join('./../dataset/train', filename + ".jpg"),
                  crop_image)


# Resize all of the images to be 100 x 100.
def resize():
  files = os.listdir('./../dataset/train')
  for file in files:
    image = cv2.imread(os.path.join('./../dataset/train', file))
    resized_image = cv2.resize(image, (WIDTH, HEIGHT),
                               interpolation=cv2.INTER_AREA)
    filename = os.path.basename(file).split('.')[0]
    cv2.imwrite(os.path.join('./../dataset/train', filename + ".jpg"),
                resized_image)


def main():
  conversion()
  resize()


if __name__ == '__main__':
  main()
