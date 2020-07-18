# Data preparation
## Description...

# Native Python Modules
import os

# Third Party Python Modules
import cv2

# Global Constants
WIDTH = 100
HEIGHT = 100

def conversion(character_name, cascade_file = os.path.join(os.getcwd(), 'data-models', 'lbpcascade_animeface.xml')):
  # 1) Create a cascade file filled with the trained classifier from Nagadomi.
  cascade = cv2.CascadeClassifier(cascade_file)
  raw_images = os.listdir(os.path.join('dataset', 'raw', character_name))

  # 2) Loop through the dataset/raw folder
  for raw_image in raw_images:
    # 3) Resize the image to be 100 x 100.
    # First, read the image, then convert them into greyscale for easier computing.
    image = cv2.imread(os.path.join('dataset', 'raw', raw_image))
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
      cv2.imwrite(os.path.join('dataset', 'train', character_name, filename + '.jpg'), crop_image)

def resize(character_name):
  files = os.listdir(os.path.join('dataset', 'train', character_name))

  for file in files:
    image = cv2.imread(os.path.join('dataset', 'train', character_name, file))
    resized_image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
    filename = os.path.basename(file).split('.')[0]
    cv2.imwrite(os.path.join('dataset', 'train', filename, '.jpg'), resized_image)

def main():
  if len(sys.argv) != 1:
    sys.stderr.write("Usage: python preparation.py <character_name> (character name must be a folder inside the 'raw' and 'train' folder, and must not contain spaces!) \n")
    sys.exit(-1)
  
  conversion(sys.argv[1])
  resize(sys.argv[1])

if __name__ == "__main__":
  main()