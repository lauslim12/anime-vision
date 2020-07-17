# Native Python Modules
import os
import sys
import json
import glob
from zipfile import ZipFile

# Third party Python modules
import cv2
from dotenv import load_dotenv
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Load Environment Variables
env_path = './../.env'
load_dotenv(dotenv_path = env_path)

# Global Constants
HEIGHT = 100
WIDTH = 100

# Conversion using Haar Cascade Algorithm
def conversion(cascade_file="lbpcascade_animeface.xml"):
  # 1) Create a cascade file filled with the trained classifier from Nagadomi.
  cascade = cv2.CascadeClassifier(cascade_file)
  raw_images = os.listdir('./../dataset/raw')

  # 2) Loop through the dataset/raw folder
  for raw_image in raw_images:
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
    resized_image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
    filename = os.path.basename(file).split('.')[0]
    cv2.imwrite(os.path.join('./../dataset/train', filename + ".jpg"), resized_image)

# Testing response from IBM Watson.
def authenticate_watson():
  authenticator = IAMAuthenticator(os.getenv('IBMCLOUD_API_KEY'))
  visual_recognition = VisualRecognitionV3(
    version = '2018-03-19',
    authenticator = authenticator
  )

  visual_recognition.set_service_url(os.getenv('IBMCLOUD_API_URL'))
  return visual_recognition

# Create custom classifier from processed images.
def create_custom_classifier(visual_recognition):
  with open('erza.zip', 'rb') as erza, open('jeanne.zip', 'rb') as jeanne:
    model = visual_recognition.create_classifier('erza_scarlet', 
                                                  positive_examples = {'erza': erza}, 
                                                  negative_examples = jeanne).get_result()
  
  print(json.dumps(model, indent = 2))

def get_classifiers(visual_recognition):
  classifiers = visual_recognition.list_classifiers(verbose=True).get_result()
  print(json.dumps(classifiers, indent=2))
  classifier_id = classifiers['classifiers'][0]['classifier_id']

  return classifier_id

def test_classifier(visual_recognition, classifier_id):
  with open('./../dataset/test/erza-test.jpg', 'rb') as image:
    classes = visual_recognition.classify(
      images_file = image,
      threshold = '0.6',
      classifier_ids = classifier_id).get_result()
  
  print(json.dumps(classes, indent = 2))

def main():
  # conversion()
  # resize()
  visual_recognition = authenticate_watson()
  # create_custom_classifier(visual_recognition)
  classifier_id = get_classifiers(visual_recognition)
  test_classifier(visual_recognition, classifier_id)

if __name__ == '__main__':
  main()
