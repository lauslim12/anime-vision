# Native Python Modules
import os
import json
from zipfile import ZipFile

# Third party Python modules
from dotenv import load_dotenv
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

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
# Work in progress!
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
  # Change directory back to the root folder. Important for easier navigation!
  os.chdir('../')

  # Load environment variables.
  load_dotenv()

  visual_recognition = authenticate_watson()
  create_custom_classifier(visual_recognition)
  classifier_id = get_classifiers(visual_recognition)
  test_classifier(visual_recognition, classifier_id)

if __name__ == '__main__':
  main()
