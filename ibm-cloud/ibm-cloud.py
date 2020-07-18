# Native Python Modules
import json
import os
import sys

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
  
  # Error handling to prevent using an empty list.
  if not classifiers['classifiers']:
    sys.exit('Your IBM Cloud Account has no classifiers! Please ensure that you have one!')
  else:
    classifier_id = classifiers['classifiers'][0]['classifier_id']

  return classifier_id

def delete_classifier(visual_recognition, classifier_id):
  classifiers = visual_recognition.delete_classifier(classifier_id).get_result()
  print(json.dumps(classifiers, indent = 2))

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

  # Load environment variables and visual recognition.
  load_dotenv()
  visual_recognition = authenticate_watson()

  # Print out error message if the user does not use arguments.
  if len(sys.argv) != 2:
    sys.stderr.write("Usage: python ibm-cloud.py <action> (check, delete, create, test) \n")
    sys.exit(-1)

  # Do process.
  if sys.argv[1] == 'delete':
    classifer_id = get_classifiers(visual_recognition)
    delete_classifier(visual_recognition, classifer_id)
  elif sys.argv[1] == 'create':
    create_custom_classifier(visual_recognition)
  elif sys.argv[1] == 'test':
    classifier_id = get_classifiers(visual_recognition)
    test_classifier(visual_recognition, classifier_id)
  elif sys.argv[1] == 'check':
    classifer_id = get_classifiers(visual_recognition)

if __name__ == '__main__':
  main()
