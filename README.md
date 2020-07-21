# Anime Vision

Anime-Vision is a repository for anime face identification with Deep Learning.

## TODO

- Improve accuracy of the model with TensorFlow's Model Maker
- Improve accuracy of the model with TensorFlow's Make Image Classifier
- Allow creating classifiers to iterate through all the zip files, not manually like in line 31. (IBM Cloud)
- Create scraper to fetch raw datasets

## Introduction

Under development.

## Requirements

- Anaconda
- Google Colaboratory
- IBM Cloud
- Keras
- OpenCV
- TensorFlow Lite
- TensorFlow 1.15
- TensorFlow 2.10

## Project Structure

Under development.

## Ways to Classify

I'm going to use several ways to classify anime faces. The ways are as follows:

- [x] Using TensorFlow 1.15's Transfer Learning Retrain Script
- [ ] Using IBM Cloud Visual Recognition to recognize faces
- [x] Using TensorFlow Lite's Model Maker
- [x] Using TensorFlow 2's Make Image Classifier
- [ ] Using TensorFlow 2's Image Classifier manually with Keras

## Accuracy

For now, I believe the one with the highest accuracy is TensorFlow 1.15's Transfer Learning. It allows accuracy for my model up to **94%!**

## How to Use

This is still a quick prototype. Will edit this section later. Ensure that you have Anaconda installed! To use this program, don't just do `git clone` and then starting it blindly. It wouldn't work. Ways to set up this program are as follows:

### Data Gathering

- First off, as I'm running this in a Windows environment, simply copy my environment and use it.

```
  $ conda --version
  $ conda env create -f anime-vision.yml
  $ conda activate anime-vision
```

- If you simply want to test the predictions, skip this step. If you want to test this program with your own anime characters, prepare your dataset to a folder in the `dataset/raw/<your_character_name>` and `dataset/train/<your_character_name>`. Make sure that the name is in lowercase and replace any spaces with underscores!

- Then, run `preparation.py <character_name>`. The accuracy for this is **only 83%**, so not all faces could get detected! I have personally seen and edited the dataset so all of the faces could be detected.

- Alternatively, you could use [qhgz2013's Anime Face Detector](https://github.com/qhgz2013/anime-face-detector). It allows up to 90% of accuracy when detecting anime faces. You might have to edit the source code so it crops the detected faces, though. It is trained with Faster-RCNN.

- Another alternative is using [Nagadomi's Anime Face 2009](https://github.com/nagadomi/animeface-2009). However, installation of that is a bit complicated, even the author said so himself.

- Either way, you could use anything that is comfortable with you.

- You are done with getting the main dataset for training. Here's where the fun part starts.

### Data Analysis

For data analysis itself, you could use five ways to classify (see 'Ways to Classify'). I'm going to start from the first way.

#### Using TensorFlow's 1.15 Transfer Learning Script

- First off, switch to the `tensorflow-1.15` folder. Then, run the `train.sh` script. This is to train the data! You could also train them manually (not using shell script) by using:

```
  $ python retrain.py \
    --output_graph ./retrained_graph.pb \
    --output_labels ./retrained_labels.txt \
    --image_dir ./../dataset/train/
```

- As a note, I intentionally left the step size as 4000 (default).
- Second, test the model by using the following command:

```
  $ python retrain-test.py <path_to_image>
```

- Third, you could see the results of the trained model!

#### Using TensorFlow Lite's Model Maker

- This script is a bit different, as it uses Jupyter Notebook to run. You can open the script in Google Colaboratory, and simply run it there!

#### Using TensorFlow 2's Make Image Classifier

- This script is the same as above.

#### Note

- When using Google Colaboratory, ensure that you have uploaded the image that you want to train in a zipped file named `dataset`. Or, you could edit it yourself straight from the Google Colaboratory.

## Disclaimer

I do NOT own any of the pictures that might be inside the `dataset` folder. All Rights Reserved to their original owners / artists / creators.

## Credits

- [FreedomOfKeima](https://github.com/freedomofkeima) for the dataset and inspiration.
- [Nagadomi](https://github.com/nagadomi) for the anime face model.
