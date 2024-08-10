# Landmark Classification and Tagging for Social Media

## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify landmarks in images, which can be used to infer the location of an image when no metadata is available. This is particularly useful for photo sharing and storage services that rely on location data to provide advanced features like automatic tagging and photo organization.

## Project Steps

The project was divided into three main phases:

1. **Create a CNN to Classify Landmarks (from Scratch):**
   - Visualized and preprocessed the dataset for training.
   - Designed and trained a CNN from scratch to classify landmarks.
   - Exported the best model using Torch Script.

2. **Create a CNN to Classify Landmarks (using Transfer Learning):**
   - Investigated various pre-trained models and selected the most suitable one for the task.
   - Fine-tuned the pre-trained network to improve classification accuracy.
   - Exported the best transfer learning model using Torch Script.

3. **Deploy the Algorithm in an App:**
   - Developed a simple app using the best-performing model.
   - Tested the app to ensure it accurately identifies landmarks and reflects on its strengths and weaknesses.

## Dataset

The dataset used in this project consists of images depicting various landmarks from around the world. The images were labeled according to the landmark they represent, which served as the basis for training the CNN models.


## Results

The models achieved a test accuracy of  57% (CNN from scratch) and 71.6%(transfer learning) in classifying landmarks, demonstrating the effectiveness of both the custom CNN and the transfer learning approach. The app successfully identified landmarks from various test images, though there were some challenges with less common landmarks.

## Future Work

- Improve model accuracy with more diverse data.
- Extend the app's functionality to handle multiple landmarks in one image.
- Explore additional pre-trained models for better transfer learning performance.
