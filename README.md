# car-logo-recognition

a.ipynb -->This file is for training the model, with CNNs.
It loads car logo images, applies data transformations (resize, flip, color adjustments), 
and trains the model to classify images into car brand categories (e.g., Hyundai, Lexus) - with computations of the loss also.
After training the model is saved as a file. And the accuracy is computed.

app.py-->In this Falsk application I provide and web app for users to upload the images. 
So,I just load the trained model and executed in evaluation mode to ensure reliable predictions. 
The uploaded image is preprocessed and fed to the trained model, which predicts the car brand. 

I have a homepage for image upload (upload.html) and an results page (results.html with POST /:) which receives the uploaded image, processes it, and 
displays the classification result.


## Requirements
Python Packages: torch, torchvision, flask, Pillow
Environment: CUDA (optional, for GPU support)

## Running the App
To run locally:

Start the jupiter server for the training of the model and run each section.
After obtaining the model you can start the Flask app.

Start the Flask app: run app.py - for the classifications
Access it via http://127.0.0.1:5000/ in your browser.
