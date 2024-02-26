# Image Classifier Application

This project was developed for Udacity AI Programming with Python Nanodegree. The project builds an image classifier application using transfer learning to classify different flower species.

The "Image_Classifier_Project.ipynb" file can be run easily in Colab. The dataset is loaded within the notebook. Additionally, the "cat_to_name.json" file is needed for label mapping. 

The application can also be run via the command line as follows:

    * The 'train.py' file can be used to train a model. Data directory, model architecture (either vgg16 or vgg13), checkpoint save directory, learning rate, hidden units, epochs, and GPU can be specified via command line arguments.
    * The 'predict.py' file can be used to make predictions using the trained model. Image path, checkpoint file, predicting the N most likely classes, label file, and GPU can be specified via command line arguments.    




