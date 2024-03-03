# Image Classifier Application

This project was developed for Udacity AI Programming with Python Nanodegree. The project builds an image classifier application using transfer learning to classify different flower species.

The "Image_Classifier_Project.ipynb" file can be run easily in Colab. The dataset is loaded within the notebook. Additionally, the "cat_to_name.json" file is needed for label mapping. 

**The application can also be run via the command line as follows:**

 * The 'train.py' file can be used to train a model. The following can be specified via command line arguments:
   * Data directory: 'path to where the data is stored'
   * Model architecture (either vgg16 or vgg13): --arch vgg16
   * Checkpoint save directory: --save_dir 'path to where checkpoint should be saved'
   * Learning rate: --learning_rate 0.003
   * Hidden units: --hidden_units 200
   * Epochs: --epochs 5
   * Device (GPU and the default is CPU if not specified): --gpu
      
 * The 'predict.py' file can be used to make predictions using the trained model. The following can be specified via command line arguments:
   * Image path: 'path to where the image is stored'
   * Checkpoint file: 'path to where the checkpoint is stored'
   * Predicting the N most likely classes: --top_k 5
   * Label file: --category_names 'path to the file that has the labels'
   * Device (GPU and the default is CPU if not specified): --gpu
