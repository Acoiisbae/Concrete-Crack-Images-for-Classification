# Concrete Crack Images for Classification

# Project Description

This project aims to develop a deep learning model for image classification using the TensorFlow framework. The dataset used in this project is sourced from (https://data.mendeley.com/datasets/5y9wdsg2zt/2).

# Project Details
First, the data is loaded by defining the path to the dataset using os.path.join. Then, the batch size and image size are specified, and the data is loaded as a TensorFlow dataset. The data is split into train and validation datasets, and some examples of the images are displayed to ensure that the dataset is loaded correctly.

Next, the dataset is pre-processed by performing a validation-test split, and the train, validation, and test datasets are converted into a prefetch dataset.

A model is developed for image augmentation using Sequential, and Transfer Learning is applied to improve the model accuracy, using MobileNetV2. A layer is created for data normalization, and the pretrained model is instantiated and set as non-trainable (frozen).

![model](https://user-images.githubusercontent.com/125865422/220685558-5f699d47-c6bc-474e-adde-d04a68eaf12f.png)

The classifier is created, and the layers are linked together to form a pipeline. The full model pipeline is then instantiated, and the model is compiled using the Adam optimizer, categorical cross-entropy loss, and accuracy metrics.

Callbacks are implemented for Early Stopping and TensorBoard, with logs generated after every batch of training to monitor metrics, and the model is saved to disk.

The model is then trained for 10 epochs, and the model accuracy is evaluated. The model confusion matrix, classification report, and accuracy score are also evaluated.

The model is then deployed to perform prediction, and the model is saved.

Finally, the TensorBoard Loss Graph is analyzed, and solutions to model overfitting issues are discussed, including adding more dropout layers, regularization and ensembling, and saving the best model using model_checkpoint.

Overall, this project provides a comprehensive overview of developing a deep learning model for image classification using TensorFlow, with detailed explanations of each step in the process.




