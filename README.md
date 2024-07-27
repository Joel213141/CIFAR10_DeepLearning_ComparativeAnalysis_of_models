# Multi-Model Image Classification Evaluation

## Project Overview
This project evaluates various deep learning models on the CIFAR-10 dataset and a custom dataset of food images. The goal is to benchmark the performance of different architectures and training procedures in classifying images into predefined categories.

## Datasets

### CIFAR-10 Dataset

The CIFAR-10 dataset is a well-known public dataset used in the field of machine learning and computer vision for benchmarking image recognition algorithms. It was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

### Key Features:
- **Dataset Composition**: CIFAR-10 consists of 60,000 32x32 color images in 10 different classes, representing a wide range of real-world objects.
- **Classes**: The dataset is divided into ten classes, with 6,000 images per class. The classes include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- **Training and Testing Split**: Typically, the dataset is divided into 50,000 training images and 10,000 testing images, facilitating the development and evaluation of machine learning models.
- **Applications**: CIFAR-10 is extensively used to train and evaluate machine learning models, especially in tasks requiring the classification of objects into multiple categories. It serves as a benchmark to compare the performance of different algorithms, especially convolutional neural networks.

### Purpose of Use:
The simplicity and manageable size of CIFAR-10 make it ideal for experimenting with new algorithms, neural network architectures, or training techniques before tackling larger and more complex datasets like ImageNet.

For more information and to download the dataset, visit the [CIFAR-10 and CIFAR-100 datasets webpage](https://www.cs.toronto.edu/~kriz/cifar.html).

### Custom Food Image Dataset
The custom food dataset contains images categorized into three classes: Cakes, Pasta, and Pizza. This dataset is used to fine-tune pre-trained models and evaluate their performance on a specialized task outside the CIFAR-10 dataset.

## Models and Training
Multiple neural network architectures were evaluated, including:
- **Standard CNN Models**: Custom models built with convolutional layers, followed by max pooling and fully connected layers.
- **Pre-trained CIFAR-10 Model**: A model initially trained on CIFAR-10, then fine-tuned on the food image dataset.
- **Dropout Variants**: Models incorporating dropout layers to reduce overfitting.

## Training Loss Over Epochs

Below is the plot of the training loss over epochs for the CNN model:

![Training Loss](/results/images/loss.png)

## Evaluation Metrics
The models were evaluated based on their accuracy on the test sets of both datasets. Additionally, confusion matrices were generated to analyze the models' performance in detail, helping identify specific areas of strength and weakness in classification tasks.

### Results Summary

| Model                 | CIFAR-10 Accuracy | Food Dataset Accuracy |
|-----------------------|-------------------|-----------------------|
| Base CNN              | 59%               | 67%                   |
| Pre-trained on CIFAR-10 | 71%               | 73%                   |
| CNN with Dropout      | 52%               | 71%                   |

### Confusion Matrices
Confusion matrices for each model are provided to detail the classification performance across different classes.

## Conclusion
This project highlights the effectiveness of using transfer learning and dropout layers to enhance model performance on both general and specific tasks. The results indicate significant potential for adapting pre-trained models to new domains, which can be especially useful in practical applications where labeled data for specific tasks may be limited.

## Models and Training
- **CNN from Scratch**: Trained a custom CNN model on CIFAR-10 from the ground up.
- **Fine-Tuning**: Adapted a pre-trained CNN model to new classes by modifying the final layer and training on a targeted subset of CIFAR-10.
- **Feature Extraction**: Utilized the CNN as a feature extractor and applied SVM and AdaBoost classifiers on these features.

## Evaluation Metrics
The following tables summarize the performance metrics for the different approaches on the CIFAR-10 test set. Each table provides the model accuracy and a confusion matrix to detail the model's performance across the ten classes in CIFAR-10 (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).

### CNN from Scratch
| Metric         | Value |
| -------------- | ----- |
| Overall Accuracy | 59%   |
| Confusion Matrix | Provided in 'confusion_matrix_cnn.csv' |

### Fine-Tuning CNN
| Metric         | Value |
| -------------- | ----- |
| Overall Accuracy | 71%   |
| Confusion Matrix | Provided in 'confusion_matrix_fine_tune.csv' |

### SVM Classifier on CNN Features
| Metric         | Value |
| -------------- | ----- |
| Overall Accuracy | 57.9% |
| Confusion Matrix | Provided in 'confusion_matrix_svm.csv' |

### AdaBoost Classifier on CNN Features
| Metric         | Value |
| -------------- | ----- |
| Overall Accuracy | 36.8% |
| Confusion Matrix | Provided in 'confusion_matrix_adaboost.csv' |

## Files and Directories
- `src/`: Contains the source code files for the project.
- `data/`: Dataset folder that is used for training and testing the models.
- `models/`: Saved models after training.

## Requirements
To run the code, you will need Python 3.x along with the PyTorch library, torchvision, numpy, sklearn, and matplotlib. You can install all necessary libraries with:
```bash
pip install -r requirements.txt
```

## Usage
To train and evaluate the models, run the scripts located in the src/ directory. Each script corresponds to a different part of the project:

train_cnn.py: For training the CNN from scratch.
fine_tune_cnn.py: For fine-tuning the pre-trained CNN.
feature_classification.py: For running SVM and AdaBoost classifiers on CNN features.

## Results
Results including detailed accuracy reports and visualizations of confusion matrices can be found in the results/ directory. These illustrate the strengths and weaknesses of each model across the different classes of the CIFAR-10 dataset.
