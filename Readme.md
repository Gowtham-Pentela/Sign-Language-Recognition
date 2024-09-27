# Sign Language Recognition using CNN

This project aims to develop a Convolutional Neural Network (CNN) model for recognizing American Sign Language (ASL) letters (A-Z). The dataset used contains grayscale images of hand signs representing different letters. The model has achieved **99.023% accuracy** on the test dataset.

## Dataset
- **Train Dataset**: `sign_mnist_train.csv`
- **Test Dataset**: `sign_mnist_test.csv`
- The images are 28x28 pixels grayscale with corresponding labels representing different sign language letters.

## Libraries Used
- **Pandas**: For data manipulation.
- **Numpy**: For numerical computations.
- **Seaborn & Matplotlib**: For visualizations.
- **Scikit-learn**: For data preprocessing and train-test splitting.
- **TensorFlow/Keras**: For building and training the neural network.
- **Visualkeras**: For visualizing the model layers.
- **ImageDataGenerator**: For real-time data augmentation during training.

## Model Architecture
- **Input Shape**: 28x28x1 (grayscale images)
- **Convolution Layers**: 2 Conv2D layers with ReLU activation and batch normalization.
- **Pooling Layers**: MaxPooling layers to reduce the spatial size.
- **Dropout**: A dropout layer to reduce overfitting.
- **Fully Connected Layer**: A Dense layer with 24 output units (since the dataset contains 24 classes excluding J and Z) and softmax activation for classification.

## Preprocessing
- **Normalization**: The pixel values are scaled to the range [0, 1] by dividing by 255.
- **Label Binarization**: The labels are one-hot encoded.
- **Data Augmentation**: Random rotations, zooms, width and height shifts are applied during training using `ImageDataGenerator`.

## Model Performance
- **Accuracy**: 99.023%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Learning Rate Schedule**: ReduceLROnPlateau callback is used to reduce the learning rate when the validation accuracy plateaus.

## Training and Validation
The model was trained for 10 epochs with a batch size of 128. The training and validation accuracy and loss curves are visualized to assess the performance.

## Visualization
- **Model Architecture**: Visualized using `visualkeras.layered_view`.
- **Data Visualizations**: Heatmaps of correlations and null values, along with a countplot of the labels.

## Installation
To run this project, install the following dependencies:
```bash
pip install numpy pandas seaborn matplotlib tensorflow visualkeras scikit-learn
```

## How to Run
1. Place the `sign_mnist_train.csv` and `sign_mnist_test.csv` in the dataset directory.
2. Run the Python script to preprocess data, train the model, and visualize results.
3. The final accuracy is printed after training along with training/validation plots.

## Future Improvements
- Implement further hyperparameter tuning to enhance the model's performance.
- Explore deeper architectures or transfer learning techniques for even higher accuracy.
  
--- 
