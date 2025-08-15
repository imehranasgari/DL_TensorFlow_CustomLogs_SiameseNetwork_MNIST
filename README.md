# Handwritten Digit Similarity Detection using a Siamese Network

## Problem Statement and Goal of Project

This project addresses the challenge of "similarity learning" rather than traditional classification. The primary goal is to develop and train a deep learning model that can determine if two images of handwritten digits belong to the same class (i.e., are the same number). This task is fundamental in applications like signature verification, face recognition, and one-shot learning, where the model must generalize to new classes without retraining.

## Solution Approach

To tackle this similarity problem, I implemented a **Siamese Network** using TensorFlow and Keras. This architecture is specifically designed for learning a similarity metric between two inputs.

The core components of the solution are:

  * **Shared-Weight CNN Towers**: The network uses two identical Convolutional Neural Network (CNN) "towers" that share the same weights. Each tower processes one of the input images to generate a low-dimensional feature embedding (a vector representation).
  * **Euclidean Distance Metric**: The similarity between the two input images is calculated by finding the Euclidean distance between their corresponding feature embeddings. A smaller distance implies higher similarity.
  * **Contrastive Loss Function**: The model is trained using a custom **contrastive loss** function. This function is crucial for similarity learning as it:
      * **Minimizes** the distance between embeddings of similar pairs (positive pairs).
      * **Maximizes** the distance between embeddings of dissimilar pairs (negative pairs) up to a defined margin.

This approach effectively teaches the model to create an embedding space where similar items are clustered together and dissimilar ones are pushed apart.

## Technologies & Libraries

  * **TensorFlow & Keras**: For building, training, and evaluating the neural network.
  * **NumPy**: For numerical operations and data manipulation.
  * **Matplotlib**: For visualizing the dataset and training results.
  * **Jupyter Notebook**: For interactive development and experimentation.

## Description about Dataset

The project utilizes the standard **MNIST dataset**, which consists of 60,000 training and 10,000 testing images of handwritten digits (0-9).

For this specific task, the dataset was preprocessed into pairs of images:

  * **Positive Pairs**: Two images of the *same* digit (Label: 1).
  * **Negative Pairs**: Two images of *different* digits (Label: 0).

A custom function was created to generate these pairs, ensuring a balanced set for training the Siamese network on the concept of similarity.

## Installation & Execution Guide

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Siamese_me.ipynb
    ```

## Key Results / Performance

The model was trained for 10 epochs and demonstrated excellent performance in learning the similarity metric.

  * **Test Accuracy**: **98.76%**
  * **Test Loss**: **0.00998**

The training history below shows strong convergence, with validation accuracy consistently improving and loss decreasing, indicating that the model generalized well without significant overfitting.

**Model Accuracy**

**Model Loss**

## Screenshots / Sample Output

Here are some examples of the input data and the model's predictions on the test set.

**Training & Validation Image Pairs**
The model was trained on pairs of images labeled as similar (1) or dissimilar (0).

**Test Set Predictions**
The model correctly predicts whether the digits in a pair are the same (True: 1) or different (True: 0), with the prediction score indicating the calculated distance.

## Additional Learnings / Reflections

This project was a valuable exercise in moving beyond standard classification tasks and implementing a more advanced neural network architecture from scratch.

  * **Custom Loss Functions**: A key takeaway was the hands-on implementation of the contrastive loss function. This provided a deeper understanding of how loss functions can be tailored to guide the learning process for specific, non-standard objectives like metric learning.
  * **Siamese Architecture**: Building the twin-tower architecture with shared weights reinforced my understanding of creating efficient, specialized models in Keras.
  * **Data Preprocessing for Similarity**: The project highlighted the critical importance of thoughtful data preparation. Creating meaningful positive and negative pairs was essential for the model to learn the concept of similarity effectively.

While this implementation was on a simple dataset like MNIST, the principles learned are directly applicable to more complex real-world problems such as biometric verification or content-based image retrieval.

-----

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## 👤 Author

## Mehran Asgari

## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)

## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.