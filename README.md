# Machine Learning Neural Networks Project

This project implements foundational and advanced machine learning models, including Naïve Bayes, Perceptron, regression neural networks, and Recurrent Neural Networks (RNNs) for language identification. The assignment was structured to build up from the basics of classification to deeper neural architectures, using a custom neural network mini-library.

## Table of Contents
- [Project Structure](#project-structure)
- [Implemented Models](#implemented-models)
- [Installation](#installation)
- [Usage](#usage)
- [Details of Each Part](#details-of-each-part)
- [References & Attribution](#references--attribution)

---

## Project Structure

```
.
├── autograder.py # Custom autograder for automatic testing/grading
├── backend.py # Dataset loading & backend utilities
├── models.py # Main implementations for all models (edit this file)
├── nn.py # Neural network mini-library (do not edit)
├── util.py # Utilities and Counter class (do not edit)
├── data/ # Data folder containing MNIST and other datasets
├── ML_Project_nn.pdf # Assignment PDF with full project instructions
```
- **Files you should edit:**  
  `models.py` (main logic for Naïve Bayes, Perceptron, Regression, Neural Network models)
- **Files you should NOT edit:**  
  `nn.py`, `autograder.py`, `backend.py`, `util.py`, and files in `data/`.

---

## Implemented Models

### 1. Naïve Bayes Digit Classifier
- Implements a Naïve Bayes classifier for handwritten digit recognition using the MNIST dataset.
- Laplace smoothing parameter is tuned using validation accuracy.

### 2. Perceptron Classifier
- Binary perceptron for classifying 2D points with a linear decision boundary.
- Full online training loop with convergence criterion.

### 3. Regression Neural Network
- Neural network regressor for approximating trigonometric functions (e.g., `y = sin(x^2)`).
- Customizable depth, batch size, and learning rate.

### 4. Neural Network Digit Classifier
- Fully connected deep neural network to classify handwritten digits (larger MNIST dataset, higher accuracy).
- Batch training with ReLU non-linearity.

### 5. RNN for Language Identification *(if included in your assignment)*
- Recurrent Neural Network to classify the language of given word sequences.

---

## Installation

**Dependencies:**
- Python 3.12+ (3.9+ should work)
- `numpy`
- `matplotlib`

Install dependencies:
```bash
python autograder.py --check-dependencies
```
Test dependencies installation:
```python autograder.py --check-dependencies```
A window should appear with an animated line segment (for matplotlib test).
## Usage

You can run and test each part using the ```autograder.py``` script. For example, to test Question 1 (Naïve Bayes):
```bash
python autograder.py -q q1
python autograder.py -q q2
python autograder.py -q q3
# etc.
```
Or run all tests:
```bash
python autograder.py
```
# Running Models (Inside Python)

Example of how to iterate over a dataset:
```python
for x, y in dataset.iterate_once(batch_size):
    # x and y are nn.Constant nodes, shapes: [batch_size x input_dim], [batch_size x output_dim]
    ...
```
# Details of Each Part
Q1: Naïve Bayes (models.NaiveBayesDigitClassificationModel)

   - Trains on a subset of MNIST (5000 training, 1000 validation, 1000 test).

   - Implements feature counting and Laplace smoothing, with hyperparameter tuning.

   - Achieves full credit if validation accuracy ≥ 79%.

Q2: Perceptron (models.PerceptronModel)

   - Binary classifier with online weight updates.

   - Trains until 100% accuracy on training data (convergence).

Q3: Regression NN (models.RegressionModel)

  -  Batched fully-connected feedforward network (at least one hidden layer).

   - Uses ReLU activations and MSE loss.

Q4: Digit Classification NN

  -  Deep neural network for MNIST digit recognition (larger dataset).

   - Structure: Input → Dense → ReLU → Dense ... → Output

Q5: RNN Language Identification

   - (If assigned) RNN to classify the language of word sequences.


# References & Attribution

  -  Assignment design, utility code, and base frameworks adapted from UC Berkeley's CS188 and related educational materials.

   - Numpy Documentation

   - Matplotlib Documentation

   - NN mini-library and datasets: See ```nn.py```, ```backend.py```, and ```data/```.
