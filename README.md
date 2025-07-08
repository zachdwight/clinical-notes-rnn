# Simple C++ Recurrent Neural Network (RNN) for Clinical Note Classification

This repository contains a basic, from-scratch implementation of a Recurrent Neural Network (RNN) in C++. Its purpose is to demonstrate the fundamental concepts of RNN architecture, forward propagation, loss calculation, and a simplified backward propagation (training) process.

**Disclaimer:** This is a demo for educational purposes. It uses a **highly simplified RNN architecture and a rudimentary training mechanism**. It is **NOT suitable for real-world medical applications** or any production use. Real-world deep learning for clinical tasks requires robust frameworks (e.g., TensorFlow, PyTorch), advanced architectures (e.g., LSTMs, GRUs, Transformers), and vast, properly curated datasets.  HOWEVER...a little elbow grease with examples like this can yield a better understanding and ownership of ML/AI techniques.

---

## Table of Contents

- [Simple C++ Recurrent Neural Network (RNN) for Clinical Note Classification](#simple-c-recurrent-neural-network-rnn-for-clinical-note-classification)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [How it Works](#how-it-works)
  - [Files in this Repository](#files-in-this-repository)
  - [Prerequisites](#prerequisites)
  - [Building and Running](#building-and-running)
  - [Output Example](#output-example)
  - [Limitations and Future Work](#limitations-and-future-work)
  - [Contributing](#contributing)
  - [License](#license)

---

## Project Overview

This project aims to classify short, simulated clinical notes into one of three categories: "Cold," "Flu," or "Pneumonia." It demonstrates how an RNN can process sequential data (words in a note) to produce a classification output.

## Features

* **From-Scratch Implementation:** No external deep learning libraries are used for the core RNN logic (only standard C++ libraries).
* **Basic RNN Layer:** Implements the fundamental recurrent connections.
* **Tanh Activation:** Used in the hidden layer.
* **Softmax Output:** Produces probability distributions over the diagnosis categories.
* **Mean Squared Error (MSE) Loss:** A simple loss function for demonstration.
* **Rudimentary Gradient Descent:** A simplified backpropagation mechanism for weight updates.
* **Simulated Training Data:** Includes 100 fictional clinical notes for training and testing.

## How it Works

1.  **Vocabulary Mapping:** Words in the clinical notes are mapped to unique numerical IDs.
2.  **One-Hot Encoding:** Each word ID is converted into a one-hot vector, serving as input to the RNN.
3.  **Forward Pass:**
    * For each word in a note, the RNN updates its internal "hidden state."
    * The hidden state captures information from the current word and the previous words in the sequence.
    * After processing all words, the final hidden state is fed into an output layer.
    * The output layer (with Softmax) produces a probability distribution over the possible diagnoses.
4.  **Training (Simplified):**
    * **Loss Calculation:** The Mean Squared Error (MSE) between the predicted probabilities and the true (one-hot encoded) diagnosis is calculated.
    * **Backward Pass (Backpropagation):** The error is propagated backward through the network. This calculates how much each weight and bias contributed to the error.
        * **Note:** The backpropagation implemented here for the recurrent part is highly simplified and does not constitute a full Backpropagation Through Time (BPTT), which is necessary for robust learning in RNNs.
    * **Weight Update:** Weights and biases are adjusted slightly in the direction that reduces the loss, using a small `learning_rate`.
5.  **Iteration:** This forward pass, loss calculation, backward pass, and weight update cycle is repeated for many "epochs" (passes over the entire dataset) to allow the network to "learn" the patterns.

## Files in this Repository

* `simulated_data.h`: Contains the `SIMULATED_CLINICAL_NOTES` array, which is a `std::vector` of `std::pair<std::string, std::string>` representing the note text and its corresponding diagnosis.
* `simple_rnn.cpp`: The main source file containing:
    * Basic matrix and vector operations (addition, multiplication, transpose, etc.).
    * Implementations of `tanh` and `softmax` activation functions and their derivatives.
    * Mean Squared Error loss function and its derivative.
    * The `SimpleRNN` class with `forward`, `backward`, and `train` methods.
    * `main` function to define vocabulary, initialize the RNN, train it, and test its predictions.

## Prerequisites

* A C++17 compatible compiler (e.g., g++, clang++).

## Building and Running

1.  **Save the files:**
    * Save the content of `synthetic_training_data.h` into a file named `synthetic_training_data.h`.
    * Save the content of `clinical_rnn.cpp` into a file named `clinical_rnn.cpp`.

2.  **Compile:** Open your terminal or command prompt, navigate to the directory where you saved the files, and compile using:

    ```bash
    g++ clinical_rnn.cpp -o clinicall_rnn -std=c++17 -O2 -Wall
    ```

3.  **Run:** Execute the compiled program:

    ```bash
    ./clinical_rnn
    ```

## Output Example

During training, you will observe the average loss decreasing and accuracy increasing (though not necessarily linearly or perfectly, especially with this simplified model):

   ```bash
RNN Initialized with:
  Vocab Size: 213
  Hidden Size: 10
  Output Size: 3
  Learning Rate: 0.01
  Epochs: 100

--- Starting Training ---
Epoch 1 completed. Average Loss: 0.220134, Accuracy: 38.00%
Epoch 2 completed. Average Loss: 0.218739, Accuracy: 39.00%
Epoch 3 completed. Average Loss: 0.217235, Accuracy: 39.00%
Epoch 4 completed. Average Loss: 0.215605, Accuracy: 41.00%
Epoch 5 completed. Average Loss: 0.213871, Accuracy: 43.00%
Epoch 6 completed. Average Loss: 0.212054, Accuracy: 47.00%
Epoch 7 completed. Average Loss: 0.210162, Accuracy: 50.00%
Epoch 8 completed. Average Loss: 0.208197, Accuracy: 50.00%
Epoch 9 completed. Average Loss: 0.206154, Accuracy: 54.00%
Epoch 10 completed. Average Loss: 0.204027, Accuracy: 54.00%
Epoch 11 completed. Average Loss: 0.201807, Accuracy: 54.00%
Epoch 12 completed. Average Loss: 0.199483, Accuracy: 56.00%
Epoch 13 completed. Average Loss: 0.197048, Accuracy: 59.00%
Epoch 14 completed. Average Loss: 0.194492, Accuracy: 60.00%
Epoch 15 completed. Average Loss: 0.191807, Accuracy: 58.00%
Epoch 16 completed. Average Loss: 0.188987, Accuracy: 60.00%
Epoch 17 completed. Average Loss: 0.186029, Accuracy: 64.00%
Epoch 18 completed. Average Loss: 0.182932, Accuracy: 64.00%
Epoch 19 completed. Average Loss: 0.179698, Accuracy: 65.00%
Epoch 20 completed. Average Loss: 0.176334, Accuracy: 68.00%
Epoch 21 completed. Average Loss: 0.172852, Accuracy: 68.00%
Epoch 22 completed. Average Loss: 0.169269, Accuracy: 70.00%
Epoch 23 completed. Average Loss: 0.165605, Accuracy: 71.00%
Epoch 24 completed. Average Loss: 0.161886, Accuracy: 71.00%
Epoch 25 completed. Average Loss: 0.158135, Accuracy: 71.00%
Epoch 26 completed. Average Loss: 0.154374, Accuracy: 74.00%
Epoch 27 completed. Average Loss: 0.150614, Accuracy: 75.00%
Epoch 28 completed. Average Loss: 0.146845, Accuracy: 77.00%
Epoch 29 completed. Average Loss: 0.143030, Accuracy: 76.00%
Epoch 30 completed. Average Loss: 0.139096, Accuracy: 76.00%
Epoch 31 completed. Average Loss: 0.134937, Accuracy: 75.00%
Epoch 32 completed. Average Loss: 0.130446, Accuracy: 75.00%
Epoch 33 completed. Average Loss: 0.125589, Accuracy: 79.00%
Epoch 34 completed. Average Loss: 0.120570, Accuracy: 80.00%
Epoch 35 completed. Average Loss: 0.116014, Accuracy: 76.00%
Epoch 36 completed. Average Loss: 0.112696, Accuracy: 83.00%
Epoch 37 completed. Average Loss: 0.110510, Accuracy: 82.00%
Epoch 38 completed. Average Loss: 0.108391, Accuracy: 84.00%
Epoch 39 completed. Average Loss: 0.105777, Accuracy: 83.00%
Epoch 40 completed. Average Loss: 0.102902, Accuracy: 85.00%
Epoch 41 completed. Average Loss: 0.100086, Accuracy: 85.00%
Epoch 42 completed. Average Loss: 0.097468, Accuracy: 85.00%
Epoch 43 completed. Average Loss: 0.095072, Accuracy: 85.00%
Epoch 44 completed. Average Loss: 0.092881, Accuracy: 85.00%
Epoch 45 completed. Average Loss: 0.090871, Accuracy: 86.00%
Epoch 46 completed. Average Loss: 0.089019, Accuracy: 86.00%
Epoch 47 completed. Average Loss: 0.087305, Accuracy: 86.00%
Epoch 48 completed. Average Loss: 0.085714, Accuracy: 86.00%
Epoch 49 completed. Average Loss: 0.084234, Accuracy: 84.00%
Epoch 50 completed. Average Loss: 0.082853, Accuracy: 85.00%
Epoch 51 completed. Average Loss: 0.081564, Accuracy: 85.00%
Epoch 52 completed. Average Loss: 0.080359, Accuracy: 85.00%
Epoch 53 completed. Average Loss: 0.079230, Accuracy: 85.00%
Epoch 54 completed. Average Loss: 0.078173, Accuracy: 85.00%
Epoch 55 completed. Average Loss: 0.077182, Accuracy: 85.00%
Epoch 56 completed. Average Loss: 0.076252, Accuracy: 85.00%
Epoch 57 completed. Average Loss: 0.075379, Accuracy: 85.00%
Epoch 58 completed. Average Loss: 0.074559, Accuracy: 86.00%
Epoch 59 completed. Average Loss: 0.073789, Accuracy: 86.00%
Epoch 60 completed. Average Loss: 0.073067, Accuracy: 86.00%
Epoch 61 completed. Average Loss: 0.072388, Accuracy: 87.00%
Epoch 62 completed. Average Loss: 0.071752, Accuracy: 87.00%
Epoch 63 completed. Average Loss: 0.071155, Accuracy: 87.00%
Epoch 64 completed. Average Loss: 0.070597, Accuracy: 87.00%
Epoch 65 completed. Average Loss: 0.070074, Accuracy: 87.00%
Epoch 66 completed. Average Loss: 0.069585, Accuracy: 87.00%
Epoch 67 completed. Average Loss: 0.069129, Accuracy: 87.00%
Epoch 68 completed. Average Loss: 0.068704, Accuracy: 87.00%
Epoch 69 completed. Average Loss: 0.068308, Accuracy: 87.00%
Epoch 70 completed. Average Loss: 0.067941, Accuracy: 87.00%
Epoch 71 completed. Average Loss: 0.067602, Accuracy: 87.00%
Epoch 72 completed. Average Loss: 0.067287, Accuracy: 87.00%
Epoch 73 completed. Average Loss: 0.066998, Accuracy: 87.00%
Epoch 74 completed. Average Loss: 0.066731, Accuracy: 87.00%
Epoch 75 completed. Average Loss: 0.066487, Accuracy: 87.00%
Epoch 76 completed. Average Loss: 0.066263, Accuracy: 87.00%
Epoch 77 completed. Average Loss: 0.066059, Accuracy: 87.00%
Epoch 78 completed. Average Loss: 0.065873, Accuracy: 86.00%
Epoch 79 completed. Average Loss: 0.065705, Accuracy: 86.00%
Epoch 80 completed. Average Loss: 0.065553, Accuracy: 85.00%
Epoch 81 completed. Average Loss: 0.065415, Accuracy: 85.00%
Epoch 82 completed. Average Loss: 0.065292, Accuracy: 84.00%
Epoch 83 completed. Average Loss: 0.065182, Accuracy: 84.00%
Epoch 84 completed. Average Loss: 0.065083, Accuracy: 84.00%
Epoch 85 completed. Average Loss: 0.064996, Accuracy: 84.00%
Epoch 86 completed. Average Loss: 0.064918, Accuracy: 84.00%
Epoch 87 completed. Average Loss: 0.064850, Accuracy: 84.00%
Epoch 88 completed. Average Loss: 0.064790, Accuracy: 84.00%
Epoch 89 completed. Average Loss: 0.064737, Accuracy: 84.00%
Epoch 90 completed. Average Loss: 0.064690, Accuracy: 84.00%
Epoch 91 completed. Average Loss: 0.064650, Accuracy: 86.00%
Epoch 92 completed. Average Loss: 0.064614, Accuracy: 86.00%
Epoch 93 completed. Average Loss: 0.064583, Accuracy: 86.00%
Epoch 94 completed. Average Loss: 0.064557, Accuracy: 86.00%
Epoch 95 completed. Average Loss: 0.064533, Accuracy: 85.00%
Epoch 96 completed. Average Loss: 0.064512, Accuracy: 85.00%
Epoch 97 completed. Average Loss: 0.064494, Accuracy: 85.00%
Epoch 98 completed. Average Loss: 0.064478, Accuracy: 85.00%
Epoch 99 completed. Average Loss: 0.064463, Accuracy: 84.00%
Epoch 100 completed. Average Loss: 0.064450, Accuracy: 84.00%
--- Training Finished ---

--- Final Predictions After Training ---

Testing Note 1 (True: Cold): "Patient presents with mild cough, runny nose, and sore throat."
  Predicted probabilities:
    Cold: 0.93%
    Flu: 96.44%
    Pneumonia: 2.63%
  Most probable diagnosis: Flu (True: Cold)

Testing Note 2 (True: Flu): "Complains of body aches, chills, and a fever of 102F. Feels exhausted."
  Predicted probabilities:
    Cold: 10.43%
    Flu: 1.39%
    Pneumonia: 88.18%
  Most probable diagnosis: Pneumonia (True: Flu)

Testing Note 3 (True: Pneumonia): "Patient presents with productive cough (green sputum), shortness of breath, and fever."
  Predicted probabilities:
    Cold: 16.40%
    Flu: 0.16%
    Pneumonia: 83.44%
  Most probable diagnosis: Pneumonia (True: Pneumonia)

Testing Note 4 (True: Cold): "No fever, but has a slight sore throat and runny nose."
  Predicted probabilities:
    Cold: 8.69%
    Flu: 61.17%
    Pneumonia: 30.14%
  Most probable diagnosis: Flu (True: Cold)

Testing Note 5 (True: Flu): "Severe fatigue, headache, and generalized muscle pain. Fever."
  Predicted probabilities:
    Cold: 18.32%
    Flu: 3.26%
    Pneumonia: 78.42%
  Most probable diagnosis: Pneumonia (True: Flu)

Testing Note 6 (True: Pneumonia): "Fever 102F, severe cough producing thick phlegm, and shortness of breath."
  Predicted probabilities:
    Cold: 33.68%
    Flu: 1.68%
    Pneumonia: 64.64%
  Most probable diagnosis: Pneumonia (True: Pneumonia)
  ```

*(Note: Actual accuracy and loss values will vary slightly due to random initialization.)*

## Limitations and Future Work

* **Simplified Backpropagation:** The current `backward` implementation for the recurrent part is a significant simplification. A full Backpropagation Through Time (BPTT) implementation is much more complex, requiring accumulation of gradients across all time steps and careful handling of the hidden state dependencies.
* **Vanishing/Exploding Gradients:** Simple RNNs struggle with learning long-term dependencies due to these issues. This implementation does not address them.
* **No LSTMs/GRUs:** For more robust sequence processing, advanced architectures like Long Short-Term Memory (LSTM) networks or Gated Recurrent Units (GRUs) are necessary. Implementing these from scratch is a substantial undertaking.
* **Limited Vocabulary and Data:** The vocabulary is hardcoded and limited. The training data is very small and synthetic. Real-world applications require extensive text preprocessing (tokenization, stemming/lemmatization, handling OOV words), larger vocabularies, and much more data.
* **Basic Optimizer:** Uses simple Gradient Descent. More advanced optimizers (Adam, RMSprop) are standard in modern deep learning.
* **No Regularization:** Techniques like dropout or L2 regularization are typically used to prevent overfitting.
* **No Mini-Batching:** Training is done one sample at a time (Stochastic Gradient Descent). Mini-batching is standard for efficiency and stability.
* **Matrix Library:** The matrix operations are basic. For production code, using a highly optimized linear algebra library (e.g., Eigen) is essential.

## Contributing

Feel free to fork this repository, experiment with the code, and suggest improvements. Please remember its purely educational nature.  Thanks to Google Gemini for helping write some notes and clean up my code/comments.

## License

This project is open-source and available under the [MIT License](LICENSE).
