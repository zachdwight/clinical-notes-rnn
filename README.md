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
  - [Quick Start](#quick-start)
  - [CLI Usage](#cli-usage)
  - [Data Formats](#data-formats)
  - [Example Workflows](#example-workflows)
  - [Files in this Repository](#files-in-this-repository)
  - [Prerequisites](#prerequisites)
  - [Building and Running](#building-and-running)
  - [Output Example](#output-example)
  - [Limitations and Future Work](#limitations-and-future-work)
  - [Recent Improvements](#recent-improvements)
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
* **Cross-Entropy Loss:** Proper loss function for multi-class classification with softmax.
* **Full Backpropagation Through Time (BPTT):** Accumulates gradients across all time steps for better learning.
* **Gradient Clipping:** Prevents exploding gradients for stable training.
* **Learning Rate Decay:** Improves convergence during training.
* **Flexible CLI Interface:** Train on custom data, save models, and run inference without retraining.
* **Model Serialization:** Save trained models to JSON and load them for inference.
* **Custom Data Support:** Load training data from CSV files.
* **Per-Class Accuracy Metrics:** Track performance per diagnosis category.
* **Batch Inference:** Classify multiple samples and export results to CSV.

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

## Quick Start

### Help Message
```bash
./clinical_rnn
```

### Train on Custom Data
```bash
./clinical_rnn --mode train --data training_data.csv --config config.json --model my_model.json
```

### Single Prediction (No Retraining)
```bash
./clinical_rnn --mode infer --model my_model.json --input "Patient has fever and cough"
```

### Batch Classification
```bash
./clinical_rnn --mode infer --model my_model.json --data test_data.csv --output results.csv
```

---

## CLI Usage

The program now supports flexible command-line interfaces for training and inference without code modification.

### Mode: Training

Train a new model on custom clinical note data:

```bash
./clinical_rnn --mode train \
  --data training_data.csv \
  --config config.json \
  --model my_model.json
```

**Arguments:**
- `--mode train` - Activate training mode
- `--data <file>` - **Required.** CSV file with training data (columns: text, label)
- `--model <file>` - **Required.** Output path for the trained model (JSON format)
- `--config <file>` - **Optional.** JSON configuration file for hyperparameters (uses defaults if omitted)

**Output:**
- `my_model.json` - Trained model including architecture, vocabulary, and weights
- `training_history.csv` - Loss and accuracy metrics per epoch (for plotting)

### Mode: Inference

Use a trained model to classify new clinical notes without retraining:

```bash
./clinical_rnn --mode infer --model my_model.json --input "Patient symptoms here"
```

**Single Prediction:**
```bash
./clinical_rnn --mode infer \
  --model my_model.json \
  --input "Patient has fever and sore throat"
```

**Batch Inference:**
```bash
./clinical_rnn --mode infer \
  --model my_model.json \
  --data test_data.csv \
  --output predictions.csv
```

**Arguments:**
- `--mode infer` - Activate inference mode
- `--model <file>` - **Required.** Path to trained model JSON file
- `--input <text>` - Single clinical note to classify (alternative to --data)
- `--data <file>` - CSV file for batch inference (alternative to --input)
- `--output <file>` - **Optional.** Output file for batch results (default: predictions.csv)

**Output:**
- Single prediction: Console output with diagnosis and probabilities
- Batch mode: CSV file with predictions and per-class probabilities

---

## Data Formats

### CSV Training Format (training_data.csv)

```csv
text,label
"Patient presents with mild cough and runny nose",Cold
"Sore throat and slight fever of 99F",Cold
"Complains of body aches and chills",Flu
"High fever 102F with severe fatigue",Flu
"Productive cough with green sputum",Pneumonia
"Severe shortness of breath and fever",Pneumonia
```

**Requirements:**
- First row is header: `text,label`
- Text field: clinical note (quoted if contains commas)
- Label field: one of the diagnosis categories (Cold, Flu, Pneumonia)
- At least 2 samples recommended for meaningful training

### JSON Configuration Format (config.json)

```json
{
  "training": {
    "num_epochs": 100,
    "learning_rate": 0.01,
    "learning_rate_decay": 0.05,
    "use_data_split": true,
    "gradient_clip_norm": 1.0
  },
  "architecture": {
    "hidden_size": 10
  },
  "output": {
    "save_model": "my_model.json",
    "save_history": "training_history.csv"
  }
}
```

**Parameters:**
- `num_epochs` - Number of training passes over the dataset (default: 100)
- `learning_rate` - Step size for weight updates (default: 0.01)
- `learning_rate_decay` - Decay rate applied per epoch (default: 0.05)
- `use_data_split` - Split data into train/val/test sets (default: true)
- `gradient_clip_norm` - Maximum gradient magnitude (default: 1.0, prevents overflow)
- `hidden_size` - RNN hidden state dimension (default: 10)

### Saved Model Format (my_model.json)

The model file is a JSON document containing:

```json
{
  "architecture": {
    "vocab_size": 233,
    "hidden_size": 10,
    "output_size": 3,
    "learning_rate": 0.01,
    "gradient_clip_norm": 1.0
  },
  "vocabulary": {
    "fever": 0,
    "cough": 1,
    "patient": 2,
    ...
  },
  "diagnosis_labels": ["Cold", "Flu", "Pneumonia"],
  "weights": {
    "W_hh": [[...], [...], ...],
    "W_xh": [[...], [...], ...],
    "W_hy": [[...], [...], ...],
    "b_h": [[...]],
    "b_y": [[...]]
  }
}
```

**Portable Format:**
- Self-contained: no external files needed
- Human-readable: can inspect vocabulary and architecture
- Platform-independent: JSON is text-based

### Predictions CSV Format (predictions.csv)

Output from batch inference:

```csv
text,true_label,predicted_label,Cold_probability,Flu_probability,Pneumonia_probability
"Patient has fever and cough",Cold,Cold,0.892304,0.087231,0.020465
"Severe body aches",Flu,Flu,0.012340,0.965432,0.022228
"Productive cough with shortness of breath",Pneumonia,Pneumonia,0.034567,0.012345,0.953088
```

**Columns:**
- `text` - Original clinical note
- `true_label` - Ground truth diagnosis (from input CSV)
- `predicted_label` - Model's prediction
- `<Diagnosis>_probability` - Confidence for each class

---

## Example Workflows

### Workflow 1: Train and Deploy

**Step 1: Prepare training data**
Create `my_clinical_notes.csv` with clinical notes and labels.

**Step 2: Configure training (optional)**
Create `config.json` with desired hyperparameters.

**Step 3: Train the model**
```bash
./clinical_rnn --mode train \
  --data my_clinical_notes.csv \
  --config config.json \
  --model trained_model.json
```

**Step 4: Inspect results**
```bash
# View training curves
# Plot data from training_history.csv (epoch, train_loss, train_accuracy, val_loss, val_accuracy)

# View test set metrics
# Check console output for per-class accuracy
```

### Workflow 2: Interactive Predictions

Use a trained model for one-off predictions:

```bash
./clinical_rnn --mode infer \
  --model trained_model.json \
  --input "Patient presents with persistent cough and chest pain"
```

Output:
```
Input: "Patient presents with persistent cough and chest pain"
Predictions:
  Cold: 12.34%
  Flu: 23.45%
  Pneumonia: 64.21%
Most probable: Pneumonia
```

### Workflow 3: Batch Processing

Classify a collection of test notes:

```bash
./clinical_rnn --mode infer \
  --model trained_model.json \
  --data new_patient_notes.csv \
  --output classified_results.csv
```

Then analyze results in a spreadsheet or script:
```bash
# Count predictions per class
cut -d',' -f3 classified_results.csv | sort | uniq -c

# Compare predictions to true labels
paste <(cut -d',' -f2 new_patient_notes.csv) \
      <(cut -d',' -f3 classified_results.csv) | \
      awk -F',' '$1==$2 {print "match"} $1!=$2 {print "mismatch"}' | \
      uniq -c
```

---

## Files in this Repository

### Core Files
* `clinical_rnn.cpp` - Main executable source file (~2000 lines)
  * Matrix and vector operations
  * Activation functions (tanh, softmax) and their derivatives
  * Cross-entropy loss function
  * Full Backpropagation Through Time (BPTT) implementation
  * SimpleRNN class with forward/backward/train methods
  * **New:** CLI argument parsing and help system
  * **New:** Model serialization (save_model/load_model)
  * **New:** CSV data loading and vocabulary building
  * **New:** Batch inference and CSV output
  * **New:** JSON configuration loading

* `synthetic_training_data.h` - Contains `SIMULATED_CLINICAL_NOTES` array
  * 100 fictional clinical notes with ground truth labels
  * Used as fallback for default training mode

### Configuration & Data Files (Examples)
* `config.json` - Example training configuration
  * Hyperparameters: epochs, learning rate, decay, hidden size
  * Output paths for model and training history

* `training_data.csv` - Example training dataset
  * 18 sample clinical notes with diagnosis labels
  * Can be used as template for custom datasets

* `test_data.csv` - Example test dataset
  * 6 sample clinical notes for batch inference testing

* `README.md` - This file with comprehensive documentation

### Generated Files (after running)
* `my_model.json` - Trained model (created by --mode train)
  * Contains vocabulary, weights, and architecture
  * Used for inference with --mode infer

* `training_history.csv` - Training metrics (created by --mode train)
  * Columns: Epoch, TrainLoss, TrainAccuracy, ValLoss, ValAccuracy
  * Can be plotted to visualize learning curves

* `predictions.csv` - Batch predictions (created by --mode infer with --data)
  * Contains original text, true label, predicted label, and probabilities

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

## Recent Improvements

This project has been significantly enhanced with practical features for machine learning workflows:

### Critical Bug Fixes (Commit 4479bcb)
- Fixed BPTT to accumulate gradients across all time steps (not just final step)
- Replaced MSE+softmax with cross-entropy loss (mathematically correct for classification)
- Added gradient clipping to prevent training instability

### Code Quality Improvements (Commit b5b2bc5)
- Implemented UNK token support for unknown words
- Added train/validation/test data splits (70/15/15)
- Per-class accuracy reporting to identify weak areas
- Learning rate decay for better convergence
- Comprehensive inline documentation

### Testing & Visualization (Commit 1c69ff0)
- TrainingHistory struct to track metrics across epochs
- CSV export of loss/accuracy curves for analysis
- Consolidated tokenization logic into reusable function
- Per-epoch validation metrics during training

### Flexible CLI Architecture (Commit 50fd6a0)
- Command-line interface for train/infer modes
- Model serialization to JSON (save and reuse)
- CSV data loading for custom datasets
- JSON configuration files for hyperparameters
- Batch inference with CSV output
- Help system with usage examples
- Single-prediction mode from CLI

## Contributing

Feel free to fork this repository, experiment with the code, and suggest improvements. Please remember its purely educational nature. The codebase is now documented to support both learning and practical use.

Thanks to Claude AI for recent architectural improvements and testing enhancements.

## License

This project is open-source and available under the [MIT License](LICENSE).
