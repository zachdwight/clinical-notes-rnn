// ============================================================================
// Clinical Notes RNN - Educational Recurrent Neural Network Implementation
// ============================================================================
//
// This is an educational implementation of a Simple RNN designed to classify
// clinical notes into three diagnostic categories: Cold, Flu, or Pneumonia.
//
// Key Components:
// 1. Matrix Operations: Basic linear algebra for neural network computation
// 2. Activation Functions: tanh (hidden layer), softmax (output layer)
// 3. Loss Function: Cross-entropy loss with softmax (standard for classification)
// 4. Training: Full Backpropagation Through Time (BPTT) with gradient clipping
// 5. Evaluation: Per-class accuracy metrics to identify model weaknesses
// 6. Data Handling: Train/validation/test splits, UNK token for unknown words
//
// Improvements over basic RNNs:
// - Proper BPTT accumulates gradients across all time steps (not just final step)
// - Gradient clipping prevents exploding/vanishing gradients
// - Learning rate decay improves convergence
// - Evaluation metrics track both overall and per-class performance
// - Training history saved to CSV for loss curve visualization
//
// Limitations:
// - This implementation is NOT suitable for production use on real medical data
// - Naive matrix multiplication (use Eigen/Armadillo for production)
// - Single-sample training (no mini-batching)
// - No regularization techniques
// - Small synthetic dataset (100 samples)
//
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cctype>
#include <iomanip>

#include "synthetic_training_data.h"

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// ============================================================================
// HELPER STRUCTURES FOR TRAINING AND EVALUATION
// ============================================================================

// Training metrics for a single epoch (used to track learning progress)
struct EpochMetrics {
    int epoch;
    double train_loss;
    double train_accuracy;
    double val_loss;
    double val_accuracy;
    std::map<std::string, double> val_per_class_accuracy;
};

// Container to track all metrics throughout training (for visualization and analysis)
struct TrainingHistory {
    std::vector<EpochMetrics> epochs;
    std::map<std::string, double> test_per_class_accuracy;
    double test_loss;
    double test_accuracy;

    // Write training history to CSV file for plotting
    void save_to_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << filename << " for writing.\n";
            return;
        }

        // Header
        file << "Epoch,TrainLoss,TrainAccuracy,ValLoss,ValAccuracy\n";

        // Data rows
        for (const auto& metrics : epochs) {
            file << metrics.epoch << ","
                 << std::fixed << std::setprecision(6) << metrics.train_loss << ","
                 << std::fixed << std::setprecision(6) << metrics.train_accuracy << ","
                 << std::fixed << std::setprecision(6) << metrics.val_loss << ","
                 << std::fixed << std::setprecision(6) << metrics.val_accuracy << "\n";
        }
        file.close();
        std::cout << "Saved training history to " << filename << "\n";
    }
};

// ============================================================================
// TOKENIZATION HELPER FUNCTION
// ============================================================================

// Convert a text string to a sequence of word IDs using the vocabulary
// Unknown words are mapped to the unk_token_id parameter
std::vector<int> tokenize_text(const std::string& text,
                                const std::map<std::string, int>& vocab_map,
                                int unk_token_id) {
    std::vector<int> word_ids;
    std::string current_word;

    for (char c : text) {
        // Split on whitespace and punctuation
        if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
            if (!current_word.empty()) {
                // Convert to lowercase
                std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                               [](unsigned char ch) { return std::tolower(ch); });

                // Look up in vocabulary, use UNK for unknown words
                if (vocab_map.count(current_word)) {
                    word_ids.push_back(vocab_map.at(current_word));
                } else {
                    word_ids.push_back(unk_token_id);
                }
                current_word.clear();
            }
        } else {
            current_word += c;
        }
    }

    // Don't forget the last word
    if (!current_word.empty()) {
        std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        if (vocab_map.count(current_word)) {
            word_ids.push_back(vocab_map.at(current_word));
        } else {
            word_ids.push_back(unk_token_id);
        }
    }

    return word_ids;
}

// ============================================================================
// CLI AND SERIALIZATION UTILITIES
// ============================================================================

// Command-line argument container
struct ProgramArgs {
    std::string mode = "train";           // "train" or "infer"
    std::string model_file = "";          // Path to model file
    std::string data_file = "";           // Path to training/inference data
    std::string config_file = "";         // Path to config file
    std::string input_text = "";          // Single input text for inference
    std::string output_file = "";         // Output file for batch results
};

// Training configuration container
struct TrainingConfig {
    int num_epochs = 100;
    double learning_rate = 0.01;
    double learning_rate_decay = 0.05;
    bool use_data_split = true;
    double gradient_clip_norm = 1.0;
    int hidden_size = 10;
    std::string save_model = "model.json";
    std::string save_history = "training_history.csv";
};

// Prediction result for batch inference
struct PredictionResult {
    std::string text;
    std::string true_label;
    std::string predicted_label;
    std::vector<double> probabilities;
    std::vector<std::string> class_names;
};

// ============================================================================
// HELP MESSAGE
// ============================================================================

void print_help_message() {
    std::cout << R"(
================================================================================
  Clinical Notes RNN - Flexible Training and Inference Tool
================================================================================

USAGE:
  ./clinical_rnn --mode train --data <file> --config <file> --model <file>
  ./clinical_rnn --mode infer --model <file> --input "<text>"
  ./clinical_rnn --mode infer --model <file> --data <file> --output <file>

MODES:
  train     Train a new model on custom data
  infer     Use a trained model to classify clinical notes

REQUIRED ARGUMENTS:
  --mode <train|infer>      Operation mode (default: train)
  --model <path>            Model file path (save for train, load for infer)

TRAINING ARGUMENTS:
  --data <path>             CSV file with training data (required for train)
  --config <path>           JSON config file with hyperparameters (optional)

INFERENCE ARGUMENTS:
  --input <text>            Single clinical note to classify
                            OR use --data for batch inference
  --data <path>             CSV file with test data (for batch inference)
  --output <path>           Output CSV file for batch results (optional)

EXAMPLES:

1. Train a model:
   ./clinical_rnn --mode train \
     --data training_notes.csv \
     --config training_config.json \
     --model my_model.json

2. Classify a single note:
   ./clinical_rnn --mode infer \
     --model my_model.json \
     --input "Patient has fever and sore throat"

3. Batch classification:
   ./clinical_rnn --mode infer \
     --model my_model.json \
     --data test_notes.csv \
     --output predictions.csv

DATA FORMATS:

CSV Format (training_notes.csv):
  text,label
  "Patient presents with fever and cough",Cold
  "Severe body aches and chills",Flu
  "Productive cough and shortness of breath",Pneumonia

JSON Config (training_config.json):
  {
    "training": {
      "num_epochs": 100,
      "learning_rate": 0.01,
      "learning_rate_decay": 0.05,
      "use_data_split": true
    },
    "architecture": {
      "hidden_size": 10
    }
  }

Saved Model (my_model.json):
  - Contains architecture, vocabulary, class labels, and trained weights
  - Can be used for inference without retraining
  - Portable across systems

================================================================================
)" << std::endl;
}

// ============================================================================
// ARGUMENT PARSING
// ============================================================================

ProgramArgs parse_arguments(int argc, char* argv[]) {
    ProgramArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            args.model_file = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            args.data_file = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            args.config_file = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            args.input_text = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_file = argv[++i];
        }
    }

    return args;
}

// ============================================================================
// JSON SERIALIZATION UTILITIES
// ============================================================================

// Simple JSON helpers - escapes strings for JSON
std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

// Convert Matrix to JSON string
std::string matrix_to_json(const Matrix& mat) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < mat.size(); ++i) {
        ss << "[";
        for (size_t j = 0; j < mat[i].size(); ++j) {
            ss << std::fixed << std::setprecision(10) << mat[i][j];
            if (j < mat[i].size() - 1) ss << ",";
        }
        ss << "]";
        if (i < mat.size() - 1) ss << ",";
    }
    ss << "]";
    return ss.str();
}

// Parse Matrix from JSON array string
Matrix json_to_matrix(const std::string& json_str) {
    Matrix mat;
    std::stringstream ss(json_str);
    char c;

    ss >> c;  // consume '['
    while (ss >> c && c != ']') {
        if (c == '[') {
            Vector row;
            double val;
            while (ss >> val) {
                row.push_back(val);
                ss >> c;
                if (c != ',') break;
            }
            if (!row.empty()) mat.push_back(row);
        }
    }

    return mat;
}

// ============================================================================
// CONFIG FILE LOADING
// ============================================================================

TrainingConfig load_config_json(const std::string& filename) {
    TrainingConfig config;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << "\n";
        std::cerr << "Using default configuration.\n";
        return config;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Simple JSON parsing - look for key: value pairs
    auto extract_int = [&](const std::string& key) {
        auto pos = content.find("\"" + key + "\"");
        if (pos != std::string::npos) {
            auto colon = content.find(":", pos);
            if (colon != std::string::npos) {
                auto comma = content.find(",", colon);
                if (comma == std::string::npos) comma = content.find("}", colon);
                std::string value_str = content.substr(colon + 1, comma - colon - 1);
                // Trim whitespace
                value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
                value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);
                try { return std::stoi(value_str); } catch (...) {}
            }
        }
        return 0;
    };

    auto extract_double = [&](const std::string& key) {
        auto pos = content.find("\"" + key + "\"");
        if (pos != std::string::npos) {
            auto colon = content.find(":", pos);
            if (colon != std::string::npos) {
                auto comma = content.find(",", colon);
                if (comma == std::string::npos) comma = content.find("}", colon);
                std::string value_str = content.substr(colon + 1, comma - colon - 1);
                // Trim whitespace
                value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
                value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);
                try { return std::stod(value_str); } catch (...) {}
            }
        }
        return 0.0;
    };

    auto extract_bool = [&](const std::string& key) {
        auto pos = content.find("\"" + key + "\"");
        if (pos != std::string::npos) {
            auto colon = content.find(":", pos);
            if (colon != std::string::npos) {
                std::string rest = content.substr(colon + 1);
                return rest.find("true") < rest.find("false");
            }
        }
        return false;
    };

    // Extract configuration values
    config.num_epochs = extract_int("num_epochs");
    if (config.num_epochs == 0) config.num_epochs = 100;

    config.learning_rate = extract_double("learning_rate");
    if (config.learning_rate < 0.0001) config.learning_rate = 0.01;

    config.learning_rate_decay = extract_double("learning_rate_decay");
    config.use_data_split = extract_bool("use_data_split");
    config.gradient_clip_norm = extract_double("gradient_clip_norm");
    if (config.gradient_clip_norm < 0.1) config.gradient_clip_norm = 1.0;

    config.hidden_size = extract_int("hidden_size");
    if (config.hidden_size == 0) config.hidden_size = 10;

    file.close();
    return config;
}

// ============================================================================
// CSV DATA LOADING
// ============================================================================

std::vector<std::pair<std::string, std::string>> load_dataset_csv(const std::string& filename) {
    std::vector<std::pair<std::string, std::string>> dataset;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file: " << filename << "\n";
        return dataset;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;  // Skip header
            continue;
        }

        if (line.empty()) continue;

        // Parse CSV line (handle quoted fields)
        size_t quote_pos = line.find('"');
        if (quote_pos == std::string::npos) {
            // Simple case: no quotes
            auto comma = line.find(',');
            if (comma != std::string::npos) {
                std::string text = line.substr(0, comma);
                std::string label = line.substr(comma + 1);
                // Trim whitespace
                label.erase(0, label.find_first_not_of(" \t\n\r"));
                dataset.push_back({text, label});
            }
        } else {
            // Quoted field
            size_t end_quote = line.find('"', quote_pos + 1);
            if (end_quote != std::string::npos) {
                std::string text = line.substr(quote_pos + 1, end_quote - quote_pos - 1);
                // Find label after closing quote
                size_t comma = line.find(',', end_quote);
                if (comma != std::string::npos) {
                    std::string label = line.substr(comma + 1);
                    label.erase(0, label.find_first_not_of(" \t\n\r"));
                    dataset.push_back({text, label});
                }
            }
        }
    }

    file.close();
    std::cout << "Loaded " << dataset.size() << " samples from " << filename << "\n";
    return dataset;
}

// ============================================================================
// VOCABULARY BUILDING
// ============================================================================

std::map<std::string, int> build_vocabulary(const std::vector<std::pair<std::string, std::string>>& dataset) {
    std::map<std::string, int> vocab;
    int word_id = 0;

    for (const auto& note_pair : dataset) {
        const std::string& text = note_pair.first;
        std::string current_word;

        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                if (!current_word.empty()) {
                    std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                   [](unsigned char ch) { return std::tolower(ch); });
                    if (vocab.find(current_word) == vocab.end()) {
                        vocab[current_word] = word_id++;
                    }
                    current_word.clear();
                }
            } else {
                current_word += c;
            }
        }

        if (!current_word.empty()) {
            std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                           [](unsigned char ch) { return std::tolower(ch); });
            if (vocab.find(current_word) == vocab.end()) {
                vocab[current_word] = word_id++;
            }
        }
    }

    std::cout << "Built vocabulary with " << vocab.size() << " unique words\n";
    return vocab;
}

void printMatrix(const Matrix& mat, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << ":\n";
    }
    for (const auto& row : mat) {
        for (double val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }
}

Matrix subtract(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    if (rows != B.size() || cols != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions mismatch for subtraction.");
    }
    Matrix C(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// O(n³) naive matrix multiplication. For production use, consider Eigen or Armadillo libraries
// which use optimized algorithms and SIMD instructions. This naive approach is fine for
// educational purposes and small matrices (vocab_size < 300, hidden_size < 50).
Matrix multiply(const Matrix& A, const Matrix& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    }

    Matrix C(rowsA, Vector(colsB, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix add(const Matrix& A, const Matrix& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();

    if (rowsA != B.size() || colsA != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions mismatch for addition.");
    }

    Matrix C(rowsA, Vector(colsA));
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsA; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Matrix hadamard_product(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    if (rows != B.size() || cols != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions mismatch for Hadamard product.");
    }
    Matrix C(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }
    return C;
}

Matrix scale(const Matrix& mat, double scalar) {
    Matrix result = mat;
    for (auto& row : result) {
        for (double& val : row) {
            val *= scalar;
        }
    }
    return result;
}

Matrix transpose(const Matrix& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Matrix transposed(cols, Vector(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

Matrix apply_function(const Matrix& mat, double (*func)(double)) {
    Matrix result = mat;
    for (auto& row : result) {
        for (double& val : row) {
            val = func(val);
        }
    }
    return result;
}

double tanh_activation(double x) {
    return std::tanh(x);
}

double tanh_derivative(double x) {
    return 1.0 - std::pow(std::tanh(x), 2);
}

Matrix tanh_derivative_from_output(const Matrix& output) {
    Matrix result = output;
    for (auto& row : result) {
        for (double& val : row) {
            val = 1.0 - (val * val);
        }
    }
    return result;
}

Vector softmax(const Vector& vec) {
    Vector exp_vec(vec.size());
    double sum_exp = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        exp_vec[i] = std::exp(vec[i]);
        sum_exp += exp_vec[i];
    }

    Vector result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = exp_vec[i] / sum_exp;
    }
    return result;
}

// FIXED #2: Cross-entropy loss with softmax (replaces MSE)
double cross_entropy_loss(const Vector& predictions, const Vector& targets) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (targets[i] > 0) {  // Only for true class (one-hot encoded)
            loss -= targets[i] * std::log(std::max(predictions[i], 1e-10));
        }
    }
    return loss;
}

// FIXED #2: Cross-entropy derivative with softmax is simply (predictions - targets)
Vector cross_entropy_derivative(const Vector& predictions, const Vector& targets) {
    Vector grad(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad[i] = predictions[i] - targets[i];
    }
    return grad;
}

// FIXED #3: Gradient clipping to prevent exploding gradients
double clip_value(double value, double max_val = 1.0) {
    if (value > max_val) return max_val;
    if (value < -max_val) return -max_val;
    return value;
}

Matrix clip_gradients(const Matrix& gradients, double max_norm = 1.0) {
    Matrix clipped = gradients;
    for (auto& row : clipped) {
        for (double& val : row) {
            val = clip_value(val, max_norm);
        }
    }
    return clipped;
}

class SimpleRNN {
public:
    int vocab_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    double gradient_clip_norm;
    int unk_token_id;  // ID for unknown words (reserved as last vocab slot)

    Matrix W_hh;
    Matrix W_xh;
    Matrix W_hy;

    Matrix b_h;
    Matrix b_y;

    std::vector<Matrix> hidden_states_history;
    std::vector<Matrix> pre_tanh_activations;

    SimpleRNN(int vocab_s, int hidden_s, int output_s, double lr)
        : vocab_size(vocab_s), hidden_size(hidden_s), output_size(output_s), learning_rate(lr),
          gradient_clip_norm(1.0), unk_token_id(vocab_s - 1) {  // Last vocab index reserved for UNK

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d_tanh(0.0, std::sqrt(2.0 / (hidden_size + vocab_size)));
        std::normal_distribution<> d_linear(0.0, std::sqrt(2.0 / (hidden_size + output_size)));

        auto initialize_matrix_tanh = [&](size_t rows, size_t cols) {
            Matrix mat(rows, Vector(cols));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[i][j] = d_tanh(gen);
                }
            }
            return mat;
        };
        auto initialize_matrix_linear = [&](size_t rows, size_t cols) {
            Matrix mat(rows, Vector(cols));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[i][j] = d_linear(gen);
                }
            }
            return mat;
        };
        auto initialize_matrix_bias = [&](size_t rows, size_t cols) {
            return Matrix(rows, Vector(cols, 0.0));
        };

        W_hh = initialize_matrix_tanh(hidden_size, hidden_size);
        W_xh = initialize_matrix_tanh(vocab_size, hidden_size);
        W_hy = initialize_matrix_linear(hidden_size, output_size);

        b_h = initialize_matrix_bias(1, hidden_size);
        b_y = initialize_matrix_bias(1, output_size);
    }

    Matrix one_hot_encode(int word_id) const {
        Matrix vector(1, Vector(vocab_size, 0.0));
        // Use word_id if valid, otherwise use UNK token (last vocab index)
        if (word_id < 0 || word_id >= vocab_size) {
            vector[0][unk_token_id] = 1.0;
        } else {
            vector[0][word_id] = 1.0;
        }
        return vector;
    }

    Vector forward(const std::vector<int>& word_ids_sequence) {
        hidden_states_history.clear();
        pre_tanh_activations.clear();

        Matrix h_t(1, Vector(hidden_size, 0.0));
        hidden_states_history.push_back(h_t);

        for (int word_id : word_ids_sequence) {
            Matrix x_t = one_hot_encode(word_id);

            Matrix h_prev_whh = multiply(hidden_states_history.back(), W_hh);
            Matrix x_t_wxh = multiply(x_t, W_xh);

            Matrix sum_inputs = add(h_prev_whh, x_t_wxh);
            Matrix h_t_unactivated = add(sum_inputs, b_h);

            pre_tanh_activations.push_back(h_t_unactivated);

            h_t = apply_function(h_t_unactivated, tanh_activation);
            hidden_states_history.push_back(h_t);
        }

        Matrix final_h_t = hidden_states_history.back();
        Matrix output_unactivated_matrix = add(
            multiply(final_h_t, W_hy),
            b_y
        );

        return softmax(output_unactivated_matrix[0]);
    }

    // FIXED #1: Proper Backpropagation Through Time (BPTT)
    void backward(const std::vector<int>& word_ids_sequence, const Vector& predictions, const Vector& targets) {
        // Use cross-entropy derivative instead of MSE
        Vector d_output_pred = cross_entropy_derivative(predictions, targets);
        Matrix d_output_mat(1, d_output_pred);

        // Output layer gradients
        Matrix final_h_t = hidden_states_history.back();
        Matrix d_W_hy = multiply(transpose(final_h_t), d_output_mat);
        Matrix d_b_y = d_output_mat;

        // Initialize cumulative gradients for recurrent weights
        Matrix d_W_hh_cumulative(hidden_size, Vector(hidden_size, 0.0));
        Matrix d_W_xh_cumulative(vocab_size, Vector(hidden_size, 0.0));
        Matrix d_b_h_cumulative(1, Vector(hidden_size, 0.0));

        // Backpropagate through time: loop backward through all time steps
        Matrix d_h_t_prop = multiply(d_output_mat, transpose(W_hy));

        int T = word_ids_sequence.size();
        for (int t = T - 1; t >= 0; --t) {
            // Backprop through tanh activation
            Matrix d_h_t_unactivated = hadamard_product(d_h_t_prop, tanh_derivative_from_output(hidden_states_history[t + 1]));

            // Accumulate gradients for recurrent weights at time step t
            Matrix h_prev = hidden_states_history[t];
            Matrix x_t = one_hot_encode(word_ids_sequence[t]);

            Matrix d_W_hh_t = multiply(transpose(h_prev), d_h_t_unactivated);
            Matrix d_W_xh_t = multiply(transpose(x_t), d_h_t_unactivated);

            // Add to cumulative gradients
            d_W_hh_cumulative = add(d_W_hh_cumulative, d_W_hh_t);
            d_W_xh_cumulative = add(d_W_xh_cumulative, d_W_xh_t);
            d_b_h_cumulative = add(d_b_h_cumulative, d_h_t_unactivated);

            // Backprop gradient to previous time step
            if (t > 0) {
                d_h_t_prop = multiply(d_h_t_unactivated, transpose(W_hh));
            }
        }

        // FIXED #3: Apply gradient clipping before weight updates
        d_W_hy = clip_gradients(d_W_hy, gradient_clip_norm);
        d_b_y = clip_gradients(d_b_y, gradient_clip_norm);
        d_W_hh_cumulative = clip_gradients(d_W_hh_cumulative, gradient_clip_norm);
        d_W_xh_cumulative = clip_gradients(d_W_xh_cumulative, gradient_clip_norm);
        d_b_h_cumulative = clip_gradients(d_b_h_cumulative, gradient_clip_norm);

        // Update weights with clipped gradients
        W_hy = subtract(W_hy, scale(d_W_hy, learning_rate));
        b_y = subtract(b_y, scale(d_b_y, learning_rate));
        W_hh = subtract(W_hh, scale(d_W_hh_cumulative, learning_rate));
        W_xh = subtract(W_xh, scale(d_W_xh_cumulative, learning_rate));
        b_h = subtract(b_h, scale(d_b_h_cumulative, learning_rate));
    }

    // Split data into train/val/test: 70% train, 15% val, 15% test
    void split_data(const std::vector<std::pair<std::string, std::string>>& all_data,
                    std::vector<std::pair<std::string, std::string>>& train_data,
                    std::vector<std::pair<std::string, std::string>>& val_data,
                    std::vector<std::pair<std::string, std::string>>& test_data) {
        std::vector<std::pair<std::string, std::string>> shuffled_data = all_data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

        size_t train_size = (shuffled_data.size() * 7) / 10;
        size_t val_size = (shuffled_data.size() * 15) / 100;

        train_data.assign(shuffled_data.begin(), shuffled_data.begin() + train_size);
        val_data.assign(shuffled_data.begin() + train_size, shuffled_data.begin() + train_size + val_size);
        test_data.assign(shuffled_data.begin() + train_size + val_size, shuffled_data.end());
    }

    // Evaluate on a dataset and return accuracy metrics
    struct EvaluationMetrics {
        double loss;
        double accuracy;
        std::map<std::string, double> per_class_accuracy;
    };

    EvaluationMetrics evaluate(const std::vector<std::pair<std::string, std::string>>& data,
                               const std::map<std::string, int>& vocab_map,
                               const std::vector<std::string>& diagnosis_labels) {
        EvaluationMetrics metrics;
        metrics.loss = 0.0;
        metrics.accuracy = 0.0;
        for (const auto& label : diagnosis_labels) {
            metrics.per_class_accuracy[label] = 0.0;
        }

        std::map<std::string, int> class_correct;
        std::map<std::string, int> class_total;
        for (const auto& label : diagnosis_labels) {
            class_correct[label] = 0;
            class_total[label] = 0;
        }

        int correct = 0;
        for (const auto& note_pair : data) {
            const std::string& note_text = note_pair.first;
            const std::string& true_label = note_pair.second;

            // Tokenize text using helper function
            std::vector<int> word_ids = tokenize_text(note_text, vocab_map, unk_token_id);
            if (word_ids.empty()) continue;

            Vector predictions = forward(word_ids);
            metrics.loss += cross_entropy_loss(predictions, create_target_vector(true_label, diagnosis_labels));

            auto max_it = std::max_element(predictions.begin(), predictions.end());
            int predicted_index = std::distance(predictions.begin(), max_it);
            std::string predicted_label = diagnosis_labels[predicted_index];

            class_total[true_label]++;
            if (predicted_index == std::distance(diagnosis_labels.begin(),
                    std::find(diagnosis_labels.begin(), diagnosis_labels.end(), true_label))) {
                correct++;
                class_correct[true_label]++;
            }
        }

        if (data.empty()) return metrics;

        metrics.loss /= data.size();
        metrics.accuracy = static_cast<double>(correct) / data.size();
        for (const auto& label : diagnosis_labels) {
            if (class_total[label] > 0) {
                metrics.per_class_accuracy[label] = static_cast<double>(class_correct[label]) / class_total[label];
            }
        }

        return metrics;
    }

    Vector create_target_vector(const std::string& label, const std::vector<std::string>& labels) {
        Vector target(labels.size(), 0.0);
        auto it = std::find(labels.begin(), labels.end(), label);
        if (it != labels.end()) {
            target[std::distance(labels.begin(), it)] = 1.0;
        }
        return target;
    }

    // Main training function with full metrics tracking
    // Returns a TrainingHistory object containing loss/accuracy curves for analysis and visualization
    TrainingHistory train(const std::vector<std::pair<std::string, std::string>>& training_data,
                          const std::map<std::string, int>& vocab_map,
                          const std::vector<std::string>& diagnosis_labels,
                          int num_epochs,
                          bool use_data_split = true,
                          double learning_rate_decay = 0.0) {
        TrainingHistory history;

        // Split data if requested
        std::vector<std::pair<std::string, std::string>> train_set = training_data;
        std::vector<std::pair<std::string, std::string>> val_set;
        std::vector<std::pair<std::string, std::string>> test_set;

        if (use_data_split && training_data.size() > 10) {
            split_data(training_data, train_set, val_set, test_set);
            std::cout << "Data split: Train=" << train_set.size() << ", Val=" << val_set.size()
                      << ", Test=" << test_set.size() << "\n";
        }

        std::cout << "\n--- Starting Training (BPTT, Cross-Entropy Loss, Gradient Clipping, UNK Tokens) ---\n";
        std::cout << "Learning rate: " << learning_rate;
        if (learning_rate_decay > 0) {
            std::cout << " (with decay: " << learning_rate_decay << ")";
        }
        std::cout << "\n";

        double initial_lr = learning_rate;

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Apply learning rate decay
            if (learning_rate_decay > 0) {
                learning_rate = initial_lr / (1.0 + learning_rate_decay * epoch);
            }

            double epoch_loss = 0.0;
            int correct_predictions = 0;
            int total_predictions = 0;

            for (const auto& note_pair : train_set) {
                const std::string& note_text = note_pair.first;
                const std::string& true_diagnosis_label = note_pair.second;

                // Tokenize text using helper function
                std::vector<int> word_ids = tokenize_text(note_text, vocab_map, unk_token_id);
                if (word_ids.empty()) {
                    continue;
                }

                Vector predictions = forward(word_ids);
                Vector targets = create_target_vector(true_diagnosis_label, diagnosis_labels);

                double current_loss = cross_entropy_loss(predictions, targets);
                epoch_loss += current_loss;

                backward(word_ids, predictions, targets);

                auto max_it = std::max_element(predictions.begin(), predictions.end());
                int predicted_index = std::distance(predictions.begin(), max_it);
                int true_index = std::distance(diagnosis_labels.begin(), std::find(diagnosis_labels.begin(), diagnosis_labels.end(), true_diagnosis_label));

                if (predicted_index == true_index) {
                    correct_predictions++;
                }
                total_predictions++;
            }

            // Compute per-epoch training metrics
            double train_loss = epoch_loss / total_predictions;
            double train_accuracy = static_cast<double>(correct_predictions) / total_predictions;

            // Print training metrics
            std::cout << "Epoch " << epoch + 1 << " | Train Loss: " << std::fixed << std::setprecision(6)
                      << train_loss << " | Train Acc: " << std::fixed << std::setprecision(2)
                      << train_accuracy * 100.0 << "%";

            // Create epoch metrics record
            EpochMetrics epoch_metrics;
            epoch_metrics.epoch = epoch + 1;
            epoch_metrics.train_loss = train_loss;
            epoch_metrics.train_accuracy = train_accuracy;

            // Evaluate on validation set if available
            if (!val_set.empty()) {
                EvaluationMetrics val_metrics = evaluate(val_set, vocab_map, diagnosis_labels);
                epoch_metrics.val_loss = val_metrics.loss;
                epoch_metrics.val_accuracy = val_metrics.accuracy;
                epoch_metrics.val_per_class_accuracy = val_metrics.per_class_accuracy;

                std::cout << " | Val Loss: " << std::fixed << std::setprecision(6) << val_metrics.loss
                          << " | Val Acc: " << std::fixed << std::setprecision(2) << val_metrics.accuracy * 100.0 << "%";
            }
            std::cout << "\n";

            // Save epoch metrics to history
            history.epochs.push_back(epoch_metrics);
        }

        // Final evaluation on test set
        if (!test_set.empty()) {
            std::cout << "\n--- Test Set Evaluation ---\n";
            EvaluationMetrics test_metrics = evaluate(test_set, vocab_map, diagnosis_labels);
            history.test_loss = test_metrics.loss;
            history.test_accuracy = test_metrics.accuracy;
            history.test_per_class_accuracy = test_metrics.per_class_accuracy;

            std::cout << "Test Loss: " << std::fixed << std::setprecision(6) << test_metrics.loss << "\n";
            std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_metrics.accuracy * 100.0 << "%\n";
            std::cout << "Per-Class Accuracy:\n";
            for (const auto& label : diagnosis_labels) {
                std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
                          << test_metrics.per_class_accuracy[label] * 100.0 << "%\n";
            }
        }

        std::cout << "--- Training Finished ---\n";
        return history;
    }

    // Save trained model to JSON file (includes architecture, vocab, and weights)
    void save_model(const std::string& filename,
                    const std::map<std::string, int>& vocabulary,
                    const std::vector<std::string>& diagnosis_labels = {}) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << "\n";
            return;
        }

        file << "{\n";

        // Architecture section
        file << "  \"architecture\": {\n";
        file << "    \"vocab_size\": " << vocab_size << ",\n";
        file << "    \"hidden_size\": " << hidden_size << ",\n";
        file << "    \"output_size\": " << output_size << ",\n";
        file << "    \"learning_rate\": " << std::fixed << std::setprecision(10) << learning_rate << ",\n";
        file << "    \"gradient_clip_norm\": " << gradient_clip_norm << "\n";
        file << "  },\n";

        // Vocabulary section
        file << "  \"vocabulary\": {\n";
        int vocab_count = 0;
        for (const auto& pair : vocabulary) {
            file << "    \"" << json_escape(pair.first) << "\": " << pair.second;
            if (++vocab_count < vocabulary.size()) file << ",";
            file << "\n";
        }
        file << "  },\n";

        // Diagnosis labels section
        file << "  \"diagnosis_labels\": [";
        if (!diagnosis_labels.empty()) {
            file << "\n";
            for (size_t i = 0; i < diagnosis_labels.size(); ++i) {
                file << "    \"" << diagnosis_labels[i] << "\"";
                if (i < diagnosis_labels.size() - 1) file << ",";
                file << "\n";
            }
            file << "  ";
        }
        file << "],\n";

        // Weights section
        file << "  \"weights\": {\n";
        file << "    \"W_hh\": " << matrix_to_json(W_hh) << ",\n";
        file << "    \"W_xh\": " << matrix_to_json(W_xh) << ",\n";
        file << "    \"W_hy\": " << matrix_to_json(W_hy) << ",\n";
        file << "    \"b_h\": " << matrix_to_json(b_h) << ",\n";
        file << "    \"b_y\": " << matrix_to_json(b_y) << "\n";
        file << "  }\n";

        file << "}\n";
        file.close();

        std::cout << "Model saved to " << filename << "\n";
    }

    // Load trained model from JSON file
    bool load_model(const std::string& filename,
                    std::map<std::string, int>& vocabulary,
                    std::vector<std::string>& diagnosis_labels) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open model file: " << filename << "\n";
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        // Simple JSON parsing
        // Extract W_hh
        auto extract_matrix = [&](const std::string& key) {
            size_t pos = content.find("\"" + key + "\"");
            if (pos != std::string::npos) {
                size_t start = content.find("[", pos);
                if (start != std::string::npos) {
                    int bracket_count = 0;
                    size_t end = start;
                    for (size_t i = start; i < content.size(); ++i) {
                        if (content[i] == '[') bracket_count++;
                        if (content[i] == ']') bracket_count--;
                        if (bracket_count == 0) {
                            end = i + 1;
                            break;
                        }
                    }
                    std::string matrix_str = content.substr(start, end - start);
                    return json_to_matrix(matrix_str);
                }
            }
            return Matrix();
        };

        // Load weights
        W_hh = extract_matrix("W_hh");
        W_xh = extract_matrix("W_xh");
        W_hy = extract_matrix("W_hy");
        b_h = extract_matrix("b_h");
        b_y = extract_matrix("b_y");

        // Extract vocabulary
        auto vocab_start = content.find("\"vocabulary\"");
        if (vocab_start != std::string::npos) {
            auto vocab_section_start = content.find("{", vocab_start);
            auto vocab_section_end = content.find("}", vocab_start);
            if (vocab_section_start != std::string::npos && vocab_section_end != std::string::npos) {
                std::string vocab_str = content.substr(vocab_section_start + 1, vocab_section_end - vocab_section_start - 1);
                std::istringstream iss(vocab_str);
                std::string line;
                while (std::getline(iss, line)) {
                    auto quote_pos = line.find("\"");
                    if (quote_pos != std::string::npos) {
                        auto end_quote = line.find("\"", quote_pos + 1);
                        if (end_quote != std::string::npos) {
                            std::string word = line.substr(quote_pos + 1, end_quote - quote_pos - 1);
                            auto colon = line.find(":", end_quote);
                            if (colon != std::string::npos) {
                                auto val_str = line.substr(colon + 1);
                                try {
                                    int id = std::stoi(val_str);
                                    vocabulary[word] = id;
                                } catch (...) {}
                            }
                        }
                    }
                }
            }
        }

        // Extract diagnosis labels
        auto labels_pos = content.find("\"diagnosis_labels\"");
        if (labels_pos != std::string::npos) {
            auto arr_start = content.find("[", labels_pos);
            auto arr_end = content.find("]", labels_pos);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string labels_str = content.substr(arr_start + 1, arr_end - arr_start - 1);
                std::istringstream iss(labels_str);
                std::string item;
                while (std::getline(iss, item, ',')) {
                    // Trim and remove quotes
                    item.erase(0, item.find_first_not_of(" \t\n\r\""));
                    item.erase(item.find_last_not_of(" \t\n\r\"") + 1);
                    if (!item.empty()) {
                        diagnosis_labels.push_back(item);
                    }
                }
            }
        }

        std::cout << "Model loaded from " << filename << "\n";
        std::cout << "  Vocabulary size: " << vocabulary.size() << "\n";
        std::cout << "  Diagnosis labels: " << diagnosis_labels.size() << "\n";

        return true;
    }
};

// ============================================================================
// BATCH INFERENCE AND OUTPUT
// ============================================================================

void predict_single(SimpleRNN& rnn,
                    const std::map<std::string, int>& vocabulary,
                    const std::string& input_text,
                    const std::vector<std::string>& diagnoses) {
    std::vector<int> word_ids = tokenize_text(input_text, vocabulary, rnn.unk_token_id);
    if (word_ids.empty()) {
        std::cout << "Warning: No recognized words in input\n";
        return;
    }

    Vector predictions = rnn.forward(word_ids);

    std::cout << "Input: \"" << input_text << "\"\n";
    std::cout << "Predictions:\n";
    for (size_t i = 0; i < diagnoses.size(); ++i) {
        std::cout << "  " << diagnoses[i] << ": "
                  << std::fixed << std::setprecision(2) << (predictions[i] * 100.0) << "%\n";
    }

    auto max_it = std::max_element(predictions.begin(), predictions.end());
    int predicted_index = std::distance(predictions.begin(), max_it);
    std::cout << "Most probable: " << diagnoses[predicted_index] << "\n";
}

std::vector<PredictionResult> batch_predict(SimpleRNN& rnn,
                                            const std::map<std::string, int>& vocabulary,
                                            const std::vector<std::pair<std::string, std::string>>& test_data,
                                            const std::vector<std::string>& diagnoses) {
    std::vector<PredictionResult> results;

    for (const auto& note_pair : test_data) {
        std::vector<int> word_ids = tokenize_text(note_pair.first, vocabulary, rnn.unk_token_id);
        if (word_ids.empty()) continue;

        Vector predictions = rnn.forward(word_ids);

        auto max_it = std::max_element(predictions.begin(), predictions.end());
        int predicted_index = std::distance(predictions.begin(), max_it);

        PredictionResult result;
        result.text = note_pair.first;
        result.true_label = note_pair.second;
        result.predicted_label = diagnoses[predicted_index];
        result.class_names = diagnoses;
        result.probabilities.assign(predictions.begin(), predictions.end());

        results.push_back(result);
    }

    return results;
}

void save_predictions_csv(const std::string& filename,
                          const std::vector<PredictionResult>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return;
    }

    // Header
    file << "text,true_label,predicted_label";
    if (!results.empty() && !results[0].class_names.empty()) {
        for (const auto& class_name : results[0].class_names) {
            file << "," << class_name << "_probability";
        }
    }
    file << "\n";

    // Data rows
    for (const auto& result : results) {
        file << "\"" << result.text << "\"," << result.true_label << "," << result.predicted_label;
        for (double prob : result.probabilities) {
            file << "," << std::fixed << std::setprecision(6) << prob;
        }
        file << "\n";
    }

    file.close();
    std::cout << "Results saved to " << filename << "\n";
}

int main(int argc, char* argv[]) {
    // Show help if no arguments provided
    if (argc == 1) {
        print_help_message();
        return 0;
    }

    // Parse command-line arguments
    ProgramArgs args = parse_arguments(argc, argv);

    std::vector<std::string> diagnoses = {"Cold", "Flu", "Pneumonia"};

    if (args.mode == "train") {
        // ====== TRAINING MODE ======
        if (args.data_file.empty() || args.model_file.empty()) {
            std::cerr << "Error: --data and --model required for training mode\n";
            return 1;
        }

        // Load config or use defaults
        TrainingConfig config = load_config_json(args.config_file.empty() ? "default" : args.config_file);

        // Load training data
        auto dataset = load_dataset_csv(args.data_file);
        if (dataset.empty()) {
            std::cerr << "Error: No data loaded from " << args.data_file << "\n";
            return 1;
        }

        // Build vocabulary from data
        std::map<std::string, int> vocabulary = build_vocabulary(dataset);

        // Create and train RNN
        int vocab_size = vocabulary.size() + 1;  // +1 for UNK token
        SimpleRNN rnn(vocab_size, config.hidden_size, diagnoses.size(), config.learning_rate);
        rnn.gradient_clip_norm = config.gradient_clip_norm;

        std::cout << "\nStarting training...\n";
        std::cout << "  Data: " << dataset.size() << " samples\n";
        std::cout << "  Vocabulary: " << vocabulary.size() << " words\n";
        std::cout << "  Epochs: " << config.num_epochs << "\n";
        std::cout << "  Hidden size: " << config.hidden_size << "\n";

        TrainingHistory history = rnn.train(dataset, vocabulary, diagnoses, config.num_epochs,
                                            config.use_data_split, config.learning_rate_decay);

        // Save model with embedded vocabulary
        rnn.save_model(args.model_file, vocabulary, diagnoses);

        // Save training history
        history.save_to_csv(config.save_history);

        std::cout << "\nTraining complete!\n";

    } else if (args.mode == "infer") {
        // ====== INFERENCE MODE ======
        if (args.model_file.empty()) {
            std::cerr << "Error: --model required for inference mode\n";
            return 1;
        }

        if (args.input_text.empty() && args.data_file.empty()) {
            std::cerr << "Error: --input or --data required for inference mode\n";
            return 1;
        }

        // Load model (includes vocabulary and weights)
        SimpleRNN rnn(1, 10, diagnoses.size(), 0.01);  // Dummy init, will be overwritten
        std::map<std::string, int> vocabulary;
        std::vector<std::string> loaded_diagnoses;

        if (!rnn.load_model(args.model_file, vocabulary, loaded_diagnoses)) {
            std::cerr << "Error: Failed to load model from " << args.model_file << "\n";
            return 1;
        }

        // Update diagnoses from model if loaded
        if (!loaded_diagnoses.empty()) {
            diagnoses = loaded_diagnoses;
        }

        if (!args.input_text.empty()) {
            // Single prediction
            std::cout << "\n--- Single Prediction ---\n";
            predict_single(rnn, vocabulary, args.input_text, diagnoses);

        } else if (!args.data_file.empty()) {
            // Batch inference
            std::cout << "\n--- Batch Inference ---\n";
            auto test_data = load_dataset_csv(args.data_file);
            if (test_data.empty()) {
                std::cerr << "Error: No test data loaded\n";
                return 1;
            }

            auto results = batch_predict(rnn, vocabulary, test_data, diagnoses);
            std::cout << "Predictions made for " << results.size() << " samples\n";

            // Save results to CSV
            std::string output_file = args.output_file.empty() ? "predictions.csv" : args.output_file;
            save_predictions_csv(output_file, results);
        }

    } else {
        std::cerr << "Error: Unknown mode '" << args.mode << "'. Use 'train' or 'infer'.\n";
        print_help_message();
        return 1;
    }

    return 0;
}
