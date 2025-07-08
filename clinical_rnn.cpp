#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>    // For tanh, exp
#include <numeric>  // For accumulate (sum)
#include <random>   // For random number generation
#include <algorithm> // For std::max_element, std::transform
#include <cctype>   // For std::tolower
#include <iomanip>  // For std::fixed, std::setprecision

#include "synthetic_training_data.h" // Include the simulated data header


// Cleaned up and organized with help from Google Gemini so be cautious!
// --- Basic Matrix Operations (Simplified, for demonstration only) ---
// In a real application, use a proper linear algebra library like Eigen.

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>; // Represents a row vector or column vector based on context

// Function to print a matrix (for debugging)
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

// Element-wise matrix subtraction (A - B)
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

// Matrix multiplication (A * B)
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

// Matrix addition (A + B)
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

// Element-wise multiplication (Hadamard product)
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

// Scale matrix by a scalar
Matrix scale(const Matrix& mat, double scalar) {
    Matrix result = mat;
    for (auto& row : result) {
        for (double& val : row) {
            val *= scalar;
        }
    }
    return result;
}

// Transpose a matrix
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

// Apply a function element-wise to a matrix
Matrix apply_function(const Matrix& mat, double (*func)(double)) {
    Matrix result = mat;
    for (auto& row : result) {
        for (double& val : row) {
            val = func(val);
        }
    }
    return result;
}

// --- Activation Functions and their Derivatives ---

double tanh_activation(double x) {
    return std::tanh(x);
}

// Derivative of tanh(x) = 1 - tanh(x)^2
double tanh_derivative(double x) {
    return 1.0 - std::pow(std::tanh(x), 2);
}
// For backprop, we'll often use the output of the tanh function (h_t)
// So, derivative of tanh(h) w.r.t. h is (1 - h^2)
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

// Simplified derivative of Softmax (Jacobian for a single output)
// This is typically combined with Cross-Entropy Loss, where dL/dx = y_pred - y_true
// For MSE with Softmax, the derivative is more complex. We'll use a simplified approach
// for demonstration, assuming the error dY is directly propagated.
// For true Softmax derivative, it's (diag(y) - y * y^T). This is too complex for this example.
// We will just propagate the error (dY) back through the last linear layer.

// --- Loss Function ---
// Mean Squared Error (MSE)
double mean_squared_error(const Vector& predictions, const Vector& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Prediction and target vectors must have the same size for MSE.");
    }
    double error_sum_sq = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        error_sum_sq += std::pow(predictions[i] - targets[i], 2);
    }
    return error_sum_sq / predictions.size();
}

// Derivative of MSE: d(MSE)/d(pred_i) = 2 * (pred_i - target_i) / N
Vector mean_squared_error_derivative(const Vector& predictions, const Vector& targets) {
    Vector grad(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
    }
    return grad; // This is dL/dY_pred (d_output_unactivated)
}

// --- SimpleRNN Class ---

class SimpleRNN {
public:
    int vocab_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    // Weights and Biases
    Matrix W_hh; // Hidden to hidden
    Matrix W_xh; // Input to hidden
    Matrix W_hy; // Hidden to output

    Matrix b_h;  // Hidden bias
    Matrix b_y;  // Output bias

    // For caching during forward pass to use in backward pass
    // This is crucial for backpropagation
    std::vector<Matrix> hidden_states_history; // Stores h_t for each time step
    std::vector<Matrix> pre_tanh_activations; // Stores inputs to tanh for each time step (h_t_unactivated)

    // Constructor
    SimpleRNN(int vocab_s, int hidden_s, int output_s, double lr)
        : vocab_size(vocab_s), hidden_size(hidden_s), output_size(output_s), learning_rate(lr) {

        // Use a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        // Glorot/Xavier initialization for tanh: sqrt(6 / (fan_in + fan_out))
        // Simplified to normal distribution with small variance
        std::normal_distribution<> d_tanh(0.0, std::sqrt(2.0 / (hidden_size + vocab_size))); // W_xh, W_hh
        std::normal_distribution<> d_linear(0.0, std::sqrt(2.0 / (hidden_size + output_size))); // W_hy

        // Initialize weights and biases
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
            return Matrix(rows, Vector(cols, 0.0)); // Biases often initialized to 0
        };


        W_hh = initialize_matrix_tanh(hidden_size, hidden_size);
        W_xh = initialize_matrix_tanh(vocab_size, hidden_size);
        W_hy = initialize_matrix_linear(hidden_size, output_size);

        b_h = initialize_matrix_bias(1, hidden_size);
        b_y = initialize_matrix_bias(1, output_size);
    }

    // Simplified one-hot encoding for a word ID
    Matrix one_hot_encode(int word_id) const {
        Matrix vector(1, Vector(vocab_size, 0.0));
        if (word_id >= 0 && word_id < vocab_size) {
            vector[0][word_id] = 1.0;
        }
        return vector;
    }

    // Forward pass
    Vector forward(const std::vector<int>& word_ids_sequence) {
        // Clear history for new sequence
        hidden_states_history.clear();
        pre_tanh_activations.clear();

        // Initial hidden state (h_0)
        Matrix h_t(1, Vector(hidden_size, 0.0));
        hidden_states_history.push_back(h_t); // Store h_0

        for (int word_id : word_ids_sequence) {
            Matrix x_t = one_hot_encode(word_id); // Current input (1 x vocab_size)

            Matrix h_prev_whh = multiply(hidden_states_history.back(), W_hh); // h_{t-1} @ W_hh
            Matrix x_t_wxh = multiply(x_t, W_xh);                             // x_t @ W_xh

            Matrix sum_inputs = add(h_prev_whh, x_t_wxh);
            Matrix h_t_unactivated = add(sum_inputs, b_h); // Before tanh

            pre_tanh_activations.push_back(h_t_unactivated); // Store for backprop

            h_t = apply_function(h_t_unactivated, tanh_activation); // h_t = tanh(...)
            hidden_states_history.push_back(h_t); // Store h_t
        }

        // Output Layer calculation
        Matrix final_h_t = hidden_states_history.back(); // h_T
        Matrix output_unactivated_matrix = add(
            multiply(final_h_t, W_hy),
            b_y
        );

        return softmax(output_unactivated_matrix[0]); // Return probabilities
    }

    // Backward pass (simplified for demonstration)
    void backward(const std::vector<int>& word_ids_sequence, const Vector& predictions, const Vector& targets) {
        // dL/dY_pred (d_output) - gradient of loss with respect to predictions
        Vector d_output_pred = mean_squared_error_derivative(predictions, targets);
        Matrix d_output_mat(1, d_output_pred); // Convert to Matrix for operations

        // 1. Gradients for Output Layer (W_hy, b_y)
        // dL/d(W_hy) = dL/dY_pred * dY_pred/d(output_unactivated) * d(output_unactivated)/d(W_hy)
        // For linear layer Y = H_T @ W_hy + b_y, dY/dW_hy = H_T^T
        // For Softmax with MSE, the exact derivative is complex. We'll approximate using
        // dL/d(output_unactivated) = d_output_mat. This is a common simplification for demonstration
        // in combination with MSE, but less accurate than Cross-Entropy with Softmax.

        Matrix final_h_t = hidden_states_history.back(); // H_T
        Matrix d_W_hy = multiply(transpose(final_h_t), d_output_mat); // (H_T)^T @ dL/dY_pred
        Matrix d_b_y = d_output_mat; // dL/dY_pred, summed over batch (here, batch size 1)

        // 2. Gradients propagating back to the final hidden state (d_h_T)
        // dL/d(h_T) = dL/dY_pred * dY_pred/d(h_T) = d_output_mat @ (W_hy)^T
        Matrix d_h_t_prop = multiply(d_output_mat, transpose(W_hy)); // Propagate error back through W_hy

        // 3. Gradients for Recurrent Layer (Simplified BPTT - only considers final h_t for W_hh, W_xh, b_h)
        // A true BPTT would sum gradients across all time steps. This is a *major* simplification.

        // Apply derivative of tanh to d_h_t_prop
        // dL/d(h_t_unactivated) = dL/d(h_t) * d(h_t)/d(h_t_unactivated) = d_h_t_prop * (1 - h_t^2)
        Matrix d_h_t_unactivated = hadamard_product(d_h_t_prop, tanh_derivative_from_output(final_h_t));

        // Gradients for W_hh, W_xh, b_h based on final hidden state's error
        // This is where a full BPTT would loop backwards through all time steps

        // For the *last* time step only:
        Matrix h_prev_last_step = hidden_states_history[hidden_states_history.size() - 2]; // h_{T-1}
        Matrix x_T = one_hot_encode(word_ids_sequence.back()); // x_T

        Matrix d_W_hh = multiply(transpose(h_prev_last_step), d_h_t_unactivated); // (h_{T-1})^T @ dL/d(h_T_unactivated)
        Matrix d_W_xh = multiply(transpose(x_T), d_h_t_unactivated);             // (x_T)^T @ dL/d(h_T_unactivated)
        Matrix d_b_h = d_h_t_unactivated;                                        // dL/d(h_T_unactivated)

        // --- Update Weights and Biases ---
        // Weights = Weights - learning_rate * Gradients

        W_hy = subtract(W_hy, scale(d_W_hy, learning_rate));
        b_y = subtract(b_y, scale(d_b_y, learning_rate));

        W_hh = subtract(W_hh, scale(d_W_hh, learning_rate));
        W_xh = subtract(W_xh, scale(d_W_xh, learning_rate));
        b_h = subtract(b_h, scale(d_b_h, learning_rate));
    }

    // Training loop
    void train(const std::vector<std::pair<std::string, std::string>>& training_data,
               const std::map<std::string, int>& vocab_map,
               const std::vector<std::string>& diagnosis_labels,
               int num_epochs) {

        std::map<std::string, Vector> target_vectors;
        // Pre-create one-hot target vectors for diagnoses
        for (size_t i = 0; i < diagnosis_labels.size(); ++i) {
            Vector target(diagnosis_labels.size(), 0.0);
            target[i] = 1.0;
            target_vectors[diagnosis_labels[i]] = target;
        }

        std::cout << "\n--- Starting Training ---\n";

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            double epoch_loss = 0.0;
            int correct_predictions = 0;
            int total_predictions = 0;

            for (const auto& note_pair : training_data) {
                const std::string& note_text = note_pair.first;
                const std::string& true_diagnosis_label = note_pair.second;

                // Tokenization (same as in main)
                std::vector<int> word_ids;
                std::string current_word;
                for (char c : note_text) {
                    if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                        if (!current_word.empty()) {
                            std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                           [](unsigned char ch){ return std::tolower(ch); });
                            if (vocab_map.count(current_word)) {
                                word_ids.push_back(vocab_map.at(current_word));
                            }
                            current_word.clear();
                        }
                    } else {
                        current_word += c;
                    }
                }
                if (!current_word.empty()) {
                    std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                   [](unsigned char ch){ return std::tolower(ch); });
                    if (vocab_map.count(current_word)) {
                        word_ids.push_back(vocab_map.at(current_word));
                    }
                }

                if (word_ids.empty()) {
                    // std::cout << "  [Epoch " << epoch << "] Skipping empty note.\n";
                    continue; // Skip notes with no recognized words
                }

                // Forward pass
                Vector predictions = forward(word_ids);

                // Get true target vector
                const Vector& targets = target_vectors.at(true_diagnosis_label);

                // Calculate loss
                double current_loss = mean_squared_error(predictions, targets);
                epoch_loss += current_loss;

                // Backpropagate and update weights
                backward(word_ids, predictions, targets);

                // Check accuracy
                auto max_it = std::max_element(predictions.begin(), predictions.end());
                int predicted_index = std::distance(predictions.begin(), max_it);
                int true_index = std::distance(diagnosis_labels.begin(), std::find(diagnosis_labels.begin(), diagnosis_labels.end(), true_diagnosis_label));

                if (predicted_index == true_index) {
                    correct_predictions++;
                }
                total_predictions++;
            }
            std::cout << "Epoch " << epoch + 1 << " completed. Average Loss: " << std::fixed << std::setprecision(6) << epoch_loss / total_predictions
                      << ", Accuracy: " << std::fixed << std::setprecision(2) << (static_cast<double>(correct_predictions) / total_predictions) * 100.0 << "%\n";
        }
        std::cout << "--- Training Finished ---\n";
    }
};

// --- Example Usage ---

int main() {
    // 1. Define a simple vocabulary and mapping
    std::map<std::string, int> vocab = {
        {"fever", 0}, {"cough", 1}, {"headache", 2}, {"sore", 3}, {"throat", 4},
        {"fatigue", 5}, {"pneumonia", 6}, {"shortness", 7}, {"breath", 8},
        {"muscle", 9}, {"aches", 10}, {"diagnosis", 11}, {"patient", 12},
        {"has", 13}, {"symptoms", 14}, {"mild", 15}, {"severe", 16}, {"no", 17},
        {"runny", 18}, {"nose", 19}, {"started", 20}, {"two", 21}, {"days", 22},
        {"ago", 23}, {"complains", 24}, {"stuffy", 25}, {"sneezing", 26}, {"occasional", 27},
        {"dry", 28}, {"feeling", 29}, {"slightly", 30}, {"tired", 31}, {"congestion", 32},
        {"low", 33}, {"energy", 34}, {"reports", 35}, {"body", 36}, {"eating", 37},
        {"well", 38}, {"child", 39}, {"clear", 40}, {"nasal", 41}, {"discharge", 42},
        {"seems", 43}, {"otherwise", 44}, {"active", 45}, {"post-nasal", 46}, {"drip", 47},
        {"scratchy", 48}, {"denies", 49}, {"chills", 50}, {"watery", 51}, {"eyes", 52},
        {"bit", 53}, {"under", 54}, {"the", 55}, {"weather", 56}, {"persistent", 57},
        {"episodes", 58}, {"passages", 59}, {"blocked", 60}, {"significant", 61}, {"pain", 62},
        {"run", 63}, {"down", 64}, {"worsens", 65}, {"when", 66}, {"swallowing", 67},
        {"minimal", 68}, {"minor", 69}, {"producing", 70}, {"phlegm", 71}, {"normal", 72},
        {"temperature", 73}, {"just", 74}, {"general", 75}, {"unwell", 76}, {"sniffle", 77},
        {"tickle", 78}, {"detected", 79}, {"itchy", 80}, {"malaise", 81}, {"head", 82},
        {"sinus", 83}, {"pressure", 84}, {"able", 85}, {"perform", 86}, {"daily", 87},
        {"activities", 88}, {"aches", 89}, {"are", 90}, {"primary", 91}, {"complaint", 92},
        {"blockage", 93}, {"productive", 94}, {"irritation", 95}, {"drinking", 96}, {"fine", 97},
        {"developed", 98}, {"last", 99}, {"night", 100}, {"common", 101}, {"morning", 102},
        {"frequent", 103}, {"mostly", 104}, {"congested", 105}, {"other", 106}, {"exhausted", 107},
        {"abrupt", 108}, {"onset", 109}, {"sudden", 110}, {"high", 111}, {"severe", 112},
        {"generalized", 113}, {"intermittent", 114}, {"denies", 115}, {"nasal", 116}, {"congestion", 117},
        {"chills", 118}, {"sweating", 119}, {"can", 120}, {"barely", 121}, {"get", 122},
        {"out", 123}, {"bed", 124}, {"profound", 125}, {"weakness", 126}, {"throbbing", 127},
        {"hacking", 128}, {"myalgia", 129}, {"arthralgia", 130}, {"sputum", 131}, {"feels", 132},
        {"like", 133}, {"they've", 134}, {"been", 135}, {"hit", 136}, {"by", 137},
        {"truck", 138}, {"extreme", 139}, {"tiredness", 140}, {"soreness", 141}, {"throughout", 142},
        {"body", 143}, {"very", 144}, {"weak", 145}, {"has", 146}, {"flu-like", 147},
        {"symptoms", 148}, {"classic", 149}, {"illness", 150}, {"joint", 151}, {"pain", 152},
        {"overwhelming", 153}, {"temperature", 154}, {"unable", 155}, {"go", 156}, {"to", 157},
        {"work", 158}, {"acute", 159}, {"drained", 160}, {"acute", 161}, {"onset", 162},
        {"worsening", 163}, {"wiped", 164}, {"completely", 165}, {"difficulty", 166}, {"sleeping", 167},
        {"due", 168}, {"discomfort", 169}, {"eat", 170}, {"much", 171}, {"respiratory", 172},
        {"issues", 173}, {"noted", 174}, {"dyspnea", 175}, {"green", 176}, {"chest", 177},
        {"with", 178}, {"breathing", 179}, {"decreased", 180}, {"oxygen", 181}, {"saturation", 182},
        {"wet", 183}, {"yellow", 184}, {"auscultation", 185}, {"reveals", 186}, {"crackles", 187},
        {"tightness", 188}, {"labored", 189}, {"rust-colored", 190}, {"sounds", 191}, {"lung", 192},
        {"base", 193}, {"catching", 194}, {"elevated", 195}, {"rate", 196}, {"history", 197},
        {"lung", 198}, {"issues", 199}, {"feels", 200}, {"winded", 201}, {"after", 202},
        {"minimal", 203}, {"exertion", 204}, {"rhonchi", 205}, {"purulent", 206}, {"weakness", 207},
        {"heard", 208}, {"lower", 209}, {"lobes", 210}, {"coughing", 211}, {"mucus", 212},
        {"thick", 213}, {"looks", 214}, {"appears", 215}, {"distress", 216}, {"x-ray", 217},
        {"ordered", 218}, {"blood-tinged", 219}, {"copious", 220}, {"rapid", 221}, {"shallow", 222},
        {"discomfort", 223}, {"reduced", 224}, {"side", 225}, {"discharge", 226}, {"gasping", 227},
        {"air", 228}, {"admitted", 229}, {"lie", 230}, {"flat", 231}, {"comfortably", 232}
    };


    std::vector<std::string> diagnoses = {"Cold", "Flu", "Pneumonia"};

    // 2. Network Parameters
    int vocab_size = vocab.size();
    int hidden_size = 10; // Size of the hidden state vector
    int output_size = diagnoses.size(); // Number of possible diagnoses
    double learning_rate = 0.01; // Learning rate for gradient descent
    int num_epochs = 100; // Number of training iterations over the dataset

    // 3. Instantiate the RNN
    SimpleRNN rnn(vocab_size, hidden_size, output_size, learning_rate);

    std::cout << "RNN Initialized with:\n";
    std::cout << "  Vocab Size: " << rnn.vocab_size << "\n";
    std::cout << "  Hidden Size: " << rnn.hidden_size << "\n";
    std::cout << "  Output Size: " << rnn.output_size << "\n";
    std::cout << "  Learning Rate: " << rnn.learning_rate << "\n";
    std::cout << "  Epochs: " << num_epochs << "\n";


    // 4. Train the RNN
    rnn.train(SIMULATED_CLINICAL_NOTES, vocab, diagnoses, num_epochs);


    std::cout << "\n--- Final Predictions After Training ---\n";

    // 5. Test the trained RNN with a few examples
    // We'll reuse the first few notes to see if it learned anything
    std::vector<std::pair<std::string, std::string>> test_notes = {
        {"Patient presents with mild cough, runny nose, and sore throat.", "Cold"},
        {"Complains of body aches, chills, and a fever of 102F. Feels exhausted.", "Flu"},
        {"Patient presents with productive cough (green sputum), shortness of breath, and fever.", "Pneumonia"},
        {"No fever, but has a slight sore throat and runny nose.", "Cold"},
        {"Severe fatigue, headache, and generalized muscle pain. Fever.", "Flu"},
        {"Fever 102F, severe cough producing thick phlegm, and shortness of breath.", "Pneumonia"}
    };


    int test_count = 0;
    for (const auto& note_pair : test_notes) {
        const std::string& note_text = note_pair.first;
        const std::string& true_diagnosis = note_pair.second;

        test_count++;
        std::cout << "\nTesting Note " << test_count << " (True: " << true_diagnosis << "): \"" << note_text << "\"\n";

        // Tokenization (same as training)
        std::vector<int> word_ids;
        std::string current_word;
        for (char c : note_text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                if (!current_word.empty()) {
                    std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                   [](unsigned char ch){ return std::tolower(ch); });
                    if (vocab.count(current_word)) {
                        word_ids.push_back(vocab.at(current_word));
                    }
                    current_word.clear();
                }
            } else {
                current_word += c;
            }
        }
        if (!current_word.empty()) {
            std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                           [](unsigned char ch){ return std::tolower(ch); });
            if (vocab.count(current_word)) {
                word_ids.push_back(vocab.at(current_word));
            }
        }

        if (word_ids.empty()) {
            std::cout << "  No recognized words in this note. Skipping.\n";
            continue;
        }

        Vector predictions = rnn.forward(word_ids);

        std::cout << "  Predicted probabilities:\n";
        for (size_t i = 0; i < diagnoses.size(); ++i) {
            std::cout << "    " << diagnoses[i] << ": " << std::fixed << std::setprecision(2) << (predictions[i] * 100.0) << "%\n";
        }

        auto max_it = std::max_element(predictions.begin(), predictions.end());
        int predicted_index = std::distance(predictions.begin(), max_it);
        std::string predicted_diagnosis = diagnoses[predicted_index];
        std::cout << "  Most probable diagnosis: " << predicted_diagnosis << " (True: " << true_diagnosis << ")\n";
    }


    return 0;
}
