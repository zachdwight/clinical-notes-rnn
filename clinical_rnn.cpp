#include <iostream>
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

            // Tokenize
            std::vector<int> word_ids;
            std::string current_word;
            for (char c : note_text) {
                if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                    if (!current_word.empty()) {
                        std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                       [](unsigned char ch){ return std::tolower(ch); });
                        if (vocab_map.count(current_word)) {
                            word_ids.push_back(vocab_map.at(current_word));
                        } else {
                            word_ids.push_back(unk_token_id);  // Use UNK for unknown words
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
                } else {
                    word_ids.push_back(unk_token_id);
                }
            }

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

    void train(const std::vector<std::pair<std::string, std::string>>& training_data,
               const std::map<std::string, int>& vocab_map,
               const std::vector<std::string>& diagnosis_labels,
               int num_epochs,
               bool use_data_split = true,
               double learning_rate_decay = 0.0) {

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

                std::vector<int> word_ids;
                std::string current_word;
                for (char c : note_text) {
                    if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                        if (!current_word.empty()) {
                            std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                           [](unsigned char ch){ return std::tolower(ch); });
                            if (vocab_map.count(current_word)) {
                                word_ids.push_back(vocab_map.at(current_word));
                            } else {
                                word_ids.push_back(unk_token_id);  // Use UNK for unknown words
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
                    } else {
                        word_ids.push_back(unk_token_id);
                    }
                }

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

            // Print training metrics
            std::cout << "Epoch " << epoch + 1 << " | Train Loss: " << std::fixed << std::setprecision(6)
                      << epoch_loss / total_predictions << " | Train Acc: " << std::fixed << std::setprecision(2)
                      << (static_cast<double>(correct_predictions) / total_predictions) * 100.0 << "%";

            // Evaluate on validation set if available
            if (!val_set.empty()) {
                EvaluationMetrics val_metrics = evaluate(val_set, vocab_map, diagnosis_labels);
                std::cout << " | Val Loss: " << std::fixed << std::setprecision(6) << val_metrics.loss
                          << " | Val Acc: " << std::fixed << std::setprecision(2) << val_metrics.accuracy * 100.0 << "%";
            }
            std::cout << "\n";
        }

        // Final evaluation on test set
        if (!test_set.empty()) {
            std::cout << "\n--- Test Set Evaluation ---\n";
            EvaluationMetrics test_metrics = evaluate(test_set, vocab_map, diagnosis_labels);
            std::cout << "Test Loss: " << std::fixed << std::setprecision(6) << test_metrics.loss << "\n";
            std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_metrics.accuracy * 100.0 << "%\n";
            std::cout << "Per-Class Accuracy:\n";
            for (const auto& label : diagnosis_labels) {
                std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
                          << test_metrics.per_class_accuracy[label] * 100.0 << "%\n";
            }
        }

        std::cout << "--- Training Finished ---\n";
    }
};

int main() {
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

    int vocab_size = vocab.size();
    int hidden_size = 10;
    int output_size = diagnoses.size();
    double learning_rate = 0.01;
    int num_epochs = 100;

    SimpleRNN rnn(vocab_size, hidden_size, output_size, learning_rate);

    std::cout << "RNN Initialized with:\n";
    std::cout << "  Vocab Size: " << rnn.vocab_size << "\n";
    std::cout << "  Hidden Size: " << rnn.hidden_size << "\n";
    std::cout << "  Output Size: " << rnn.output_size << "\n";
    std::cout << "  Learning Rate: " << rnn.learning_rate << "\n";
    std::cout << "  Gradient Clip Norm: " << rnn.gradient_clip_norm << "\n";
    std::cout << "  Epochs: " << num_epochs << "\n";

    // Train with data splitting and learning rate decay
    rnn.train(SIMULATED_CLINICAL_NOTES, vocab, diagnoses, num_epochs, true, 0.05);

    std::cout << "\n--- Final Predictions After Training ---\n";

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

        std::vector<int> word_ids;
        std::string current_word;
        for (char c : note_text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '(' || c == ')') {
                if (!current_word.empty()) {
                    std::transform(current_word.begin(), current_word.end(), current_word.begin(),
                                   [](unsigned char ch){ return std::tolower(ch); });
                    if (vocab.count(current_word)) {
                        word_ids.push_back(vocab.at(current_word));
                    } else {
                        word_ids.push_back(rnn.unk_token_id);  // Use UNK for unknown words
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
            } else {
                word_ids.push_back(rnn.unk_token_id);
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
