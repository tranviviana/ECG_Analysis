package Sequential;
import java.util.Random;

public class Utils {

    // Generate a random weight matrix
    public static double[][] generateRandomMatrix(int rows, int cols) {
        Random random = new Random();
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextGaussian();
            }
        }
        return matrix;
    }

    // Generate a random bias vector
    public static double[] generateRandomVector(int size) {
        Random random = new Random();
        double[] vector = new double[size];
        for (int i = 0; i < size; i++) {
            vector[i] = random.nextGaussian();
        }
        return vector;
    }

    // Convolution operation
    public static double[][] conv2d(double[][] input, double[][] kernel) {
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int kernelHeight = kernel.length;
        int kernelWidth = kernel[0].length;

        int outputHeight = inputHeight - kernelHeight + 1;
        int outputWidth = inputWidth - kernelWidth + 1;
        double[][] output = new double[outputHeight][outputWidth];

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kernelHeight; ki++) {
                    for (int kj = 0; kj < kernelWidth; kj++) {
                        sum += input[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                output[i][j] = sum;
            }
        }

        return output;
    }

    // ReLU activation
    public static double[][] relu(double[][] input) {
        int height = input.length;
        int width = input[0].length;
        double[][] output = new double[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i][j] = Math.max(0, input[i][j]);
            }
        }

        return output;
    }

    // Max pooling operation
    public static double[][] maxPool(double[][] input, int poolHeight, int poolWidth) {
        int inputHeight = input.length;
        int inputWidth = input[0].length;

        int outputHeight = inputHeight / poolHeight;
        int outputWidth = inputWidth / poolWidth;
        double[][] output = new double[outputHeight][outputWidth];

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int pi = 0; pi < poolHeight; pi++) {
                    for (int pj = 0; pj < poolWidth; pj++) {
                        max = Math.max(max, input[i * poolHeight + pi][j * poolWidth + pj]);
                    }
                }
                output[i][j] = max;
            }
        }

        return output;
    }
    // Flatten 2D array to 1D array
    public static double[] flatten(double[][] input) {
        int height = input.length;
        int width = input[0].length;
        double[] output = new double[height * width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i * width + j] = input[i][j];
            }
        }

        return output;
    }

    // Dense layer
    public static double[] dense(double[] input, double[][] weights, double[] biases) {
        int outputSize = biases.length;
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < input.length; j++) {
                sum += input[j] * weights[i][j];
            }
            output[i] = sum;
        }

        return output;
    }

    // ReLU activation for 1D array
    public static double[] relu(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

// Softmax activation
public static double[] softmax(double[] input) {
    double max = Double.NEGATIVE_INFINITY;
    for (double value : input) {
        if (value > max) {
            max = value;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < input.length; i++) {
        input[i] = Math.exp(input[i] - max);
        sum += input[i];
    }

    for (int i = 0; i < input.length; i++) {
        input[i] /= sum;
    }

    return input;
}

    // Cross entropy loss
    public static double crossEntropyLoss(double[] predictions, int label) {
        return -Math.log(predictions[label]);
    }

// Calculate the gradient of the softmax loss
public static double[] softmaxGradient(double[] predictions, int label) {
    double[] gradient = new double[predictions.length];
    for (int i = 0; i < predictions.length; i++) {
        gradient[i] = predictions[i] - (i == label ? 1 : 0);
    }
    return gradient;
}

    // Update weights using gradient descent
    public static void updateWeights(double[][] weights, double[] biases, double[] gradients, double[] inputs, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            biases[i] -= learningRate * gradients[i];
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] -= learningRate * gradients[i] * inputs[j];
            }
        }
    }
}

