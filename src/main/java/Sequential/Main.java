package Sequential;

import java.io.FileWriter;
import java.util.Random;



class Utils {
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

    public static int argmax(double[] input) {
        int maxIndex = 0;
        for (int i = 1; i < input.length; i++) {
            if (input[i] > input[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
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

class CNN {

    private double[][] conv1Kernel;
    private double[][] conv2Kernel;
    private double[][] denseWeights;
    private double[] denseBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    public CNN() {
        conv1Kernel = Utils.generateRandomMatrix(5, 1); // 5x1 convolution kernel
        conv2Kernel = Utils.generateRandomMatrix(5, 1); // 5x1 convolution kernel
        denseWeights = Utils.generateRandomMatrix(128, 6400); // Adjust size as needed
        denseBiases = Utils.generateRandomVector(128);
        outputWeights = Utils.generateRandomMatrix(10, 128); // 10 output classes
        outputBiases = Utils.generateRandomVector(10);
    }

    public double[] forward(double[][] input) {
        double[][] conv1 = Utils.conv2d(input, conv1Kernel);
        double[][] relu1 = Utils.relu(conv1);
        double[][] pool1 = Utils.maxPool(relu1, 2, 1);

        double[][] conv2 = Utils.conv2d(pool1, conv2Kernel);
        double[][] relu2 = Utils.relu(conv2);
        double[][] pool2 = Utils.maxPool(relu2, 2, 1);

        double[] flat = Utils.flatten(pool2);
        double[] dense = Utils.dense(flat, denseWeights, denseBiases);
        double[] relu3 = Utils.relu(dense);
        double[] output = Utils.dense(relu3, outputWeights, outputBiases);

        return Utils.softmax(output);
    }

    public void backward(double[][] input, int label, double learningRate) {
        // Forward pass
        double[][] conv1 = Utils.conv2d(input, conv1Kernel);
        double[][] relu1 = Utils.relu(conv1);
        double[][] pool1 = Utils.maxPool(relu1, 2, 1);

        double[][] conv2 = Utils.conv2d(pool1, conv2Kernel);
        double[][] relu2 = Utils.relu(conv2);
        double[][] pool2 = Utils.maxPool(relu2, 2, 1);

        double[] flat = Utils.flatten(pool2);
        double[] dense = Utils.dense(flat, denseWeights, denseBiases);
        double[] relu3 = Utils.relu(dense);
        double[] output = Utils.dense(relu3, outputWeights, outputBiases);
        double[] predictions = Utils.softmax(output);

        // Backward pass (Gradient calculation)
        double[] outputGradients = Utils.softmaxGradient(predictions, label);
        double[] relu3Gradients = new double[relu3.length];
        double[] denseGradients = new double[denseWeights.length];
        double[] flatGradients = new double[flat.length];

        // Gradients for the dense layer
        for (int i = 0; i < denseGradients.length; i++) {
            for (int j = 0; j < outputGradients.length; j++) {
                denseGradients[i] += outputWeights[j][i] * outputGradients[j];
            }
            relu3Gradients[i] = (relu3[i] > 0) ? denseGradients[i] : 0;
        }

        // Gradients for the convolutional and pooling layers
        for (int i = 0; i < flat.length; i++) {
            for (int j = 0; j < denseGradients.length; j++) {
                flatGradients[i] += denseWeights[j][i] * relu3Gradients[j];
            }
        }

        // Update weights and biases
        Utils.updateWeights(outputWeights, outputBiases, outputGradients, relu3, learningRate);
        //Utils.updateWeights(denseWeights, denseBiases, relu3Gradients, flat, learningRate);

        // (Backward propagation through pooling and convolution layers would need to update conv1Kernel and conv2Kernel)
    }

    public int classify(double[][] input) throws Exception {
        double[] output = forward(input);
        return Utils.argmax(output);
    }
}


public class Main {

    public static void main(String[] args) throws Exception {
        FileWriter sequentialEpochFile = new FileWriter("./src/main/java/Sequential/SequentialEpoch.txt");

        for(int v = 0; v < 12; v++) {
            CNN cnn = new CNN();
            // Dummy data for training
            double[][][] trainData = new double[1000][28][28]; // 1000 samples of 28x28 ECG data
            int[] trainLabels = new int[1000]; // Corresponding labels
            Random random = new Random();




            for (int i = 0; i < trainData.length; i++) {
                for (int j = 0; j < 28; j++) {
                    for (int k = 0; k < 28; k++) {
                        trainData[i][j][k] = random.nextDouble();
                    }
                }
                trainLabels[i] = random.nextInt(10); // Random label from 0 to 9
            }

            // Training loop
            int epochs = 10;
            double learningRate = 0.01;

            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalLoss = 0.0;
                for (int i = 0; i < trainData.length; i++) {
                    double[] predictions = cnn.forward(trainData[i]);
                    totalLoss += Utils.crossEntropyLoss(predictions, trainLabels[i]);
                    cnn.backward(trainData[i], trainLabels[i], learningRate);
                }
                //System.out.println("Epoch " + epoch + " - Loss: " + totalLoss / trainData.length);
                // 10 epochs per trial
                sequentialEpochFile.write(String.valueOf(totalLoss / trainData.length));
                sequentialEpochFile.write(" ");


            }

            // Evaluation on test data would go here
            // Generate random test ECG data
            double[][][] testData = new double[100][28][28]; // 100 samples of 28x28 ECG data
            random = new Random();

            for (int i = 0; i < testData.length; i++) {
                for (int j = 0; j < 28; j++) {
                    for (int k = 0; k < 28; k++) {
                        testData[i][j][k] = random.nextDouble();
                    }
                }
            }

            // Classify test data
            for (int i = 0; i < testData.length; i++) {
                int predictedLabel = cnn.classify(testData[i]);
                //System.out.println("Sample " + i + " - Predicted Label: " + predictedLabel);
            }
            //another test generated occurs here
            sequentialEpochFile.write("\r\n");
        }


        sequentialEpochFile.close();

    }
}