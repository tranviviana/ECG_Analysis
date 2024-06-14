package Threaded;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;

public class CNN {

    private double[][] conv1Kernel;
    private double[][] conv2Kernel;
    private double[][] denseWeights;
    private double[] denseBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    private ExecutorService executor;

    public CNN(int numThreads) {
        conv1Kernel = Utils.generateRandomMatrix(5, 5); // 5x5 convolution kernel
        conv2Kernel = Utils.generateRandomMatrix(5, 5); // 5x5 convolution kernel
        denseWeights = Utils.generateRandomMatrix(128, 320); // Adjust size as needed
        denseBiases = Utils.generateRandomVector(128);
        outputWeights = Utils.generateRandomMatrix(10, 128); // 10 output classes
        outputBiases = Utils.generateRandomVector(10);

        executor = Executors.newFixedThreadPool(numThreads);
    }

    public double[] forward(double[][] input) throws Exception {
        Future<double[][]> conv1Future = executor.submit(() -> Utils.conv2d(input, conv1Kernel));
        double[][] conv1 = conv1Future.get();
        double[][] relu1 = Utils.relu(conv1);
        double[][] pool1 = Utils.maxPool(relu1, 2, 2);

        Future<double[][]> conv2Future = executor.submit(() -> Utils.conv2d(pool1, conv2Kernel));
        double[][] conv2 = conv2Future.get();
        double[][] relu2 = Utils.relu(conv2);
        double[][] pool2 = Utils.maxPool(relu2, 2, 2);

        double[] flat = Utils.flatten(pool2);
        double[] dense = Utils.dense(flat, denseWeights, denseBiases);
        double[] relu3 = Utils.relu(dense);
        double[] output = Utils.dense(relu3, outputWeights, outputBiases);

        return Utils.softmax(output);
    }

    public void backward(double[][] input, int label, double learningRate) throws Exception {
        // Forward pass
        Future<double[][]> conv1Future = executor.submit(() -> Utils.conv2d(input, conv1Kernel));
        double[][] conv1 = conv1Future.get();
        double[][] relu1 = Utils.relu(conv1);
        double[][] pool1 = Utils.maxPool(relu1, 2, 2);

        Future<double[][]> conv2Future = executor.submit(() -> Utils.conv2d(pool1, conv2Kernel));
        double[][] conv2 = conv2Future.get();
        double[][] relu2 = Utils.relu(conv2);
        double[][] pool2 = Utils.maxPool(relu2, 2, 2);

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
        Utils.updateWeights(denseWeights, denseBiases, relu3Gradients, flat, learningRate);

        // (Backward propagation through pooling and convolution layers would need to update conv1Kernel and conv2Kernel)
    }

    public void shutdown() {
        executor.shutdown();
    }
}

