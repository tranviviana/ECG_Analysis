package Sequential;
public class CNN {

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
        Utils.updateWeights(denseWeights, denseBiases, relu3Gradients, flat, learningRate);

        // (Backward propagation through pooling and convolution layers would need to update conv1Kernel and conv2Kernel)
    }
}

