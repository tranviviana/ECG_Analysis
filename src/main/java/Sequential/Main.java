package Sequential;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
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
            System.out.println("Epoch " + epoch + " - Loss: " + totalLoss / trainData.length);
        }

        // Evaluation on test data would go here
    }
}
