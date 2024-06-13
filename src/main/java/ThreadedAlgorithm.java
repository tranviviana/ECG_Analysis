import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.evaluation.classification.Evaluation;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

//benchmarking library
import java.util.concurrent.TimeUnit;

public class ThreadedAlgorithm {
    public void run() {
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                // Your parallel algorithm implementation
                public class HardwareThreadedCNNForECGAnalysis {

                    public static void main(String[] args) throws Exception {
                        int numLinesToSkip = 0;
                        char delimiter = ',';
                        int batchSize = 64;
                        int labelIndex = 187;  // Assuming the label is the last column (for example, if you have 188 columns)
                        int numClasses = 5;    // Number of classes in the dataset
                        int numEpochs = 10;
                        int numThreads = Runtime.getRuntime().availableProcessors();  // Use available processors

                        RecordReader rrTrain = new CSVRecordReader(numLinesToSkip, delimiter);
                        rrTrain.initialize(new FileSplit(new File("path/to/train/ecg_data.csv")));

                        RecordReader rrTest = new CSVRecordReader(numLinesToSkip, delimiter);
                        rrTest.initialize(new FileSplit(new File("path/to/test/ecg_data.csv")));

                        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, numClasses);
                        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, labelIndex, numClasses);

                        DataNormalization normalizer = new NormalizerStandardize();
                        normalizer.fit(trainIter);              // Collect statistics from training data
                        trainIter.setPreProcessor(normalizer);
                        testIter.setPreProcessor(normalizer);
                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .updater(new Adam(0.001))
                        .list()
                        .layer(new ConvolutionLayer.Builder(5, 1)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 1)
                        .stride(2, 1)
                        .build())
                        .layer(new ConvolutionLayer.Builder(5, 1)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 1)
                        .stride(2, 1)
                        .build())
                        .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(128).build())
                       .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                        .setInputType(InputType.convolutionalFlat(1, 188, 1))  // Assuming each ECG record is a 1x188 signal
                        .build();

                        MultiLayerNetwork model = new MultiLayerNetwork(conf);
                        model.init();
                        model.setListeners(new ScoreIterationListener(10));

                        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
                        for (int i = 0; i < numEpochs; i++) {
                            executor.submit(() -> {
                            model.fit(trainIter);
                            });
                        }
        
                        executor.shutdown();
                        while (!executor.isTerminated()) {
                            // Wait until all threads are finished
                        }

                        Evaluation eval = model.evaluate(testIter);
                        System.out.println(eval.stats());
                    }
                }

            });
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        ThreadedAlgorithm algorithm = new ThreadedAlgorithm();
        algorithm.run();
        long endTime = System.nanoTime();
        long duration = (endTime - startTime) / 1_000_000;  // Convert to milliseconds
        System.out.println("Threaded Execution Time: " + duration + " ms");
    }
}






