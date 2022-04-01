package training;


import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class FirstModel {
    protected static final Logger log = LoggerFactory.getLogger(FirstModel.class);
    public static WordVectors wordVectors;
    public static String DATA_PATH = new File("C:/Users/Xameer/Desktop/SKYMIND/Development/TwitterSentiment/src/main/resources","Twitter").getAbsolutePath();

    public static void main(String[] args) throws Exception {
        int batchSize = 100;
        int nEpoch = 10;
        int truncateTweetsToLength = 90; //Truncate tweets with length (# words) greater than this
        File file = new ClassPathResource("wordVector/twitter100WithName.txt").getFile();

        wordVectors = WordVectorSerializer.readWord2VecModel(file);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        TweetsIterator iTrain = new TweetsIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .truncateLength(truncateTweetsToLength)
                .tokenizerFactory(tokenizerFactory)
                .train(true)
                .build();

        TweetsIterator iTest = new TweetsIterator.Builder()
                .dataDirectory(DATA_PATH)
                .wordVectors(wordVectors)
                .batchSize(batchSize)
                .tokenizerFactory(tokenizerFactory)
                .truncateLength(truncateTweetsToLength)
                .train(false)
                .build();

//        DataSetIterator train = new AsyncDataSetIterator(iTrain,1);
//        DataSetIterator test = new AsyncDataSetIterator(iTest,1);

        int input = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        int output = iTrain.getLabels().size();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(0.0018))
//                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                .layer( new LSTM.Builder().nIn(input).nOut(200)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(output).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        System.out.println("Starting training...");
        net.setListeners(new ScoreIterationListener(1),
                new EvaluativeListener(iTest, 1, InvocationType.EPOCH_END),
                new StatsListener(storage,10));
        net.fit(iTrain, nEpoch);

        System.out.println("Evaluating...");
        Evaluation eval = net.evaluate(iTest);
        System.out.println(eval.stats());

        net.save(new File("C:/Users/Xameer/Desktop/SKYMIND/Development/TwitterSentiment/src/main/resources", "TwitterModel.net"), true);
        System.out.println("Model Saved...");
    }

}
