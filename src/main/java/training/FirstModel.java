package training;


import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class FirstModel {
    protected static final Logger log = LoggerFactory.getLogger(FirstModel.class);
    public static WordVectors wordVectors;
    static int batchSize =50;
    static int nEpoch = 1;
    static int truncateTweetsToLength = 90; //Truncate tweets with length (# words) greater than this

    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("wordVector/twitter100WithName.txt").getFile();

        wordVectors = WordVectorSerializer.readWord2VecModel(file);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        TweetsIterator iTrain = new TweetsIterator.Builder()
                .dataDirectory()
    }
}
