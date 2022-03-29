package training;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class FirstModel {
    protected static final Logger log = LoggerFactory.getLogger(FirstModel.class);

    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("strong-negatives.txt").getFile();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String s) {
                return s.toLowerCase();
            }
        });
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .layerSize(300)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Save vectors....");
        WordVectorSerializer.writeWord2VecModel(vec, "./src/main/resources/savedModel/malayTwitterNegatives.txt");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File("./src/main/resources/savedModel/malayTwitterNegatives.txt"));

//        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("pathToSaveModel.txt");
//        WeightLookupTable weightLookupTable = word2Vec.lookupTable();
//        Iterator<INDArray> vectors = weightLookupTable.vectors();
//        INDArray wordVectorMatrix = word2Vec.getWordVectorMatrix("man");
//        double[] wordVector = word2Vec.getWordVector("man");

//        log.info("Closest Words:");
//        Collection<String> kingList = vec.wordsNearest(Arrays.asList("king", "woman"), Arrays.asList("queen"), 10);
//        Collection<String> lst = vec.wordsNearest("sabah", 10);
//        System.out.println(lst);
//        UIServer server = UIServer.getInstance();
//        System.out.println("Started on port " + server.getPort());

//        double cosSim = vec.similarity("day","night");
//        System.out.println(cosSim);


    }
}
