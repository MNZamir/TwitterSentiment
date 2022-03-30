package training;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

public class PrepareWordVec {

    protected static final Logger log = LoggerFactory.getLogger(PrepareWordVec.class);

    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("twitterFullCleanDataWithName.txt").getFile();
        //data need to be clean using regex -
        // link - \w+:\/\/\S+
        // @username - @[A-Za-z0-9]+
        // remove '@' in username (but still have value @ in data. Why?) - @[^0-9a-zA-Z]+
        //remove all username start with underscore - (\s|^)@\w+
        // #hashtag - #[A-Za-z0-9_]+

        //data clean using string search
        // &amp, &lt, &gt

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new LineSentenceIterator(file);
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
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        // Write word vectors to a file
        WordVectorSerializer.writeWord2VecModel(vec,"./src/main/resources/wordVector/twitter100WithName.txt");

        log.info("Closest Words: ");
        Collection<String> list = vec.wordsNearest("hari", 10);
        double cosSim = vec.similarity("siang","malam");
        System.out.println(list);
        System.out.println("Persamaan siang dan malam : " + cosSim);

//        log.info("Save vectors....");
//        WordVectorSerializer.writeWord2VecModel(vec, "./src/main/resources/savedModel/malayTwitterNegatives.txt");
//        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File("./src/main/resources/savedModel/malayTwitterNegatives.txt"));

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
