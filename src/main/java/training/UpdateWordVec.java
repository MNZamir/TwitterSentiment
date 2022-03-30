package training;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

public class UpdateWordVec {
    protected static final Logger log = LoggerFactory.getLogger(UpdateWordVec.class);

    public static void main(String[] args) throws Exception {
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("./src/main/resources/wordVector/twitter100WithName.txt");
//        File file = new ClassPathResource("").getFile(); //add new txt data file to update model
//        SentenceIterator iterator = new LineSentenceIterator(file);
//        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
//        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

//        word2Vec.setTokenizerFactory(tokenizerFactory);
//        word2Vec.setSentenceIterator(iterator);

        log.info("Uptrain Word2Vec...");
//        word2Vec.fit();

        Collection<String> list = word2Vec.wordsNearestSum("makan",5);
        log.info("Closest words to 'day' on 2nd run: " + list);

        //Save word vector
//        WordVectorSerializer.writeWord2VecModel(word2Vec, "./src/main/resources/wordVector/twitter100WithName.txt");
    }
}
