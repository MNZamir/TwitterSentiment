package training;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;
import java.util.logging.Logger;

public class UpdateWordVec {
    protected static final Logger log = LoggerFactory.getLogger(UpdateWordVec.class);

    public static void main(String[] args) {
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("./src/main/resources/wordVector/twitter100WithName.txt");
        File file = new ClassPathResource("").getFile();
        SentenceIterator iterator = new LineSentenceIterator(file);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIterator(iterator);

        log.info("Uptrain Word2Vec...");
        word2Vec.fit();

        Collection<String> list = word2Vec.wordsNearestSum("malam",10);
        log.info("Closest words to 'day' on 2nd run: " + list);

        //Save word vector
        WordVectorSerializer.writeWordVectors(word2Vec, "./src/main/resources/wordVector/twitter100WithName.txt");
    }
}
