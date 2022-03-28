package training;

import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public class SecondModel {


    public static void main(String[] args) throws Exception {
        InputStream tokenStream = new FileInputStream(new File(System.getProperty("user.home"),"en-token.zip"));
        InputStream personModelStream = new FileInputStream(new File(System.getProperty("user.home"),"en-ner-person.zip"));

        TokenizerModel tm = new TokenizerModel(tokenStream);
        TokenizerME tokenizer = new TokenizerME(tm);


    }
}
