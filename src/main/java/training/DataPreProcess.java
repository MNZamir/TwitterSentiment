//package training;
//
//import java.io.BufferedReader;
//import java.io.IOException;
//import java.nio.charset.StandardCharsets;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//
////import static TextClassificationModel.sentimentClassification.tweets;
//
//public class DataPreProcess {
//
//    static void csvToString(String path) {
//
//        try (BufferedReader br = Files.newBufferedReader(Paths.get(path), StandardCharsets.US_ASCII)) {
//            // read the first line from the text file
//            String line = br.readLine();
//            // loop until all lines are read
//            while (line != null) {
//                String[] attributes = line.split("\\t");
//                Tweet tweet = createTweet(attributes);
//                tweets.add(tweet);
//                line = br.readLine();
//            }
//
////          print text output
//            int num = 700;
//            System.out.println(tweets.get(num).getText());
//            System.out.println(tweets.get(num).getSentiment());
//        }
//        catch (IOException ioe) {
//            ioe.printStackTrace();
//        }
//    }
//
//    private static Tweet createTweet(String[] metadata){
//        String text = metadata[0];
//        String sentiment = metadata[2];
//
//        return new Tweet(text, sentiment);
//    }
//}
//
//class Tweet {
//    private final String text;
//    private final String sentiment;
//
//    public Tweet(String t, String s) {
//        this.text = t;
//        this.sentiment = s;
//    }
//
//    public String getText(){
//        return text;
//    }
//
//    public String getSentiment(){
//        return sentiment;
//    }
//}