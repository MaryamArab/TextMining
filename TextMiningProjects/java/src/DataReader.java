import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

public class DataReader {

    private static String inputFileName= "Data/Input/reut2-021.sgm";
    private static final String outFileName = "Data/Output/reut021/";

//    public static void main(String[] args) {
//        try {
//            new DataReader().generateContent(null);
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }
//    }

    private String generateContent(String url) throws IOException {
        StringBuffer sb = new StringBuffer();
        Map<String, String> map = new HashMap<String, String>();

        InputStream is = null;
        is = new FileInputStream(inputFileName);
        Document doc = Jsoup.parse(is, "UTF-8", "www.reuters.com", Parser.xmlParser());
        Elements titles = doc.select("TITLE");
        Elements topics = doc.select("TOPICS");
        Elements siblings = null;
        Elements bodies = null;
        System.out.println("Parsing...");


        BufferedWriter br = new BufferedWriter(new FileWriter("labels-021.csv"));
        StringBuilder sbb = new StringBuilder();
        for(int i = 0; i<topics.size(); i++){
            String x = topics.get(i).toString().substring(8)
                    .replace("<D>", "\n")
                    .replace("</D>", "")
                    .replace("</TOPICS>", "")
                    .replaceAll("(?m)^[ \t]*\r?\n", "")
                    ;

            System.out.println(i+1+18031 + ": " + x);
            if ( x.toString() == "\t")
                System.out.print("this x "+i+ "is empty");
            sbb.append(i+1+18031 +"\t"+ x);
            sbb.append("\n");
        }
        br.write(String.valueOf(sbb));
        br.close();
        System.out.println("**********************************");
        for (Element title : titles) {
            siblings = title.siblingElements();
           // System.out.println("TITLE -->"+ title.text());

            for (Element sib : siblings) {
                bodies = sib.getElementsByTag("BODY");
                for (Element body : bodies)
                    if (body != null) {
                        //System.out.println("BODY -->"+ body.text());
                        map.put(title.text(), body.text());
                        //System.out.println("Map Content: "+ map.get(title.text()));
                    }
            }
        }
        if (!map.isEmpty())
            writeToFile(map);

        return null;
    }

    private void writeToFile(Map<String, String> outputMap) {
        BufferedWriter bw = null;
        FileWriter fw = null;

            int i=18031;
            for (Map.Entry<String, String> entry : outputMap.entrySet()) {
                try{
                    i++;
                    StringBuffer sb = new StringBuffer();
                    //System.out.println("Writing to file: " +outFileName+entry.getKey()+".txt" );
                    fw = new FileWriter(outFileName+i+".txt" );
                    bw = new BufferedWriter(fw);

                    //System.out.println("Entry"+entry);
                    //bw.write(sb.append(entry.getKey()).append("|").append(entry.getValue()).append("\n").toString());
                    sb.append(entry.getKey()).append("|").append(entry.getValue()).append("\n");

                    //System.out.println(sb.append(entry.getKey()));//.append("|").append(entry.getValue()).append("\n").toString());
                    bw.write(sb.toString());
                }
             catch (IOException e) {

                e.printStackTrace();

            } finally {

                try {

                    if (bw != null)
                        bw.close();

                    if (fw != null)
                        fw.close();

                } catch (IOException ex) {

                    ex.printStackTrace();

                }

            }
            }
            System.out.println("index: "+i);
            System.out.println("Done");



        }
    }

