package utils;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;
import moa.streams.InstanceStream;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Stream;

public class FileUtils {

    public static void appendFile(String inputPath, String outputPath) {
        FileReader fr;
        FileWriter fw;

        try {
            fr = new FileReader(inputPath);
            (new File(outputPath).getParentFile()).mkdirs();
            fw = new FileWriter(outputPath, true);

            int c = fr.read();
            while(c !=- 1) {
                fw.write(c);
                c = fr.read();
            }

            fr.close();
            fw.close();

        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public static void parseRawResultToLatexTable(String inputPath, String outputPath) {
        DecimalFormat df2 = new DecimalFormat("0.00");
        String previousLabel = "";
        String newLine = "";

        BufferedWriter writer = null;
        BufferedReader reader = null;

        try {
            (new File(outputPath).getParentFile()).mkdirs();
            writer = new BufferedWriter(new FileWriter(outputPath));

            reader = new BufferedReader(new FileReader(inputPath));
            String line = reader.readLine();

            while (true) {
                String[] lineElements = line.split(",");
                String label = lineElements[0].split("-")[0];
                String value = df2.format(Double.parseDouble(lineElements[1])*100) + "\\%";

                if (!label.equals(previousLabel)) {
                    if (!previousLabel.isEmpty()) writer.write(newLine + "\\\\\r\n");
                    previousLabel = label;
                    newLine = "& \\texttt{" + label + "} ";
                }

                newLine += "& " + value + " ";

                line = reader.readLine();
                if (line == null) {
                    writer.write(newLine + "\\\\");
                    break;
                }
            }

        }  catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (writer != null) try { writer.close(); } catch (IOException ex) { ex.printStackTrace(); }
            if (reader != null) try { reader.close(); } catch (IOException ex) { ex.printStackTrace(); }
        }
    }

    public static void parseRawResultToCSV(String inputPath, String outputPath) {
        String previousLabel = "";
        BufferedWriter writer = null;
        BufferedReader reader = null;

        try {
            (new File(outputPath).getParentFile()).mkdirs();
            writer = new BufferedWriter(new FileWriter(outputPath, true));
            writer.write( inputPath + "\r\n");

            reader = new BufferedReader(new FileReader(inputPath));
            String line = reader.readLine();

            while (line != null) {
                String[] lineElements = line.split(",");
                String label = lineElements[0].split("#")[0];
                String value = lineElements[1];

                if (!label.equals(previousLabel)) {
                    previousLabel = label;
                    writer.write("\r\n" + label + ",");
                }
                writer.write(value + ",");

                line = reader.readLine();
            }

            writer.write( "\r\n\n\n\n");

        }  catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (writer != null) try { writer.close(); } catch (IOException ex) { ex.printStackTrace(); }
            if (reader != null) try { reader.close(); } catch (IOException ex) { ex.printStackTrace(); }
        }
    }

    public static LinkedHashMap<String, Double[]> parseSeriesToHashMap(String inputPath) {
        LinkedHashMap<String, Double[]> seriesHashMap = new LinkedHashMap<>();

        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(inputPath));
            String line = reader.readLine();

            while (line != null) {
                String[] labelValuesPair = line.split(",", 2);
                String label = labelValuesPair[0];
                Double[] values = Stream.of(labelValuesPair[1].split(","))
                        .mapToDouble(Double::parseDouble)
                        .boxed()
                        .toArray(Double[]::new);

                seriesHashMap.put(label, values);
                line = reader.readLine();
            }

        }  catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (reader != null) try { reader.close(); } catch (IOException ex) { ex.printStackTrace(); }
        }

        return seriesHashMap;
    }

    public static LinkedHashMap<String, Double> parseAveragesToHashMap(String inputPath) {
        LinkedHashMap<String, Double> averagesHashMap = new LinkedHashMap<>();
        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(inputPath));
            String line = reader.readLine();

            while (line != null) {
                String[] labelValuePair = line.split("#")[1].split(",");
                String label = labelValuePair[0];
                String value = labelValuePair[1];

                averagesHashMap.put(label, Double.parseDouble(value));

                line = reader.readLine();
            }

        }  catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (reader != null) try { reader.close(); } catch (IOException ex) { ex.printStackTrace(); }
        }

        return averagesHashMap;
    }

    public static List<String> getDirFileNames(File dir, String filter) {
        List<String> measurementNames = new ArrayList<>();
        File[] outputFiles = dir.listFiles((d, name) -> name.endsWith(filter));
        assert outputFiles != null;

        for (File outputFile : outputFiles) {
            measurementNames.add(outputFile.getName().split("\\.")[0]);
        }

        return measurementNames;
    }

    public static void writeArffToFile(InstancesHeader streamHeader, ArrayList<InstanceExample> stream, String outputPath) {
        File destFile = new File(outputPath);
        destFile.getParentFile().mkdirs();
        System.out.println("Writing in " + destFile.getAbsolutePath());

        try {
            Writer w = new BufferedWriter(new FileWriter(destFile));

            w.write(streamHeader.toString());
            w.write("\n");

            for (InstanceExample instance : stream) {
                w.write(instance.getData().toString());
                w.write("\n");
            }

            w.close();
        } catch (Exception e) {
            throw new RuntimeException("Failed writing to file " + destFile, e);
        }
    }
}
