import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class CodeWriter {

    String outputFileName;
    File fileOut = new File(outputFileName); // Output file for binary results

    FileWriter outputFile = new FileWriter(fileOut); // FileWriter to write to the output file

    public CodeWriter() throws IOException {

    }

    public void setFileName(String outFileName) {

    }

}
