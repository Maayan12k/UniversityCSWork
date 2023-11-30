import java.io.File;
import java.io.IOException;

public class VMTranslator {

    Parser parsy;
    CodeWriter coder;
    String outFileName;
    File fileOut = new File(outFileName); // Output file for assembly code

    public static void main(String[] args) throws IOException {

        File directory = new File("null");
        VMTranslator entry = new VMTranslator();

        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            if (files != null) {
                entry.coder.writeInit();
                for (File file : files) {
                    if (file.isFile()) {
                        entry = new VMTranslator(file.getName() + ".vm", file.getName() + ".asm");
                    }
                }
            }
        } else {
            entry.coder.writeInit();
            entry = new VMTranslator(directory.getName() + ".vm", directory.getName() + ".asm");
            entry.coder.close();
        }
        entry.coder.close();
    }

    public VMTranslator(String inputFileName, String outputFileName) throws IOException {
        outFileName = outputFileName;
        parsy = new Parser(inputFileName);
        coder = new CodeWriter(inputFileName, outputFileName, fileOut);
        String currentCommandType;

        while (parsy.hasMoreCommands()) {
            parsy.advance();
            currentCommandType = parsy.commandType();
            if (currentCommandType.equals("C_PUSH") || currentCommandType.equals("C_POP")) {
                coder.writePushPop(currentCommandType, parsy.arg1(), parsy.arg2());
            } else if (currentCommandType.equals("C_ARITHMETIC")) {
                coder.writeArithmetic(parsy.arg1());
            } else if (currentCommandType.equals("C_LABEL")) {
                coder.writeLabel(parsy.arg1());
            } else if (currentCommandType.equals("C_GOTO")) {
                coder.writeGoto(parsy.arg1());
            } else if (currentCommandType.equals("C_IF")) {
                coder.writeIf(parsy.arg1());
            } else if (currentCommandType.equals("C_FUNCTION")) {
                coder.writeFunction(parsy.arg1(), parsy.arg2());
            } else if (currentCommandType.equals("C_CALL")) {
                coder.writeCall(parsy.arg1(), parsy.arg2());
            } else if (currentCommandType.equals("C_RETURN")) {
                coder.writeReturn();
            } else if (currentCommandType.equals("COMMENT") || currentCommandType.equals("EMPTY_LINE")) {

            }
        }

    }

}
