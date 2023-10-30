import java.io.IOException;

public class VMTranslator {

    Parser parsy;
    CodeWriter coder;

    public static void main(String[] args) throws IOException {
        new VMTranslator("StackTest.vm", "StackTest.asm");
    }

    public VMTranslator(String inputFileName, String outputFileName) throws IOException {
        parsy = new Parser(inputFileName);
        coder = new CodeWriter(inputFileName, outputFileName);
        String currentCommandType;
        String arg1;
        int arg2;

        while (parsy.hasMoreCommands()) {
            parsy.advance();
            currentCommandType = parsy.commandType();
            if (currentCommandType == "C_PUSH" || currentCommandType == "C_POP") {
                arg1 = parsy.arg1();
                arg2 = parsy.arg2();
                coder.writePushPop(currentCommandType, arg1, arg2);
            } else if (currentCommandType == "C_ARITHMETIC") {
                arg1 = parsy.arg1();
                coder.writeArithmetic(arg1);
            }
        }
        coder.close();
    }

}
