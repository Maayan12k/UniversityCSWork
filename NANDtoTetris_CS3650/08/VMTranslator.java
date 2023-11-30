import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class represents a Virtual Machine (VM) translator. It translates VM
 * code
 * into assembly code for a target machine. The translator consists of a Parser
 * for parsing VM commands and a CodeWriter for generating assembly code.
 *
 * The main function of this class initializes a VMTranslator instance and
 * starts
 * the translation process by reading a VM file and writing the corresponding
 * assembly code to an output file.
 *
 * The translation process involves iterating through each VM command in the
 * input
 * file and calling the appropriate method in the CodeWriter based on the
 * command type.
 * Supported command types include arithmetic operations, push and pop commands,
 * labels,
 * conditional jumps, function declarations, function calls, and return
 * statements.
 *
 * @author Maayan Israel
 * @version 1.0
 */

public class VMTranslator {

    Parser parsy;
    CodeWriter coder;

    /**
     * Return all the .vm files in a directory
     * 
     * @param dir
     * @return
     */
    public static ArrayList<File> getVMFiles(File dir) {

        File[] files = dir.listFiles();

        ArrayList<File> result = new ArrayList<File>();

        for (File f : files) {

            if (f.getName().endsWith(".vm")) {

                result.add(f);

            }

        }

        return result;

    }

    public static void main(String[] args) throws IOException {

        // Create a VMTranslator instance with input and output file names
        VMTranslator entry = new VMTranslator("INPUT_VM_FILE_NAME_HERE//Must be in same directory.vm",
                "INPUT_ASM_FILE_NAME_HERE.asm");
        entry.coder.close(); // Close the output file after translation is complete
    }

    /**
     * Constructs a VMTranslator with the specified input and output file names.
     * Initializes the Parser and CodeWriter objects for parsing and code
     * generation.
     * Translates VM commands in the input file into equivalent assembly code.
     *
     * @param inputFileName  The name of the input VM file to be translated.
     * @param outputFileName The name of the output assembly file where the
     *                       generated
     *                       assembly code will be written.
     * @throws IOException If an I/O error occurs while reading/writing files.
     */
    public VMTranslator(String inputFileName, String outputFileName) throws IOException {
        parsy = new Parser(inputFileName);
        coder = new CodeWriter(inputFileName, outputFileName);
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
                // Ignore comments and empty lines
            }
        }

    }

}
