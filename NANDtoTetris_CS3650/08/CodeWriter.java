import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * The CodeWriter class is responsible for translating virtual machine (VM)
 * code into assembly code for a specific target platform. It provides methods
 * to generate assembly code for arithmetic and logical operations, push and
 * pop operations, function calls, function returns, labels, and branching.
 *
 * Usage:
 * 1. Create a CodeWriter object with the VM input file name and the output
 * assembly file name.
 * 2. Use the provided methods to write different types of VM instructions
 * and control structures.
 * - `writeArithmetic` for arithmetic and logical operations.
 * - `writePushPop` for push and pop operations.
 * - `writeLabel` to write an assembly label.
 * - `writeGoto` to write a jump instruction to a label.
 * - `writeIf` to write a conditional jump instruction based on the stack value.
 * - `writeCall` to generate code for calling a function.
 * - `writeReturn` to generate code for returning from a function.
 * - `writeFunction` to define a function and allocate local variables.
 * 3. Use `close` to close the output assembly file when done.
 *
 * @author Maayan Israel
 * @version 1.0
 * @since 11/2/2023
 */

public class CodeWriter {

    String outputFileName;

    String vmFileName;
    int numOfFunctionCalls;
    int numForUnique;
    File outFile;

    FileWriter outputFile; // FileWriter to write to the output file

    public CodeWriter(String vmInputFileName, String outputFileNameString) throws IOException {
        vmFileName = vmInputFileName.substring(0, vmInputFileName.indexOf("."));
        setFileName(outputFileNameString);
        writeInit();
    }

    // Set the output file name and initialize FileWriter
    public void setFileName(String outFileName) throws IOException {
        outFile = new File(outFileName);
        outputFileName = outFileName;
        outputFile = new FileWriter(outFile);
    }

    // Write assembly code for arithmetic operations
    public void writeArithmetic(String command) throws IOException {
        if (command.equals("add")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M"
                    + "\n" + "@SP" + "\n" + "M=M-1"
                    + "\n" + "@SP" + "\n" + "A=M"
                    + "\n" + "M=M+D" + "\n" + "@SP"
                    + "\n" + "M=M+1" + "\n");
        } else if (command.equals("sub")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M"
                    + "\n" + "@SP" + "\n" + "M=M-1"
                    + "\n" + "@SP" + "\n" + "A=M"
                    + "\n" + "M=M-D" + "\n" + "@SP"
                    + "\n" + "M=M+1" + "\n");
        } else if (command.equals("neg")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "D=-D"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=D"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");
        } else if (command.equals("eq")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                    + "\n" + "M=D" + "\n" + "@SP" + "\n" + "AM=M-1"
                    + "\n" + "D=M" + "\n" + "@R13" + "\n" + "D=D-M"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=-1"
                    + "\n" + "@EQUAL" + "\n" + "D;JEQ" + "\n" + "@SP"
                    + "\n" + "A=M" + "\n" + "M=0" + "\n" + "(EQUAL)"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");
        } else if (command.equals("gt")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                    + "\n" + "M=D" + "\n" + "@SP" + "\n" + "AM=M-1"
                    + "\n" + "D=M" + "\n" + "@R13" + "\n" + "D=D-M"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=-1"
                    + "\n" + "@gtTRUE" + "\n" + "D;JGT" + "\n" + "@SP"
                    + "\n" + "A=M" + "\n" + "M=0" + "\n" + "(gtTRUE)"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");

        } else if (command.equals("lt")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                    + "\n" + "M=D" + "\n" + "@SP" + "\n" + "AM=M-1"
                    + "\n" + "D=M" + "\n" + "@R13" + "\n" + "D=D-M"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=-1"
                    + "\n" + "@ltTRUE" + "\n" + "D;JLT" + "\n" + "@SP"
                    + "\n" + "A=M" + "\n" + "M=0" + "\n" + "(ltTRUE)"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");

        } else if (command.equals("and")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                    + "\n" + "M=D" + "\n" + "@SP" + "\n" + "AM=M-1"
                    + "\n" + "D=M" + "\n" + "@R13" + "\n" + "D=D&M"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=D"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");
        } else if (command.equals("or")) {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                    + "\n" + "M=D" + "\n" + "@SP" + "\n" + "AM=M-1"
                    + "\n" + "D=M" + "\n" + "@R13" + "\n" + "D=D|M"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=D"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");
        } else {
            outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "D=!D"
                    + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=D"
                    + "\n" + "@SP" + "\n" + "M=M+1" + "\n");
        }
    }

    // Write assembly code for push and pop operations
    public void writePushPop(String command, String segment, int index) throws IOException {
        if (command.equals("C_PUSH")) {
            if (segment.equals("constant")) {
                outputFile.write("@" + index + "\n" + "D=A" + "\n" + "@SP" + "\n"
                        + "A=M" + "\n" + "M=D" + "\n"
                        + "@SP" + "\n" + "M=M+1" + "\n");
            } else if (segment.equals("local")) {
                outputFile.write("@LCL" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");
            } else if (segment.equals("argument")) {
                outputFile.write("@ARG" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");

            } else if (segment.equals("this")) {
                outputFile.write("@THIS" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");

            } else if (segment.equals("that")) {
                outputFile.write("@THAT" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");

            } else if (segment.equals("pointer")) {
                outputFile.write("@3" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");

            } else if (segment.equals("temp")) {
                outputFile.write("@5" + "\n" + "D=M" + "\n" + "@" + index
                        + "\n" + "A=D+A" + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M"
                        + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");

            } else {
                String assemblyVMFileName = vmFileName + "." + index;
                outputFile.write("@" + assemblyVMFileName + "\n" + "D=M"
                        + "\n" + "@SP" + "\n" + "A=M" + "\n" + "M=D" + "\n" + "@SP"
                        + "\n" + "M=M+1" + "\n");
            }
        } else {
            if (segment.equals("local")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@LCL" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");
            } else if (segment.equals("argument")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@ARG" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment.equals("this")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@THIS" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment.equals("that")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@THAT" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment.equals("pointer")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@3" + "\n" + "D=A"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment.equals("temp")) {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@5" + "\n" + "D=A"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");
            } else {
                String assemblyVMFileName = vmFileName + "." + index;
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n"
                        + "@" + assemblyVMFileName + "\n" + "M=D" + "\n");
            }
        }

    }

    // Close the output file
    public void close() throws IOException {
        outputFile.close();
    }

    // Placeholder for initializing the virtual machine
    public void writeInit() throws IOException {
        outputFile.write("@256" + "\n" + "D=A" + "\n" + "@SP" + "\n" + "M=D" + "\n" +
                "Call Sys.init 0" + "\n");
    }

    // Write an assembly label
    public void writeLabel(String label) throws IOException {
        outputFile.write("(" + label + ")" + "\n");
    }

    // Write an assembly jump instruction
    public void writeGoto(String label) throws IOException {
        outputFile.write("@" + label + "\n" + "0;JMP" + "\n");
    }

    // Write an assembly conditional jump instruction
    public void writeIf(String label) throws IOException {
        outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n"
                + "@" + label + "\n" + "D;JEQ" + "\n");
    }

    // Write assembly code for calling a function
    public void writeCall(String functionName, int numArgs) throws IOException {
        numForUnique++;
        outputFile.write("@retAddress" + numForUnique + "\n" + "D=A" + "\n" + "@SP" + "\n" +
                "M=D" + "\n" + "@SP" + "\n" + "M=M+1" + "\n" + "@LCL" + "\n" + "D=M" + "\n" +
                "@SP" + "\n" + "M=D" + "\n" + "@SP" + "\n" + "M=M+1" + "\n" + "@ARG" + "\n" +
                "D=M" + "\n" + "@SP" + "\n" + "M=D" + "\n" + "@SP" + "\n" + "M=M+1" + "\n" +
                "@THIS" + "\n" + "D=M" + "\n" + "@SP" + "\n" + "M=D" + "\n" + "@SP" + "\n" +
                "M=M+1" + "\n" + "@THAT" + "\n" + "D=M" + "\n" + "@SP" + "\n" + "M=D" + "\n" +
                "@SP" + "\n" + "M=M+1" + "\n" + "@SP" + "\n" + "D=M" + "\n" + "@5" + "\n" +
                "D=D-A" + "\n" + "@" + numArgs + "\n" + "D=D-A" + "@ARG" + "\n" + "M=D" + "\n" +
                "SP" + "\n" + "D=M" + "\n" + "@LCL" + "\n" + "M=D" + "\n" + "@" + vmFileName + "." +
                functionName + "\n" + "0;JMP" + "\n" + "(retAddress" + numForUnique + ")" + "\n");
    }

    // Write assembly code for returning from a function
    public void writeReturn() throws IOException {
        outputFile.write("@LCL" + "\n" + "D=M" + "\n" + "@R13" + "\n" + "M=D" + "\n" +
                "@5" + "\n" + "D=A" + "\n" + "@R13" + "\n" + "D=M-D" + "\n" + "A=D" + "\n" + "D=M" + "\n" +
                "@R14" + "\n" + "M=D" + "\n" +
                "@SP" + "\n" + "A=M" + "\n" + "D=M" + "\n" + "@SP" + "\n" + "M=M-1" + "\n" + "@ARG" + "\n" + "A=M"
                + "\n" + "M=D" + "\n" +
                "@1" + "\n" + "D=A" + "\n" + "@ARG" + "\n" + "D=M+D" + "\n" + "@SP" + "\n" + "M=D" + "\n" +
                "@1" + "\n" + "D=A" + "\n" + "@R13" + "\n" + "D=M-D" + "\n" + "@THAT" + "\n" + "M=D" + "\n" +
                "@2" + "\n" + "D=A" + "\n" + "@R13" + "\n" + "D=M-D" + "\n" + "@THIS" + "\n" + "M=D" + "\n" +
                "@3" + "\n" + "D=A" + "\n" + "@R13" + "\n" + "D=M-D" + "\n" + "@ARG" + "\n" + "M=D" + "\n" +
                "@4" + "\n" + "D=A" + "\n" + "@R13" + "\n" + "D=M-D" + "\n" + "@LCL" + "\n" + "M=D" + "\n" +
                "R14" + "\n" + "D=M" + "\n" + "A=D" + "\n" + "0;JMP" + "\n");
    }

    // Write assembly code for defining a function
    public void writeFunction(String functionName, int numLocals) throws IOException {
        numOfFunctionCalls++;
        outputFile.write("(" + vmFileName + "." + functionName + ")" + "\n" +
                "@" + numLocals + "\n" + "D=A" + "\n" + "@R13" + "\n" +
                "M=D" + "\n" + "(LOOP" + numOfFunctionCalls + ")" + "\n" +
                "@R13" + "\n" + "D=M" + "\n" + "@ENDLOOP" + numOfFunctionCalls + "\n" +
                "D;JEQ" + "\n" + "@R13" + "\n" + "M=M-1" + "\n" + "@SP" + "\n" +
                "A=M" + "\n" + "M=0" + "\n" + "@SP" + "\n" + "M=M+1" + "\n" +
                "@LOOP" + numOfFunctionCalls + "\n" + "0;JMP" + "\n" + "(ENDLOOP" + numOfFunctionCalls + ")" + "\n");
    }
}
