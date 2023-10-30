import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class CodeWriter {

    String outputFileName;
    File fileOut; // Output file for binary results

    String vmFileName;

    FileWriter outputFile; // FileWriter to write to the output file

    public CodeWriter(String vmInputFileName, String outputFileNameString) throws IOException {
        vmFileName = vmInputFileName.substring(0, vmInputFileName.indexOf("."));
        setFileName(outputFileNameString);
    }

    public void setFileName(String outFileName) throws IOException {
        outputFileName = outFileName;
        fileOut = new File(outputFileName);
        outputFile = new FileWriter(fileOut);
    }

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

    public void writePushPop(String command, String segment, int index) throws IOException {
        if (command == "C_PUSH") {
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
            if (segment == "local") {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@LCL" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");
            } else if (segment == "argument") {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@ARG" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment == "this") {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@THIS" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment == "that") {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@THAT" + "\n" + "D=M"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment == "pointer") {
                outputFile.write("@SP" + "\n" + "AM=M-1" + "\n" + "D=M" + "\n" + "@R13"
                        + "\n" + "M=D" + "\n" + "@3" + "\n" + "D=A"
                        + "\n" + "@" + index + "\n" + "D=D+A" + "\n" + "@R14"
                        + "\n" + "M=D" + "\n" + "@R13" + "\n" + "D=M"
                        + "\n" + "@R14" + "\n" + "A=M" + "\n" + "M=D" + "\n");

            } else if (segment == "temp") {
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

    public void close() throws IOException {
        outputFile.close();
    }

}
