import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class JackAnalyzer {

    public static void main(String[] args) throws FileNotFoundException {

        if (args.length != 1) {

            System.out.println("Run command must be in this format:'java JackAnalyzer [filename/directory]'");

        } else {

            ArrayList<File> jackFiles = new ArrayList<>();

            String fileInName = args[0];
            File pathFile = new File(fileInName);

            String fileOutPath = "", tokenFileOutPath = "";
            File fileOut, tokenFileOut;

            if (pathFile.isFile()) {

                String path = pathFile.getAbsolutePath();

                // Only Jack Files Can be compiled
                if (!path.endsWith(".jack"))
                    throw new IllegalArgumentException(".jack file is required!");

                jackFiles.add(pathFile);

            } else if (pathFile.isDirectory()) {

                jackFiles = getJackFiles(pathFile);

                if (jackFiles.size() == 0) {
                    throw new IllegalArgumentException("Must contain at least one .jack file!");
                }

            }

            for (File f : jackFiles) {

                fileOutPath = f.getAbsolutePath().substring(0, f.getAbsolutePath().lastIndexOf(".")) + ".xml";
                tokenFileOutPath = f.getAbsolutePath().substring(0, f.getAbsolutePath().lastIndexOf(".")) + "T.xml";
                fileOut = new File(fileOutPath);
                tokenFileOut = new File(tokenFileOutPath);

                CompilationEngine compilationEngine = new CompilationEngine(f, fileOut, tokenFileOut);
                compilationEngine.compileClass();

                System.out.println("File created : " + fileOutPath);
                System.out.println("File created : " + tokenFileOutPath);
            }

        }
    }

    public static ArrayList<File> getJackFiles(File dir) {

        ArrayList<File> jackFiles = new ArrayList<File>();
        File[] files = dir.listFiles();

        if (files == null)
            return jackFiles;

        for (File file : files) {
            if (file.getName().endsWith(".jack"))
                jackFiles.add(file);
        }
        return jackFiles;

    }
}