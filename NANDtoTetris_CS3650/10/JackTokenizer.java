import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class JackTokenizer {

    public final static int KEYWORD = 1;
    public final static int SYMBOL = 2;
    public final static int IDENTIFIER = 3;
    public final static int INT_CONST = 4;
    public final static int STRING_CONST = 5;

    // constant for keyword
    public final static int CLASS = 10;
    public final static int METHOD = 11;
    public final static int FUNCTION = 12;
    public final static int CONSTRUCTOR = 13;
    public final static int INT = 14;
    public final static int BOOLEAN = 15;
    public final static int CHAR = 16;
    public final static int VOID = 17;
    public final static int VAR = 18;
    public final static int STATIC = 19;
    public final static int FIELD = 20;
    public final static int LET = 21;
    public final static int DO = 22;
    public final static int IF = 23;
    public final static int ELSE = 24;
    public final static int WHILE = 25;
    public final static int RETURN = 26;
    public final static int TRUE = 27;
    public final static int FALSE = 28;
    public final static int NULL = 29;
    public final static int THIS = 30;

    private Scanner scanny;
    private String currentToken;
    private int currentTokenType;
    private int pointer;
    private ArrayList<String> tokens;

    private static Pattern tokenPatterns;
    private static String keyWordReg;
    private static String symbolReg;
    private static String intReg;
    private static String strReg;
    private static String idReg;

    private static HashMap<String, Integer> keywordHashMap = new HashMap<String, Integer>();
    private static HashSet<Character> opSet = new HashSet<Character>();

    static {

        keywordHashMap.put("class", CLASS);
        keywordHashMap.put("constructor", CONSTRUCTOR);
        keywordHashMap.put("function", FUNCTION);
        keywordHashMap.put("method", METHOD);
        keywordHashMap.put("field", FIELD);
        keywordHashMap.put("static", STATIC);
        keywordHashMap.put("var", VAR);
        keywordHashMap.put("int", INT);
        keywordHashMap.put("char", CHAR);
        keywordHashMap.put("boolean", BOOLEAN);
        keywordHashMap.put("void", VOID);
        keywordHashMap.put("true", TRUE);
        keywordHashMap.put("false", FALSE);
        keywordHashMap.put("null", NULL);
        keywordHashMap.put("this", THIS);
        keywordHashMap.put("let", LET);
        keywordHashMap.put("do", DO);
        keywordHashMap.put("if", IF);
        keywordHashMap.put("else", ELSE);
        keywordHashMap.put("while", WHILE);
        keywordHashMap.put("return", RETURN);

        opSet.add('+');
        opSet.add('-');
        opSet.add('*');
        opSet.add('/');
        opSet.add('&');
        opSet.add('|');
        opSet.add('<');
        opSet.add('>');
        opSet.add('=');

    }

    public JackTokenizer(File inputFile) throws FileNotFoundException {
        scanny = new Scanner(inputFile);
        String fileStringWithoutLineComments = "";
        String line = "";

        while (scanny.hasNext()) {
            line = removeComments(scanny.nextLine()).trim();

            if (line.length() > 0) {
                fileStringWithoutLineComments += line + "\n";
            }
        }

        String fileStringWithLineAndBlockComments = removeBlockComments(fileStringWithoutLineComments);

        initializeRegex();

        Matcher m = tokenPatterns.matcher(fileStringWithLineAndBlockComments);
        tokens = new ArrayList<String>();
        pointer = 0;

        while (m.find()) {
            tokens.add(m.group());
        }

        currentToken = "";
        currentTokenType = -1;
    }

    /**
     * init regex we need in tokenizer
     */
    private void initializeRegex() {

        keyWordReg = "";

        for (String seg : keywordHashMap.keySet()) {

            keyWordReg += seg + "|";

        }

        symbolReg = "[\\&\\*\\+\\(\\)\\.\\/\\,\\-\\]\\;\\~\\}\\|\\{\\>\\=\\[\\<]";
        intReg = "[0-9]+";
        strReg = "\"[^\"\n]*\"";
        idReg = "[\\w_]+";

        tokenPatterns = Pattern.compile(keyWordReg + symbolReg + "|" + intReg + "|" + strReg + "|" + idReg);
    }

    public boolean hasMoreTokens() {
        return pointer < tokens.size();
    }

    public void advance() {
        if (hasMoreTokens()) {
            currentToken = tokens.get(pointer);
            pointer++;
        } else {
            throw new IllegalStateException("No more tokens");
        }

        if (currentToken.matches(keyWordReg)) {
            currentTokenType = KEYWORD;
        } else if (currentToken.matches(symbolReg)) {
            currentTokenType = SYMBOL;
        } else if (currentToken.matches(intReg)) {
            currentTokenType = INT_CONST;
        } else if (currentToken.matches(strReg)) {
            currentTokenType = STRING_CONST;
        } else if (currentToken.matches(idReg)) {
            currentTokenType = IDENTIFIER;
        } else {

            throw new IllegalArgumentException("Unknown token:" + currentToken);
        }
    }

    /**
     * Decrements the pointer variable to get back to previous token
     */
    public void goBack() {
        if (pointer > 0) {
            pointer--;
        }
    }

    public String getCurrentToken() {
        return currentToken;
    }

    /**
     * Returns the keyword which is the current token
     * Should be called only when tokeyType() is KEYWORD
     * 
     * @return
     */
    public int keyWord() {

        if (currentTokenType == KEYWORD) {

            return keywordHashMap.get(currentToken);

        } else {
            throw new IllegalStateException("Current token is not a keyword!");
        }
    }

    /**
     * Returns the character which is the current token
     * should be called only when tokenType() is SYMBOL
     * 
     * @return if current token is not a symbol return \0
     */
    public char symbol() {

        if (currentTokenType == SYMBOL) {

            return currentToken.charAt(0);

        } else {
            throw new IllegalStateException("Current token is not a symbol!");
        }
    }

    /**
     * Return the identifier which is the current token
     * should be called only when tokenType() is IDENTIFIER
     * 
     * @return
     */
    public String identifier() {

        if (currentTokenType == IDENTIFIER) {

            return currentToken;

        } else {
            throw new IllegalStateException("Current token is not an identifier!");
        }
    }

    /**
     * Returns the integer value of the current token
     * should be called only when tokenType() is INT_CONST
     * 
     * @return
     */
    public int intVal() {

        if (currentTokenType == INT_CONST) {

            return Integer.parseInt(currentToken);
        } else {
            throw new IllegalStateException("Current token is not an integer constant!");
        }
    }

    /**
     * Returns the string value of the current token
     * without the double quotes
     * should be called only when tokenType() is STRING_CONST
     * 
     * @return
     */
    public String stringVal() {

        if (currentTokenType == STRING_CONST) {

            return currentToken.substring(1, currentToken.length() - 1);

        } else {
            throw new IllegalStateException("Current token is not a string constant!");
        }
    }

    /**
     * return if current symbol is a op
     * 
     * @return
     */
    public boolean isOp() {
        return opSet.contains(symbol());
    }

    /**
     * Returns the type of the current token
     * 
     * @return
     */
    public int tokenType() {

        return currentTokenType;
    }

    public static String removeComments(String inputString) {
        int position = inputString.indexOf("//");

        if (position != -1) {

            inputString = inputString.substring(0, position);

        }

        return inputString;
    }

    public static String removeBlockComments(String inputString) {

        int startIndex = inputString.indexOf("/*");

        if (startIndex == -1)
            return inputString;

        String result = inputString;

        int endIndex = inputString.indexOf("*/");

        while (startIndex != -1) {

            if (endIndex == -1) {

                return inputString.substring(0, startIndex - 1);

            }
            result = result.substring(0, startIndex) + result.substring(endIndex + 2);

            startIndex = result.indexOf("/*");
            endIndex = result.indexOf("*/");
        }

        return result;

    }

}