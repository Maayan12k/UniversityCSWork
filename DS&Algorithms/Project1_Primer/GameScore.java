package DataStructures_Algorithms_CSCI240;

import java.util.ArrayList;

/**
 * GameScore represents a scoreboard for a specific game,
 * which can have up to 10 top scores.
 */
public class GameScore {

    private String gameName;
    private ArrayList<GameEntry> list = new ArrayList<>();

    public GameScore(String gameName) {
        this.gameName = gameName;
    }

    /**
     * Adds a new game entry to the scoreboard if it qualifies as a top score.
     *
     * @param name  Player's name.
     * @param score Player's score.
     * @param date  Date of the game.
     */
    public void addEntry(String name, int score, String date) {
        if (score < 0 || score > 1000) {
            score = 0;
        }
        if (list.size() == 0) {
            list.add(new GameEntry(name, score, date));
            return;
        }
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i).getScore() < score) {
                list.add(i, new GameEntry(name, score, date));
                if (list.size() > 10) {
                    list.remove(10);
                }
                return;
            }
        }
        if (list.size() < 10) {
            list.add(new GameEntry(name, score, date));
        }
    }

    /**
     * Prints the top scores in the scoreboard.
     */
    public void print() {
        System.out.println("Name: " + gameName);
        for (GameEntry e : list) {
            System.out.println(e.getName() + ", " + e.getScore() + ", " + e.getDate());
        }
    }

    public String getGameName() {
        return gameName;
    }

    public static void main(String[] args) {
        // Sample usage and test for GameScore
    }
}




