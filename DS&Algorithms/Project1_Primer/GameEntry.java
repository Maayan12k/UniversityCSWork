package DataStructures_Algorithms_CSCI240;

public class GameEntry {
    private String name;
    private int score;
    private String date;

    public GameEntry(String name, int score, String date) {
        this.name = name;
        this.date = date;
        this.score = score;
    }

 // Getter and Setter methods...

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getScore() {
		return score;
	}

	public void setScore(int score) {
		this.score = score;
	}

	public String getDate() {
		return date;
	}

	public void setDate(String date) {
		this.date = date;
	}

    }
