import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Scanner;

public class PomonaTransitSystem {

    static String url = "jdbc:mysql://localhost:3306/LAB4";
    static String username = "testLabUser";
    static String password = "testLabPassword";

    static Connection connection = null;
    static Statement statement = null;
    static ResultSet resultSet = null;

    /*
     * Helper Functions
     */
    public static ResultSet executeStatement(String sql) throws SQLException {
        statement = connection.createStatement();
        return statement.executeQuery(sql);
    }

    public static void dropAllTables() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            // disable FK checks
            stmt.executeUpdate("SET FOREIGN_KEY_CHECKS = 0;");
            stmt.executeUpdate("DROP TABLE IF EXISTS ActualTripStopInfo;");
            stmt.executeUpdate("DROP TABLE IF EXISTS TripStopInfo;");
            stmt.executeUpdate("DROP TABLE IF EXISTS TripOffering;");
            stmt.executeUpdate("DROP TABLE IF EXISTS Stop;");
            stmt.executeUpdate("DROP TABLE IF EXISTS Driver;");
            stmt.executeUpdate("DROP TABLE IF EXISTS Bus;");
            stmt.executeUpdate("DROP TABLE IF EXISTS Trip;");
            // re‐enable FK checks
            stmt.executeUpdate("SET FOREIGN_KEY_CHECKS = 1;");
            System.out.println("All tables dropped.");
        }
    }

    public static void createTables() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS Trip (" +
                            "  TripNumber INT PRIMARY KEY," +
                            "  StartLocationName VARCHAR(100)," +
                            "  DestinationName VARCHAR(100)" +
                            ");");
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS Bus (" +
                            "  BusID INT PRIMARY KEY," +
                            "  Model VARCHAR(100)," +
                            "  Year INT" +
                            ");");
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS Driver (" +
                            "  DriverName VARCHAR(100) PRIMARY KEY," +
                            "  DriverTelephoneNumber VARCHAR(20)" +
                            ");");
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS Stop (" +
                            "  StopNumber INT PRIMARY KEY," +
                            "  StopAddress VARCHAR(200)" +
                            ");");

            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS TripOffering (" +
                            "  TripNumber INT," +
                            "  Date DATE," +
                            "  ScheduledStartTime TIME," +
                            "  ScheduledArrivalTime TIME," +
                            "  DriverName VARCHAR(100)," +
                            "  BusID INT," +
                            "  PRIMARY KEY (TripNumber, Date, ScheduledStartTime)," +
                            "  FOREIGN KEY (TripNumber) REFERENCES Trip(TripNumber)," +
                            "  FOREIGN KEY (DriverName) REFERENCES Driver(DriverName)," +
                            "  FOREIGN KEY (BusID) REFERENCES Bus(BusID)" +
                            ");");
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS TripStopInfo (" +
                            "  TripNumber INT," +
                            "  StopNumber INT," +
                            "  SequenceNumber INT," +
                            "  DrivingTime INT," +
                            "  PRIMARY KEY (TripNumber, StopNumber)," +
                            "  FOREIGN KEY (TripNumber) REFERENCES Trip(TripNumber)," +
                            "  FOREIGN KEY (StopNumber) REFERENCES Stop(StopNumber)" +
                            ");");
            stmt.executeUpdate(
                    "CREATE TABLE IF NOT EXISTS ActualTripStopInfo (" +
                            "  TripNumber INT," +
                            "  Date DATE," +
                            "  ScheduledStartTime TIME," +
                            "  StopNumber INT," +
                            "  ScheduledArrivalTime TIME," +
                            "  ActualStartTime TIME," +
                            "  ActualArrivalTime TIME," +
                            "  NumberOfPassengerIn INT," +
                            "  NumberOfPassengerOut INT," +
                            "  PRIMARY KEY (TripNumber, Date, ScheduledStartTime, StopNumber)," +
                            "  FOREIGN KEY (TripNumber, Date, ScheduledStartTime) " +
                            "    REFERENCES TripOffering(TripNumber, Date, ScheduledStartTime) ON DELETE CASCADE," +
                            "  FOREIGN KEY (StopNumber) REFERENCES Stop(StopNumber)" +
                            ");");
            System.out.println("All tables created (if not existed).");
        }
    }

    public static void populateDummyData() throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            // Trip
            stmt.executeUpdate(
                    "INSERT INTO Trip (TripNumber, StartLocationName, DestinationName) VALUES " +
                            "(101, 'City A', 'City B'), " +
                            "(102, 'City C', 'City D');");
            // Bus
            stmt.executeUpdate(
                    "INSERT INTO Bus (BusID, Model, Year) VALUES " +
                            "(1, 'Volvo 9700', 2019), " +
                            "(2, 'Scania Touring', 2020);");
            // Driver
            stmt.executeUpdate(
                    "INSERT INTO Driver (DriverName, DriverTelephoneNumber) VALUES " +
                            "('John Doe', '555-1234'), " +
                            "('Jane Smith', '555-5678');");
            // Stop
            stmt.executeUpdate(
                    "INSERT INTO Stop (StopNumber, StopAddress) VALUES " +
                            "(1, '123 Main St'), " +
                            "(2, '456 Elm St'), " +
                            "(3, '789 Oak St'), " +
                            "(4, '101 Maple Ave');");
            // TripOffering
            stmt.executeUpdate(
                    "INSERT INTO TripOffering " +
                            "(TripNumber, Date, ScheduledStartTime, ScheduledArrivalTime, DriverName, BusID) VALUES " +
                            "(101, '2025-05-01', '08:00:00', '10:00:00', 'John Doe', 1), " +
                            "(102, '2025-05-02', '09:00:00', '11:30:00', 'Jane Smith', 2);");
            // ActualTripStopInfo
            stmt.executeUpdate(
                    "INSERT INTO ActualTripStopInfo " +
                            "(TripNumber, Date, ScheduledStartTime, StopNumber, ScheduledArrivalTime, ActualStartTime, ActualArrivalTime, NumberOfPassengerIn, NumberOfPassengerOut) VALUES "
                            +
                            "(101, '2025-05-01', '08:00:00', 1, '08:15:00', '08:05:00', '08:20:00', 5, 0), " +
                            "(101, '2025-05-01', '08:00:00', 2, '08:30:00', '08:25:00', '08:35:00', 3, 1), " +
                            "(102, '2025-05-02', '09:00:00', 3, '09:20:00', '09:05:00', '09:25:00', 7, 2), " +
                            "(102, '2025-05-02', '09:00:00', 4, '09:40:00', '09:30:00', '09:45:00', 4, 3);");
            // TripStopInfo
            stmt.executeUpdate(
                    "INSERT INTO TripStopInfo (TripNumber, StopNumber, SequenceNumber, DrivingTime) VALUES " +
                            "(101, 1, 1, 15), " +
                            "(101, 2, 2, 15), " +
                            "(102, 3, 1, 20), " +
                            "(102, 4, 2, 20);");
            System.out.println("Dummy data populated successfully.");
        }
    }
    /*
     * Helper Functions END
     */

    public static void numberOne() {
        // Implement the logic for viewing all trips based on a Source, Destination, and
        // Date
    }

    public static void numberTwo() {
        // Implement the logic for editing a schedule of a Trip Offering
    }

    public static void numberThree() {
        // Implement the logic for displaying the stops for a given Trip
    }

    public static void numberFour() {
        // Implement the logic for displaying the weekly schedule of a given driver and
        // date
    }

    public static void numberFive() {
        // Implement the logic for adding a driver to the system
    }

    public static void numberSix() {
        // Implement the logic for adding a bus to the system
    }

    public static void numberSeven() {
        // Implement the logic for deleting a bus from the system
    }

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            connection = DriverManager.getConnection(url, username, password);

            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    dropAllTables();
                    if (connection != null && !connection.isClosed()) {
                        connection.close();
                    }
                } catch (SQLException e) {
                    System.err.println("Error during shutdown: " + e.getMessage());
                }
            }));

            createTables();
            populateDummyData();

            System.out.println("Welcome to Pomona Transit System!");

            int choice = 0;
            boolean invalidInput = true;
            while (invalidInput) {
                System.out.println("Please enter your desired choice:");

                System.out.println("1. View all trips based on a Source, Destination, and Date");
                System.out.println("2. Edit a schedule of a Trip Offering");
                System.out.println("3. Display the stops for a given Trip");
                System.out.println("4. Display the weekly shcedule of a given driver and date.");
                System.out.println("5. Add a drive to the system");
                System.out.println("6. Add a bus to the system");
                System.out.println("7. Delete a bus from the system");

                System.out.print("Enter your choice: ");
                choice = scan.nextInt();

                if (choice <= 7 && choice >= 1) {
                    invalidInput = false;
                } else {
                    System.out.println("Invalid input. Please try again.");
                }
            }

            switch (choice) {
                case 1:
                    numberOne();
                    break;
                case 2:
                    numberTwo();
                    break;
                case 3:
                    numberThree();
                    break;
                case 4:
                    numberFour();
                    break;
                case 5:
                    numberFive();
                    break;
                case 6:
                    numberSix();
                    break;
                case 7:
                    numberSeven();
                    break;
            }

        } catch (ClassNotFoundException e) {
            System.err.println("JDBC driver not found: " + e.getMessage());
        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        } finally {
            scan.close();
        }
    }
}