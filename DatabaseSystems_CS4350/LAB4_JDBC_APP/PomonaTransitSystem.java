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
            System.out.println("\nAll tables dropped.");
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
                            "  FOREIGN KEY (DriverName) REFERENCES Driver(DriverName) ON UPDATE CASCADE," +
                            "  FOREIGN KEY (BusID) REFERENCES Bus(BusID) ON DELETE SET NULL" +
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
                            "('Jorge Freeman', '535-5678'), " +
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

    public static void numberOne(Scanner scan) {
        System.out.println("Please enter the source:");
        String sourceInput = scan.nextLine();
        System.out.println("Please enter the destination:");
        String destinationInput = scan.nextLine();
        System.out.println("Please enter the date (YYYY-MM-DD):");
        String dateInput = scan.nextLine();

        String sql = "SELECT t.TripNumber, " +
                "       toff.ScheduledStartTime, " +
                "       toff.ScheduledArrivalTime, " +
                "       toff.DriverName, " +
                "       toff.BusID, " +
                "       toff.Date " +
                "FROM Trip AS t " +
                "JOIN TripOffering AS toff " +
                "  ON t.TripNumber = toff.TripNumber " +
                "WHERE t.StartLocationName = '" + sourceInput + "' " +
                "  AND t.DestinationName   = '" + destinationInput + "' " +
                "  AND toff.Date           = '" + dateInput + "' " +
                "ORDER BY toff.ScheduledStartTime;";

        try {
            resultSet = executeStatement(sql);

            while (resultSet.next()) {
                String tripNumber = resultSet.getString("TripNumber");
                String busNumber = resultSet.getString("BusID");
                String driverName = resultSet.getString("DriverName");
                String date = resultSet.getString("Date");

                System.out.println("Trip Number: " + tripNumber + ", Bus Number: " + busNumber + ", Driver Name: "
                        + driverName + ", Date: " + date);
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberTwo(Scanner scan) {
        int choice = 0;
        boolean invalidInput = true;
        while (invalidInput) {
            System.out.println("What would you like to do?");
            System.out.println("1. Delete a trip.");
            System.out.println("2. Add a set of trip offerings.");
            System.out.println("3. Change the driver for a trip offering.");
            System.out.println("4. Change the bus for a given trip offering");

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
                System.out.println("Please enter the trip number you would like to delete:");
                System.out.print("Enter your choice: ");
                int tripNumber = scan.nextInt();
                String sql = "DELETE FROM TripOffering WHERE TripNumber = " + tripNumber + ";";

                try (Statement stmt = connection.createStatement()) {
                    int rowsAffected = stmt.executeUpdate(sql);

                    if (rowsAffected > 0) {
                        System.out.println("Trip deleted successfully.");
                    } else {
                        System.out.println("No trip found with that TripNumber.");
                    }

                } catch (SQLException e) {
                    System.err.println("SQL error: " + e.getMessage());
                }
                break;
            case 2:
                System.out.println("Please enter the Trip Number:");
                tripNumber = scan.nextInt();

                System.out.print("Enter the trip date (YYYY-MM-DD): ");
                String tripDate = scan.next();
                scan.nextLine();

                System.out.print("Enter the scheduled start time (HH:MM:SS): ");
                String startTime = scan.nextLine();

                System.out.print("Enter the scheduled arrival time (HH:MM:SS): ");
                String arrivalTime = scan.nextLine();

                System.out.print("Please enter the new driver name: ");
                String driverName = scan.nextLine();

                System.out.print("Please enter the new Bus's ID: ");
                String busID = scan.nextLine();

                sql = "INSERT INTO TripOffering (TripNumber, Date, ScheduledStartTime, ScheduledArrivalTime, DriverName, BusID)"
                        +
                        " VALUES (" +
                        tripNumber + ", '" +
                        tripDate + "', '" +
                        startTime + "', '" +
                        arrivalTime + "', '" +
                        driverName + "', " +
                        busID + ")";

                try (Statement stmt = connection.createStatement()) {
                    int rowsAffected = stmt.executeUpdate(sql);

                    if (rowsAffected > 0) {
                        System.out.println("Trip Offering inserted successfully.");
                    } else {
                        System.out.println("Trip Offering was not inserted sucessfully.");
                    }

                } catch (SQLException e) {
                    System.err.println("SQL error: " + e.getMessage());
                }

                break;
            case 3:
                System.out.print("Please enter the trip number you would like to change the driver: ");
                tripNumber = scan.nextInt();
                scan.nextLine();

                System.out.print("Enter the trip date (YYYY-MM-DD): ");
                tripDate = scan.next();
                scan.nextLine();

                System.out.print("Enter the scheduled start time (HH:MM:SS): ");
                startTime = scan.nextLine();

                System.out.print("Please enter the new driver name: ");
                String newDriverName = scan.nextLine();

                sql = "UPDATE TripOffering SET DriverName = '" + newDriverName + "' " +
                        " WHERE TripNumber = " + tripNumber +
                        " AND Date = '" + tripDate + "'" +
                        " AND ScheduledStartTime = '" + startTime + "';";

                try (Statement stmt = connection.createStatement()) {
                    int rowsAffected = stmt.executeUpdate(sql);

                    if (rowsAffected > 0) {
                        System.out.println("Trip updated successfully.");
                    } else {
                        System.out.println("No trip found with that TripNumber.");
                    }

                } catch (SQLException e) {
                    System.err.println("SQL error: " + e.getMessage());
                }
                break;
            case 4:
                System.out.print("Please enter the trip number you would like to change the bus: ");
                tripNumber = scan.nextInt();
                scan.nextLine();

                System.out.print("Enter the trip date (YYYY-MM-DD): ");
                tripDate = scan.next();
                scan.nextLine();

                System.out.print("Enter the scheduled start time (HH:MM:SS): ");
                startTime = scan.nextLine();

                System.out.print("Please enter the new Bus's ID: ");
                String newBusId = scan.nextLine();

                sql = "UPDATE TripOffering SET BusID = '" + newBusId + "' " +
                        " WHERE TripNumber = " + tripNumber +
                        " AND Date = '" + tripDate + "'" +
                        " AND ScheduledStartTime = '" + startTime + "';";

                try (Statement stmt = connection.createStatement()) {
                    int rowsAffected = stmt.executeUpdate(sql);

                    if (rowsAffected > 0) {
                        System.out.println("Trip updated successfully.");
                    } else {
                        System.out.println("No trip found with that TripNumber.");
                    }

                } catch (SQLException e) {
                    System.err.println("SQL error: " + e.getMessage());
                }
                break;
        }
    }

    public static void numberThree(Scanner scan) {
        System.out.println("Please enter the Trip Number:");
        int tripNumber = scan.nextInt();

        String sql = "SELECT Stop.StopNumber, " +
                "       Stop.StopAddress, " +
                "       TripStopInfo.SequenceNumber, " +
                "       TripStopInfo.DrivingTime " +
                "FROM TripStopInfo " +
                "JOIN Stop ON TripStopInfo.StopNumber = Stop.StopNumber " +
                "WHERE TripStopInfo.TripNumber = " + tripNumber + ";";

        try {
            resultSet = executeStatement(sql);

            System.out.println("Stops for Trip Number " + tripNumber + ":");
            while (resultSet.next()) {
                String sequenceNumber = resultSet.getString("SequenceNumber");
                String stopNumber = resultSet.getString("StopNumber");
                String stopAddress = resultSet.getString("StopAddress");
                String drivingTime = resultSet.getString("DrivingTime");

                System.out.println("Stop Number: " + stopNumber + ", Stop Address: " + stopAddress +
                        ", Sequence Number: " + sequenceNumber + ", Driving Time: " + drivingTime);
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberFour(Scanner scan) {
        System.out.println("Please enter the Driver Name:");
        String driverName = scan.nextLine();

        System.out.print("Enter the trip date (YYYY-MM-DD): ");
        String tripDate = scan.next();
        scan.nextLine();

        String sql = "SELECT TF.BusID, " +
                "TF.ScheduledStartTime, " +
                "T.StartLocationName, " +
                "T.DestinationName " +
                "FROM TripOffering TF " +
                "JOIN Trip T ON T.TripNumber = TF.TripNumber " +
                "WHERE TF.DriverName = '" + driverName + "' " +
                "AND TF.Date = '" + tripDate + "';";

        try {
            resultSet = executeStatement(sql);

            System.out.println("Schedule for driver: " + driverName);
            while (resultSet.next()) {
                System.out.println("" + resultSet.getString("BusID") + " " + resultSet.getString("ScheduledStartTime")
                        + " "
                        + resultSet.getString("StartLocationName") + " " + resultSet.getString("DestinationName"));

            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberFive(Scanner scan) {
        System.out.println("Please enter the Driver Name:");
        String driverName = scan.nextLine();

        System.out.println("Please enter the driver's phone number:");
        String driverNumber = scan.nextLine();

        String sql = "INSERT INTO Driver (DriverName, DriverTelephoneNumber) VALUES ('" +
                driverName + "', '" +
                driverNumber + "');";

        try (Statement stmt = connection.createStatement()) {
            int rowsAffected = stmt.executeUpdate(sql);

            if (rowsAffected > 0) {
                System.out.println("Driver inserted successfully.");
            } else {
                System.out.println("Driver was not inserted sucessfully.");
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberSix(Scanner scan) {
        System.out.println("Please enter the bus ID:");
        int busID = scan.nextInt();

        System.out.println("Please enter the bus model:");
        String busModel = scan.next();
        scan.nextLine();

        System.out.print("Enter the year of the bus: ");
        int busYear = scan.nextInt();
        scan.nextLine();

        String sql = "INSERT INTO Bus (BusID, Model, Year) VALUES (" +
                busID + ", '" +
                busModel + "', " +
                busYear + ")";

        try (Statement stmt = connection.createStatement()) {
            int rowsAffected = stmt.executeUpdate(sql);

            if (rowsAffected > 0) {
                System.out.println("Bus inserted successfully.");
            } else {
                System.out.println("Bus was not inserted sucessfully.");
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberSeven(Scanner scan) {
        System.out.println("Please enter the bus ID for you bus you want to delete:");
        int busID = scan.nextInt();

        String sql = "DELETE FROM Bus WHERE BusID = " + busID + ";";

        try (Statement stmt = connection.createStatement()) {
            int rowsAffected = stmt.executeUpdate(sql);

            if (rowsAffected > 0) {
                System.out.println("Bus deleted successfully.");
            } else {
                System.out.println("Bus does not exist.");
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    public static void numberEight(Scanner scan) {
        System.out.println("Record the actual stop info.");
        System.out.println("Please enter the Trip Number:");
        int tripNumber = scan.nextInt();
        scan.nextLine();

        System.out.print("Enter the trip date (YYYY-MM-DD): ");
        String tripDate = scan.nextLine();

        System.out.println("Please enter the stop number:");
        int stopNumber = scan.nextInt();
        scan.nextLine();

        System.out.print("Enter the scheduled start time (HH:MM:SS): ");
        String startTime = scan.nextLine();

        System.out.print("Enter the scheduled arrival time (HH:MM:SS): ");
        String arrivalTime = scan.nextLine();

        System.out.print("Enter the actual start time (HH:MM:SS): ");
        String actualStartTime = scan.nextLine();

        System.out.print("Enter the actual arrival time (HH:MM:SS): ");
        String actualArrivalTime = scan.nextLine();

        System.out.println("Please enter the number of passengers in:");
        int passengersIn = scan.nextInt();

        System.out.println("Please enter the number of passengers out:");
        int passengersOut = scan.nextInt();

        String sql = "INSERT INTO ActualTripStopInfo " +
                "(TripNumber, Date, ScheduledStartTime, StopNumber," +
                "ScheduledArrivalTime, ActualStartTime, ActualArrivalTime, NumberOfPassengerIn," +
                "NumberOfPassengerOut) VALUES (" +
                tripNumber + ", '" + tripDate + "', '" + startTime + "', " + stopNumber + ", '" +
                arrivalTime + "', '" + actualStartTime + "', '" + actualArrivalTime + "', " +
                passengersIn + ", " + passengersOut + ")";

        try (Statement stmt = connection.createStatement()) {
            int rowsAffected = stmt.executeUpdate(sql);

            if (rowsAffected > 0) {
                System.out.println("Actual Stop info inserted sucessfully.");
            } else {
                System.out.println("Actual Stop info not inserted successfully.");
            }

        } catch (SQLException e) {
            System.err.println("SQL error: " + e.getMessage());
        }
    }

    private static int promptInt(Scanner scan, String prompt) {
        while (true) {
            System.out.print(prompt);
            String line = scan.nextLine().trim();
            try {
                return Integer.parseInt(line);
            } catch (NumberFormatException e) {
                System.out.println("→ Please enter a valid integer.");
            }
        }
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

            boolean wantsToContinue = true;
            while (wantsToContinue) {

                System.out.println("1. View all trips based on a Source, Destination, and Date");
                System.out.println("2. Edit a schedule of a Trip Offering");
                System.out.println("3. Display the stops for a given Trip");
                System.out.println("4. Display the weekly shcedule of a given driver and date.");
                System.out.println("5. Add a driver to the system");
                System.out.println("6. Add a bus to the system");
                System.out.println("7. Delete a bus from the system");
                System.out.println("8. Record actual data for a given trip offering");
                System.out.println("9. Exit");

                int choice = promptInt(scan, "Enter your choice: ");

                switch (choice) {
                    case 1:
                        numberOne(scan);
                        break;
                    case 2:
                        numberTwo(scan);
                        break;
                    case 3:
                        numberThree(scan);
                        break;
                    case 4:
                        numberFour(scan);
                        break;
                    case 5:
                        numberFive(scan);
                        break;
                    case 6:
                        numberSix(scan);
                        break;
                    case 7:
                        numberSeven(scan);
                        break;
                    case 8:
                        numberEight(scan);
                        break;
                    case 9:
                        wantsToContinue = false;
                        break;
                    default:
                        System.out.println("Invalid choice. Please pick 1–9.");

                }
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