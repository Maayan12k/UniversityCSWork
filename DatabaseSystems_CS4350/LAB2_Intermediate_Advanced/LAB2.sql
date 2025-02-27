CREATE TABLE Student (
    SSN CHAR(9) PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Major VARCHAR(50) NOT NULL
);

CREATE TABLE Course (
    CourseNumber VARCHAR(10) PRIMARY KEY,
    PrerequisiteCourseNumber VARCHAR(10),
    CourseTitle VARCHAR(100) NOT NULL,
    NumberUnits INT CHECK (NumberUnits > 0),
    FOREIGN KEY (PrerequisiteCourseNumber) REFERENCES Course(CourseNumber) ON DELETE SET NULL
);

CREATE TABLE Section (
    CourseNumber VARCHAR(10),
    Quarter VARCHAR(10),
    RoomNumber VARCHAR(10) NOT NULL,
    DayTime VARCHAR(20) NOT NULL,
    PRIMARY KEY (CourseNumber, Quarter),
    FOREIGN KEY (CourseNumber) REFERENCES Course(CourseNumber) ON DELETE CASCADE
);

CREATE TABLE Enrollment (
    SSN CHAR(9),
    CourseNumber VARCHAR(10),
    Quarter VARCHAR(10),
    Grade CHAR(2) CHECK (Grade IS NULL OR Grade IN ('A', 'B', 'C', 'D', 'F')),
    PRIMARY KEY (SSN, CourseNumber, Quarter),
    FOREIGN KEY (SSN) REFERENCES Student(SSN) ON DELETE CASCADE
);

INSERT INTO Student (SSN, Name, Major) VALUES
('123456789', 'Alice Johnson', 'Computer Science'),
('987654321', 'Bob Smith', 'Mathematics'),
('555666777', 'Charlie Brown', 'Physics'),
('111222333', 'David Lee', 'Engineering'),
('444555666', 'Eve Adams', 'Biology');

INSERT INTO Course (CourseNumber, PrerequisiteCourseNumber, CourseTitle, NumberUnits) VALUES
('CS101', NULL, 'Intro to Computer Science', 4),
('CS102', 'CS101', 'Data Structures', 4),
('MATH101', NULL, 'Calculus I', 3),
('MATH102', 'MATH101', 'Calculus II', 3),
('PHYS101', NULL, 'Physics I', 4),
('PHYS102', 'PHYS101', 'Physics II', 4),
('ENG101', NULL, 'Intro to Engineering', 3);

INSERT INTO Section (CourseNumber, Quarter, RoomNumber, DayTime) VALUES
('CS101', 'Fall2024', 'R101', 'MW 1:00-2:00PM'),
('CS102', 'Winter2025', 'R102', 'TTh 10:00-11:30AM'),
('MATH101', 'Fall2024', 'R201', 'MW 3:00-4:00PM'),
('MATH101', 'Winter2025', 'R201', 'MW 3:00-4:00PM'),
('MATH102', 'Spring2025', 'R202', 'TTh 1:00-2:30PM'),
('MATH102', 'Spring2024', 'R202', 'TTh 1:00-2:30PM'),
('PHYS101', 'Fall2024', 'R301', 'MW 1:00-2:00PM'),
('PHYS102', 'Winter2025', 'R302', 'TTh 11:00-12:30PM'),
('ENG101', 'Spring2025', 'R401', 'MW 2:00-3:30PM');

INSERT INTO Enrollment (SSN, CourseNumber, Quarter, Grade) VALUES
('123456789', 'CS101', 'Fall2024', 'A'),
('123456789', 'CS102', 'Winter2025', 'B'),
('987654321', 'MATH101', 'Fall2024', 'A'),
('987654321', 'MATH102', 'Spring2024', NULL),
('987654321', 'MATH102', 'Spring2025', 'C'),
('555666777', 'PHYS101', 'Fall2024', 'B'),
('555666777', 'PHYS102', 'Winter2025', 'A'),
('111222333', 'ENG101', 'Spring2025', 'B'),
('444555666', 'CS101', 'Fall2024', 'A'),
('444555666', 'MATH101', 'Fall2024', 'A'),
('444555666', 'MATH101', 'Winter2025', 'B'),
('444555666', 'MATH102', 'Spring2025', 'C');

#For quick access
SELECT * FROM Student;
SELECT * FROM Course;
SELECT * FROM Section;
SELECT * FROM Enrollment;


#1) List the name, SSN and the number of courses the student has taken (courses with 
# the same CourseNumber taken in different quarters are counted as different courses)
SELECT S.SSN, S.Name, COUNT(DISTINCT E.CourseNumber, E.Quarter) as numOfCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
GROUP BY S.SSN, S.Name;

#2 Answer #1 assuming courses with the same CourseNumber taken in different quarters are considered as one course
SELECT S.SSN, S.Name, COUNT(E.CourseNumber) as numOfCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
GROUP BY S.SSN, S.Name;

#3 List the name, SSN and number of courses the student has taken and completed
# , i.e. received a letter grade (courses with the same CourseNumber taken in
# different quarters are counted as different courses)
SELECT S.SSN, S.Name, COUNT(DISTINCT E.CourseNumber, E.Quarter) as numOfCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
WHERE E.Grade IS NOT NULL
GROUP BY S.SSN, S.Name;

#4 List the name, SSN and number of courses the student has taken and completed
# with a letter grade C or better (courses with the same CourseNumber taken in
# different quarters are counted as different courses).
SELECT S.SSN, S.Name, COUNT(DISTINCT E.CourseNumber, E.Quarter) as numOfCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
WHERE E.Grade IN ('A', 'B', 'C')
GROUP BY S.SSN, S.Name;

#5 List the Course Title and CourseNumber which does not have any prerequisite.
SELECT C.CourseTitle, C.CourseNumber
FROM Course C
WHERE C.PrerequisiteCourseNumber IS NULL;

#6 List the name of every student and SSN who earned an A in all courses he or she
# has completed, i.e. every grade is either Null or an A)
SELECT S.SSN, S.Name
FROM Student S
WHERE NOT EXISTS (
    SELECT 1
    FROM Enrollment E
    WHERE E.SSN = S.SSN
    AND E.Grade IS NOT NULL
    AND E.Grade != 'A'
);

#7 List the name of every student, SSN and the CourseNumber who has taken the
# course at least three times
SELECT S.Name, S.SSN, E.CourseNumber
FROM Student S
JOIN Enrollment E ON E.SSN = S.SSN
GROUP BY S.Name, S.SSN, E.CourseNumber
HAVING COUNT(DISTINCT E.Quarter) > 2;

#8 List the name of every student, SSN, every quarter, and the number of courses he
# or she has taken in the listed quarter
SELECT S.Name, S.SSN, E.Quarter, COUNT(E.CourseNumber) as numCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
GROUP BY S.Name, S.SSN, E.Quarter;

#9 List the name of every student and SSN who has not taken any course more than once.
SELECT S.Name, S.SSN
FROM Student S
WHERE S.SSN NOT IN (
    SELECT E.SSN
    FROM Enrollment E
    GROUP BY E.SSN, E.CourseNumber
    HAVING COUNT(E.CourseNumber) > 1
);

#10 List the name, SSN , the number of courses the student has taken, and the number
#of courses completed(courses with the same CourseNumber taken in different
#quarters are counted as different courses).
SELECT S.SSN, S.Name, COUNT(DISTINCT E.CourseNumber, E.Quarter) as numOfCourses
FROM Student S
JOIN Enrollment E ON S.SSN = E.SSN
WHERE E.Grade IS NOT NULL
GROUP BY S.SSN, S.Name;


#11. List every two CourseNumber and their titles which have the same prerequisites.
SELECT C1.CourseNumber, C1.CourseTitle, C2.CourseNumber, C2.CourseTitle
FROM Course C1
JOIN Course C2 ON C1.PrerequisiteCourseNumber = C2.PrerequisiteCourseNumber
WHERE C1.CourseNumber < C2.CourseNumber AND C1.PrerequisiteCourseNumber IS NOT NULL;

#12 List the name of every student and SSN who has completed all the courses he/she
#has taken and earned an A in each course, i.e. every grade is an A and no NULL values
SELECT S.SSN, S.Name
FROM Student S
WHERE NOT EXISTS (
    SELECT 1
    FROM Enrollment E
    WHERE E.SSN = S.SSN
    AND (E.Grade IS NULL OR E.Grade != 'A')
);



#13. List the name of every student and SSN who earned no A in any course.
SELECT S.SSN, S.Name
FROM Student S
WHERE NOT EXISTS (
    SELECT 1
    FROM Enrollment E
    WHERE E.SSN = S.SSN
    AND E.Grade = 'A'
);


#14. List the name and major of every student who has only taken courses that meet
#MW afternoon (12 or after).
SELECT DISTINCT S.Name, S.Major
FROM Student S
WHERE NOT EXISTS (
    SELECT 1
    FROM Enrollment E
    JOIN Section SE ON E.CourseNumber = SE.CourseNumber 
                   AND E.Quarter = SE.Quarter
    WHERE E.SSN = S.SSN
    AND (SE.DayTime NOT LIKE 'MW%' OR SE.DayTime NOT LIKE '%PM')
)
AND EXISTS (
    SELECT 1
    FROM Enrollment E
    WHERE E.SSN = S.SSN
);

#15. List the name and major of every student who has taken all the courses that meet MW afternoon.
SELECT DISTINCT S.Name, S.Major
FROM Student S
WHERE NOT EXISTS (
    SELECT 1
    FROM Section SE
    WHERE SE.DayTime LIKE 'MW%' 
    AND SE.DayTime LIKE '%PM'
    AND NOT EXISTS (
        SELECT 1
        FROM Enrollment E
        WHERE E.SSN = S.SSN
        AND E.CourseNumber = SE.CourseNumber
    )
);

#16. List the name and major of every student who has not taken any course that meets MW afternoon.
SELECT DISTINCT S.Name, S.Major
FROM Student S
LEFT JOIN Enrollment E ON S.SSN = E.SSN
LEFT JOIN Section SE ON E.CourseNumber = SE.CourseNumber AND SE.Quarter = E.Quarter
GROUP BY S.Name, S.Major
HAVING COUNT(CASE WHEN SE.DayTime LIKE 'MW%' AND SE.DayTime LIKE '%PM' THEN 1 END) = 0;

#17. List every CoursaeNumber and Quarter which has the highest enrollment.
WITH EnrollmentCounts AS (
    SELECT CourseNumber, Quarter, COUNT(*) AS count
    FROM Enrollment
    GROUP BY CourseNumber, Quarter
)
SELECT CourseNumber, Quarter
FROM EnrollmentCounts
WHERE count = (SELECT MAX(count) FROM EnrollmentCounts);

#18. List every CourseNumber and CourseTitle which has the highest enrollment based on all quarters.
WITH EnrollmentCounts AS (
    SELECT CourseNumber, COUNT(*) AS count
    FROM Enrollment
    GROUP BY CourseNumber
)
SELECT EC.CourseNumber, C.CourseTitle
FROM EnrollmentCounts EC
JOIN Course C ON C.CourseNumber = EC.CourseNumber
WHERE EC.count = (SELECT MAX(count) FROM EnrollmentCounts);

#19. List the name and major of every student who has completed the highest number of units
WITH StudentUnitTotals AS (
	SELECT S.Name, S.Major, SUM(C.NumberUnits) as totalUnits
	FROM Student S
	JOIN Enrollment E ON E.SSN = S.SSN
	JOIN Course C ON E.CourseNumber = C.CourseNumber
    WHERE E.Grade IS NOT NULL
	GROUP BY S.SSN, S.Major, S.Name
)
SELECT SS.Name, SS.Major
FROM StudentUnitTotals SS
WHERE SS.totalUnits = (SELECT MAX(totalUnits) FROM StudentUnitTotals);

#20. List every Course tile which is a prerequisite for the largest number of courses.
WITH NumOfPostRequisites AS (
		SELECT C.PrerequisiteCourseNumber, COUNT(*) as NumPostReqs
        FROM Course C
        WHERE C.PrerequisiteCourseNumber IS NOT NULL
        GROUP BY C.PrerequisiteCourseNumber
)
SELECT CC.Coursetitle
FROM NumOfPostRequisites N
JOIN Course CC ON N.PrerequisiteCourseNumber = CC.CourseNumber
WHERE N.NumPostReqs = (SELECT MAX(NumPostReqs) FROM NumOfPostRequisites);