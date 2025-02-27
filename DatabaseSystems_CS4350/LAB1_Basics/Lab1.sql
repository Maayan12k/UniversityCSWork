CREATE TABLE Course (
    CourseN INT PRIMARY KEY,
    CourseName VARCHAR(100),
    Nunit INT
);

CREATE TABLE Teacher (
    CourseN INT,
    Quarter VARCHAR(20),
    TeacherName VARCHAR(100),
    PRIMARY KEY (CourseN, Quarter),
    FOREIGN KEY (CourseN) REFERENCES Course(CourseN)
);

CREATE TABLE LocationNTime (
    CourseN INT,
    Quarter VARCHAR(20),
    DayTime VARCHAR(20),
    RoomN INT,
    PRIMARY KEY (CourseN, Quarter, DayTime),
    FOREIGN KEY (CourseN) REFERENCES Course(CourseN)
);

CREATE TABLE Student (
    StudentName VARCHAR(100),
    CourseN INT,
    Quarter VARCHAR(20),
    PRIMARY KEY (StudentName, CourseN, Quarter),
    FOREIGN KEY (CourseN) REFERENCES Course(CourseN)
);

INSERT INTO Course (CourseN, CourseName, Nunit) VALUES 
(101, 'Intro To CS', 4),
(102, 'Intro To Data', 4),
(201, 'Algorithms', 4),
(202, 'Machine Learning', 5),
(203, 'Artificial Intelligence', 4),
(204, 'Cybersecurity', 3),
(205, 'Software Engineering', 5),
(206, 'Quantum Computing', 3);

INSERT INTO Teacher (CourseN, Quarter, TeacherName) VALUES 
(101, 'Winter2011', 'Dr. Smith'),
(102, 'Winter2011', 'Prof. Johnson'),
(201, 'Winter2011', 'Karen Reed'),
(202, 'Spring2005', 'Karen Reed'),
(203, 'Fall2023', 'Karen Reed'),
(204, 'Fall2023', 'Dr. Miller'),
(201, 'Spring2025', 'Prof. Davis'),
(202, 'Spring2025', 'Dr. Brown'),
(203, 'Winter2025', 'Dr. Wilson'),
(204, 'Winter2025', 'Dr. Taylor'),
(205, 'Fall2025', 'Karen Reed'),
(205, 'Winter2025', 'Karen Reed');

INSERT INTO LocationNTime (CourseN, Quarter, DayTime, RoomN) VALUES 
(101, 'Winter2011', 'M2:00AM', 34),
(102, 'Winter2011', 'W4:50PM', 34),
(201, 'Winter2011', 'M6:00PM', 723),
(202, 'Spring2005', 'T8:00PM', 713),
(203, 'Fall2023', 'M3:00PM', 101),
(204, 'Fall2023', 'W10:00AM', 202),
(201, 'Spring2025', 'M10:00AM', 34),
(202, 'Spring2025', 'T5:00PM', 723),
(203, 'Winter2025', 'W7:00PM', 303),
(204, 'Winter2025', 'M2:30PM', 713),
(205, 'Fall2025', 'T3:30PM', 505),
(205, 'Winter2025', 'M4:00PM', 505),
(201, 'Winter2011', 'F8:00AM', 721),  
(204, 'Winter2025', 'T1:00PM', 713),  
(205, 'Winter2025', 'W9:00AM', 505);

INSERT INTO Student (StudentName, CourseN, Quarter) VALUES 
('Ron Smith', 101, 'Fall2024'),
('Ron Smith', 202, 'Spring2025'),
('David Weidman', 102, 'Winter2011'),
('David Weidman', 201, 'Winter2011'),
('Alice Johnson', 101, 'Fall2024'),
('Bob Williams', 102, 'Fall2024'),
('Charlie Brown', 102, 'Winter2025'),
('David Smith', 102, 'Winter2025'),
('Emily Davis', 205, 'Spring2025'),
('Ron Smith', 101, 'Spring2025'),
('David Weidman', 102, 'Spring2025'),
('Ron Smith', 202, 'Winter2025'),
('David Weidman', 201, 'Winter2025');

#1. List the name of every teacher (distinct names) who teaches in RoomN ‘34’ in Winter2011
SELECT DISTINCT Teacher.TeacherName 
FROM Teacher
JOIN LocationNTime 
	ON Teacher.CourseN = LocationNTime.CourseN
	AND Teacher.Quarter = LocationNTime.Quarter
WHERE LocationNTime.RoomN = 34 AND LocationNTime.Quarter = 'WINTER2011';

#2. List CourseN, CourseName, and TeacherName of every course meets on Monday PM
SELECT Course.CourseN, Course.CourseName, Teacher.TeacherName 
FROM Teacher 
JOIN Course 
	ON Course.CourseN = Teacher.CourseN 
JOIN LocationNTime
	ON Course.CourseN = LocationNTime.CourseN
WHERE LocationNTime.DayTime LIKE 'M%PM';

#3. List the name of every teacher who taught at least one course in RroomN ‘723.’
SELECT DISTINCT Teacher.TeacherName
FROM Teacher
JOIN LocationNTime ON Teacher.CourseN = LocationNTime.CourseN 
WHERE LocationNTime.RoomN = '723';

#4. List the CourseN, Quarter, RoomN and DayTime of every course taught by ‘Karen Reed’ in the Spring 2005
SELECT LocationNTime.*
FROM LocationNTime
JOIN Teacher 
	ON Teacher.CourseN = LocationNTime.CourseN 
WHERE Teacher.TeacherName = 'Karen Reed' AND LocationNTime.Quarter = 'Spring2005';

#5. List the CourseN and TeacherName of every course taken by the student ‘Ron Smith’ or by the student ‘David Weidman.’
SELECT Teacher.TeacherName, Teacher.CourseN
FROM Teacher
JOIN Student ON Teacher.CourseN = Student.CourseN
WHERE Student.StudentName = 'Ron Smith' OR Student.StudentName = 'David Weidman';

#6. List the CourseN and Quarter of every course taught by ‘Karen Reed’ and met or meets in RoomN ‘713’.
SELECT Teacher.CourseN, Teacher.Quarter
FROM Teacher
JOIN LocationNTime ON LocationNTime.CourseN = Teacher.CourseN
WHERE Teacher.TeacherName = 'Karen Reed' AND LocationNTime.RoomN = 713;


#7. List the name of every teacher who has taught the same course at least two times.
SELECT Teacher.TeacherName
FROM Teacher
GROUP BY Teacher.TeacherName, Teacher.CourseN
HAVING COUNT(Teacher.Quarter) > 1;

#8. List the name of every teacher( distinct names) who has taught at least two different courses in the same or different quarters.
SELECT Teacher.TeacherName
FROM Teacher
GROUP BY Teacher.TeacherName
HAVING COUNT(DISTINCT Teacher.CourseN) > 1;

#9. List the CourseN, CourseName, and Quarter which meets or met at least two times a week
SELECT Course.CourseName, LocationNTime.CourseN, LocationNTime.Quarter
FROM LocationNTime
JOIN Course ON LocationNTime.CourseN = Course.CourseN
GROUP BY LocationNTime.CourseN, LocationNTime.Quarter
HAVING COUNT(DISTINCT LocationNTime.DayTime) > 1;


#10. List the CourseN and CourseName of every course with number of units > 4
SELECT C.CourseN, C.CourseName
FROM Course C
WHERE C.Nunit > 4;


#11. List every course number and student’s name who has taken the course at least twice.
SELECT S.StudentName, S.CourseN
FROM Student S
GROUP BY S.StudentName, S.CourseN
HAVING COUNT(S.Quarter) > 1;

#12 Use ‘*’ to list the CourseN, CourseName, Nunit, Quarter, TeacherName of every course sorted by CourseN ascending, CourseName descending
SELECT * 
FROM Teacher
JOIN Course ON Course.CourseN = Teacher.CourseN
ORDER BY Course.CourseN ASC, Course.CourseName DESC;


#13 List the CourseN and Quarter of every course taught by two different instructors in the same quarter ordered by the CourseN in descending order.
SELECT T.CourseN, T.Quarter
FROM Teacher T
JOIN LocationNTime 
	ON T.CourseN = LocationNTime.CourseN AND
    T.Quarter = LocationNTime.Quarter
GROUP BY T.CourseN, T.Quarter
HAVING COUNT(DISTINCT T.TeacherName) = 2
ORDER BY T.CourseN DESC;