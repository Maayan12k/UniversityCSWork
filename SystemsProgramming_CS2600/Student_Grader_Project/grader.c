#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

typedef struct
{
    char name[10];
    int score;
    char grade;
} Student;

Student *students;
int sizeOfStructArray = 0;

char *calculateGrade(int score)
{
    if (score >= 90)
    {
        return "A";
    }
    else if (score >= 80)
    {
        return "B";
    }
    else if (score >= 70)
    {
        return "C";
    }
    else if (score >= 58)
    {
        return "D";
    }
    else
    {
        return "F";
    }
}
void sortStruct()
{
    Student temp;
    for (int i = 0; i < sizeOfStructArray; i++)
    {
        for (int j = i + 1; j < sizeOfStructArray; j++)
        {
            if (students[i].score < students[j].score)
            {
                temp = students[i];
                students[i] = students[j];
                students[j] = temp;
            }
        }
    }
}

void printResults()
{
    FILE *file = fopen("student_grades", "w");
    for (int i = 0; i < sizeOfStructArray; i++)
    {
        fprintf(file, "%s\t%d\t%c\n", students[i].name, students[i].score, students[i].grade);
    }
    fclose(file);
}

int main()
{
    char *path = "/Users/maayan/Desktop/School/Spring2024/CS2600/Project1/studentsWork/";
    char *pathOfGraderFile = "/Users/maayan/Desktop/School/Spring2024/CS2600/Project1/answer_code.c";
    DIR *dir = opendir(path); // errno = 2 if path is invalid

    if (errno == 2)
    {
        printf("Error! Invalid path! %s\n", strerror(errno));
        return 1;
    }

    struct dirent *file_entry;
    char fileName[20];
    char pathName[100];
    char charOfStudentFile;
    char charOfGraderFile;

    while ((file_entry = readdir(dir)) != NULL)
    {
        if (file_entry->d_type == DT_REG)
        {
            sizeOfStructArray += 1;
        }
    }
    rewinddir(dir);
    students = (Student *)malloc(sizeOfStructArray * sizeof(Student));

    int index = 0;
    while ((file_entry = readdir(dir)) != NULL)
    {
        if (file_entry->d_type == DT_REG)
        {
            strcpy(fileName, file_entry->d_name);
            strcpy(pathName, path);
            strcat(pathName, fileName);
            FILE *fileOfStudentFile = fopen(pathName, "r");
            FILE *fileOfGraderFile = fopen(pathOfGraderFile, "r");

            int score = 100;
            int numOfDifferencesInLine = 0;

            while (1)
            {
                charOfStudentFile = fgetc(fileOfStudentFile);
                charOfGraderFile = fgetc(fileOfGraderFile);

                if (charOfStudentFile == EOF && charOfGraderFile != EOF)
                {
                    score -= 1;
                    break;
                }
                else if (charOfStudentFile != EOF && charOfGraderFile == EOF)
                {
                    score -= 1;
                    break;
                }
                else if (charOfStudentFile == EOF && charOfGraderFile == EOF)
                {
                    if (numOfDifferencesInLine > 0)
                    {
                        score -= 1;
                    }
                    break;
                }

                if (charOfStudentFile != charOfGraderFile)
                {
                    numOfDifferencesInLine += 1;
                }

                if (charOfStudentFile == '\n' && charOfGraderFile != '\n')
                {
                    while (charOfGraderFile != '\n')
                    {
                        charOfGraderFile = fgetc(fileOfGraderFile);
                    }

                    score -= 1;
                    numOfDifferencesInLine = 0;
                }
                else if (charOfStudentFile != '\n' && charOfGraderFile == '\n')
                {
                    while (charOfStudentFile != '\n')
                    {
                        charOfStudentFile = fgetc(fileOfStudentFile);
                    }

                    score -= 1;
                    numOfDifferencesInLine = 0;
                }
                else if (charOfStudentFile == '\n' && charOfGraderFile == '\n')
                {
                    if (numOfDifferencesInLine > 0)
                    {
                        score -= 1;
                    }
                    numOfDifferencesInLine = 0;
                }
            }
            fclose(fileOfStudentFile);
            fclose(fileOfGraderFile);

            students[index].score = score;
            char str[10] = "student";
            char str2[8];
            sprintf(str2, "%d", index + 1);
            strcat(str, str2);
            strcpy(students[index].name, str);
            students[index].grade = *calculateGrade(score);

            index++;
        }
    }
    closedir(dir);
    sortStruct();
    printResults();
    return 0;
}
