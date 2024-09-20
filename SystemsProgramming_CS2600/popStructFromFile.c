#include <stdio.h>
#include <string.h>

typedef struct Student
{
    char name[20];
    int age;
    float gpa;
} student;

int main()
{
    char ch[10];
    int age;
    float gpa;

    // prompt user for input
    printf("Enter name: ");
    scanf("%s", ch);
    printf("Enter age: ");
    scanf("%d", &age);
    printf("Enter gpa: ");
    scanf("%f", &gpa);
    // finish prompting

    // file writing
    FILE *fp = fopen("stu.txt", "w+");
    fprintf(fp, "%s %d %f", ch, age, gpa);
    fclose(fp);
    // end file writing

    // file reading
    fp = fopen("stu.txt", "r");
    student s1;
    fscanf(fp, "%s %d %f", s1.name, &s1.age, &s1.gpa);
    fclose(fp);
    printf("Name: %s\nAge: %d\nGPA: %f\n", s1.name, s1.age, s1.gpa);

    return 0;
}