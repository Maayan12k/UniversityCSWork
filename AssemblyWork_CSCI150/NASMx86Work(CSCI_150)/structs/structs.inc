; Define a structure named "student"
struc student
    .id:    resd 1        ; Define a 4-byte (double word) field for student ID
    .name:  resb 150       ; Define a character array with 150 bytes for student name
    .major: resb 100       ; Define a character array with 100 bytes for student major
endstruc