; Define a structure 'student' with three fields: 'id', 'name', and 'major'
struc student
    .id:    resd 1      ; Reserve 4 bytes for 'id' field (assuming it's an unsigned integer)
    .name:  resb 150    ; Reserve 150 bytes for 'name' field (assuming it's a character string)
    .major: resb 100    ; Reserve 100 bytes for 'major' field (assuming it's a character string)
endstruc