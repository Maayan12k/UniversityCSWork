; who: Maayan Israel
; what: Library for some assembly functions used in CSCI 150
; why: because I am an engineer!
; when: Spring 2023

section .text

; This is version 0 of the library passes arguments via registers

global atoi
global sum_array
global print_char
global rev_str
global get_input
global print_str
global errno
global current_time
global seed_random
global gen_random
global print_uint

print_uint:
; Description: prints an unsigned integer
; Receives:     EAX = uint value
; Requires:     Character buffer (char_buff)

    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up the stack frame

    push    char_buff       ; Preserve char_buff
    push    DWORD [ebp + 8] ; Push the uint value
    call    itoa            ; Call the itoa function to convert int to string

    leave                   ; Clean up the stack frame
    ret                     ; Return

seed_random:
; Description: Gets input for the random number generator
; Receives:    EAX = input value to furnish 'next'
; Returns:     Nothing

    mov     [next], eax     ; Furnish the 'next' variable with the input value
    ret

gen_random:
; Description: Generates a random number 
; Receives:    Nothing
; Returns:     EAX = random number

    mov     eax, [next]     ; Load 'next' value
    mov     ecx, const1
    mul     ecx             ; next * 1103515245 

    add     eax, const2     ; next + 12345
    mov     [next], eax     ; Store the updated 'next' value
    xor     edx, edx        ; Clear edx

    mov     ecx, const3
    div     ecx             ; next / 65536

    mov     ecx, const4
    xor     edx, edx
    div     ecx             ; next % 32768
    mov     eax, edx        ; Return eax as the random value

    ret

current_time:
; Description: Gets the number of seconds since 1/1/1970
; Receives:    Nothing
; Returns:     EAX = number of seconds

    push        ebx         ; Preserve ebx
    mov         eax, 13      ; System call number for time
    xor         ebx, ebx    ; Clear ebx
    int         80h         ; Invoke the system call

    pop         ebx         ; Restore ebx
    ret

atoi: 
; Description: Converts a null-terminated string of digits to an integer
; Receives:    EAX = address of the string
; Returns:     EAX = the integer value
; Algorithm:   Horner's method

.factor:    equ 10
.offset:    equ 48

    push        esi         ; Preserve esi
    push        ebx         ; Preserve ebx
    mov         ESI, EAX    ; ESI = source (address of the string)
    xor         eax, eax    ; Clear eax (accumulator)
    xor         ecx, ecx    ; Clear ecx (used for character code)
    mov         ebx, .factor ; Load the factor (10) into ebx

.while:
    mov         cl, [esi]   ; Load the first character from the string
    test        cl, cl      ; Check if it's a null terminator (end of string)
    jz          .wend       ; If yes, exit the loop
    sub         cl, .offset ; Calculate the integer value from ASCII
    mul         ebx         ; Multiply eax by 10
    add         eax, ecx    ; Add the next character's integer value to eax
    inc         esi         ; Calculate the effective address of the next character
    jmp         .loop

.wend:
    pop         esi         ; Restore esi
    ret

sum_array:
; Description: Sums the values of a double word array.
; Receives:    EAX = Address of the array
;              EBX = Number of elements
; Returns:     EAX = sum
; Requirement: Nothing

    push        esi         ; Preserve esi
    mov         esi, eax    ; ESI = array source address
    xor         eax, eax    ; Clear eax (accumulator) to store the sum
    mov         ecx, ebx    ; ECX = counter (number of elements)

.loop:
    add         eax, [esi]  ; Add the value at the current array element to the sum
    add         esi, 4      ; Move to the next element (4 bytes ahead)
    loop        .loop       ; Repeat the loop for all elements

    pop         esi         ; Restore esi
    ret

print_char:
; Description: Prints a single character.
; Receives:    EAX = char value
; Requirement: Nothing
; Requirement: char_buff (a buffer provided below)

    mov         [char_buff], al ; Move the character value to the buffer
    mov         eax, 4          ; System call number for write
    mov         ebx, 1          ; File descriptor (stdout)
    mov         ecx, char_buff ; Pointer to the character buffer
    mov         edx, 1          ; Number of bytes to write
    int          80h             ; Invoke the system call to write

    ret

rev_str:
; Description: Reverses a string.
; Receives:    EAX = Address of the string
;              EBX = Size in bytes of the string
; Returns:     EAX = Address of the reversed array

push        esi         ; Preserve esi
push        edi         ; Preserve edi

test        ebx, ebx    ; Check if the size is zero
jnz         .start      ; If not zero, start reversing

mov         dword [errno], -1 ; Set errno to -1 (error) for zero-size string
jmp         .exit       ; Jump to the exit point

.start:
mov         ecx, ebx    ; Set counter to the size of the string
mov         esi, eax    ; ESI = SOURCE
mov         edi, eax    ; EDI = DESTINATION

.pushloop:              ; Push loop
movzx       edx, byte [esi] ; Move a character into edx
push        edx         ; Push the character onto the stack
inc         esi         ; Calculate the effective address of the next character
loop        .pushloop   ; Repeat the loop for all characters

mov         ecx, ebx    ; Reset the counter

.poploop:               ; Pop loop
pop         edx         ; Pop a character from the stack into edx
mov         [edi], dl   ; Store the character in the destination buffer
inc         edi         ; Calculate the effective address of the next character
loop        .poploop    ; Repeat the loop for all characters

.exit:
pop         edi         ; Restore edi
pop         esi         ; Restore esi
ret

get_input:
; Returns:    EAX = inputted string
; Receives:   EAX = Prompt address
;             EBX = Prompt size
;             ECX = Input buffer address
;             EDX = Input buffer size

; Get a string from the user

    push        ebx         ; Preserve ebx
    push        edi         ; Preserve edi
    push        ecx         ; Preserve ecx
    push        edx         ; Preserve edx
    mov         edi, ecx    ; Move the buffer address into edi

    call        print_str   ; Print the prompt

    mov         eax, 3      ; Read operation
    mov         ebx, 0      ; Read from stdin
    pop         edx         ; EDX = input buffer size
    pop         ecx         ; ECX = input buffer 

    int         80h         ; Invoke the system call to read

    dec         eax         ; Remove the newline character
    mov         BYTE [edi + eax], 0 ; Null-terminate the string

    pop         edi         ; Restore edi
    pop         ebx         ; Restore ebx

    ret

print_str:
; Receives:   EAX = Address of the string
;             EBX = Size of the string
; Requires:   Nothing
; Requires:   A buffer provided below

    push        EBX         ; Preserve ebx
    mov         ecx, EAX    ; Move the address into ecx
    mov         edx, EBX    ; Move the size into edx
    mov         eax, 4      ; System call number for write
    mov         ebx, 1      ; File descriptor (stdout)
    int         80h         ; Invoke the system call to write

    ; EAX holds the quantity of characters input
    dec         eax         ; Remove the newline character
    mov         byte [edi + eax], 0 ; Null-terminate the string

    pop         ebx         ; Restore ebx
    ret

swap:
; Description: Swaps the values of two memory locations.
; Receives:    EAX = Address of the first value
;              EBX = Address of the second value
; Requirement: Nothing

    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up the stack frame

    push        esi         ; Preserve esi
    push        edi         ; Preserve edi
    mov         esi, [ebp + 8] ; ESI holds the first address
    mov         edi, [ebp + 12] ; EDI holds the second address
    mov         eax, [esi]  ; EAX -> temp = value at the first address
    xchg        eax, [ebx]  ; Swap EAX with value at the second address
    mov         [esi], eax  ; Store temp at the first address

    pop         edi         ; Restore edi
    pop         esi         ; Restore esi

    mov         esp, ebp    ; Clean up the stack frame
    pop         ebp         ; Restore the caller's base pointer
    ret

itoa:
; Description: Converts an integer to a string.
; Receives:    EAX = Integer value
;              EBX = String buffer address (assume it's big enough)
; Returns:     Nothing
; Requires:    Nothing

    push        ebp         ; Preserve the caller's base pointer
    mov         ebp, esp    ; Set up the stack frame
    push        edi         ; Preserve edi
    push        ebx         ; Preserve ebx

    mov         ebx, 10     ; Load the factor (10) into ebx
    mov         eax, [ebp + 8] ; Load the integer value into eax
    mov         edi, [ebp + 12] ; Load the address of the string buffer into edi

.do:
    xor         edx, edx    ; Clear edx (used for remainder)
    div         ebx         ; Divide eax by 10, result in eax, remainder in edx
    add         dl, 48      ; Convert the remainder to ASCII
    mov         [edi], dl   ; Store the ASCII character in the buffer
    inc         edi         ; Calculate the effective address for the next character

    test        eax, eax    ; Check if eax is zero
    jnz         .do         ; If not zero, continue the loop

.wend:
    ; Repeatedly divide by 10 and add 48 to get the ASCII characters
    mov         byte [edi], 0 ; Null-terminate the string

    pop         ebx         ; Restore ebx
    pop         edi         ; Restore edi
    leave                   ; Clean up the stack frame
    ret

section .bss 
    char_buff_sz:   equ 255
    char_buff:      resb char_buff_sz

    input_sz: equ 255
    input: resb input_sz

section     .data
    errno: dd 0

    next:   dd    1
    const1: equ 1103515245
    const2: equ 12345
    const3: equ 65536
    const4: equ 32768
