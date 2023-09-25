; who: Maayan Israel
; what: OLD Library for assembly functions used in CSCI 150
; why: because I am an engineer!
; when: Spring 2023

;This is  version 0 of my library check folder entitled "LibraryOfFunction" for completed library

section .text

; Global function declarations
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
global print_nl
global strlen
global swap

;----------------------------
; Swap:
; Description: Swaps two values at memory addresses.
; Receives: arg1: Address of the first value
;           arg2: Address of the second value
;----------------------------
swap:
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame
    push    esi             ; Preserve esi
    push    edi             ; Preserve edi

    mov     esi, [ebp + 8]  ; Load the address of the first value into esi
    mov     edi, [ebp + 12] ; Load the address of the second value into edi
    mov     eax, [esi]      ; Load the value at the first address into eax
    xchg    [ebx], edx      ; Swap the values at the two addresses
    mov     [esi], eax      ; Store the original eax value at the first address

    pop     edi             ; Restore edi
    pop     esi             ; Restore esi
    leave                   ; Clean up the stack frame
    ret                     ; Return

;----------------------------
; Strlen:
; Description: Compute the length of a null-terminated string.
; Receives: arg1: String address
;           arg2: Maximum buffer size
; Returns:  eax = Length of the string
; Requires: Nothing
;----------------------------
strlen:
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame
    
    mov     edi, [ebp + 8]  ; Load the address of the string into edi
    mov     al, 0           ; Clear al (used to store characters)
    mov     ecx, [ebp + 12] ; Load the maximum buffer size into ecx
    cld                     ; Clear the direction flag for forward string scanning

    repne   scasb           ; Scan the string until null-terminator or ecx reaches 0
    cmp     ecx, 0          ; Check if ecx reached 0
    jz      default         ; If yes, jump to default (empty string)

    dec     edi             ; Adjust edi to point to the last character of the string
    mov     eax, edi        ; Calculate the length by subtracting the original address
    sub     eax, [ebp + 8]  ; of the string from edi (last character)
    jmp     done            ; Jump to the 'done' label

default:
    mov     eax, [ebp + 12] ; If the string is empty, return the maximum buffer size - 1

done:
    leave                   ; Clean up the stack frame
    ret                     ; Return

;----------------------------
; Print_nl:
; Description: Prints a new line character.
; Receives: Nothing
; Returns:  Nothing
; Requires: print_char
;----------------------------
print_nl:
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame

    push    0x0a            ; Push the newline character (0x0a) onto the stack
    call    print_char      ; Call the print_char function to print the newline character
    leave                   ; Clean up the stack frame
    ret                     ; Return

;----------------------------
; Print_uint:
; Description: Prints an unsigned integer.
; Receives: arg1 = uint value
; Requires: Character buffer (char_buff)
;----------------------------
print_uint:
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame

    push    char_buff       ; Push the address of the character buffer onto the stack
    push    DWORD [ebp + 8] ; Push the uint value onto the stack
    call    itoa            ; Call the itoa function to convert the uint to a string
    add     esp, 8          ; Adjust the stack pointer

    push    char_buff       ; Push the address of the character buffer onto the stack
    call    strlen          ; Call the strlen function to calculate the length of the string
    add     esp, 4          ; Adjust the stack pointer

    push    eax             ; Push the length of the string onto the stack
    push    char_buff       ; Push the address of the character buffer onto the stack
    call    print_str       ; Call the print_str function to print the string
    pop     ebx             ; Restore ebx

    leave                   ; Clean up the stack frame
    ret                     ; Return

;----------------------------
; Seed_random:
; Description: Sets the seed for the random number generator.
; Receives: arg1 = seed value
;----------------------------
seed_random:
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame

    mov     [next], [ebp + 8]  ; Store the provided seed value in the 'next' variable

    pop         ebp         ; Restore the base pointer
    ret                     ; Return

;----------------------------
; Gen_random:
; Description: Generates a random number.
; Receives: Nothing
; Returns:  EAX = random number
;----------------------------
gen_random:
    mov     eax, [next]     ; Load the current seed value
    mov     ecx, const1     ; Load the constant multiplier
    mul     ecx             ; Multiply seed by the constant

    add     eax, const2     ; Add another constant
    mov     [next], eax     ; Update the seed value
    xor     edx, edx        ; Clear edx

    mov     ecx, const3     ; Load another constant
    div     ecx             ; Divide by 65536

    mov     ecx, const4     ; Load one more constant
    xor     edx, edx        ; Clear edx
    div     ecx             ; Divide by 32768
    mov     eax, edx        ; Return the remainder in eax as the random number

    ret                     ; Return

;----------------------------
; Current_time:
; Description: Gets the number of seconds since 1/1/1970.
; Receives: Nothing
; Returns:  EAX = number of seconds
;----------------------------
current_time:
    push        ebx         ; Preserve ebx
    mov         eax, 13      ; System call number for time
    xor         ebx, ebx    ; Clear ebx
    int         80h         ; Invoke the system call

    pop         ebx         ; Restore ebx

    ret                     ; Return

;----------------------------
; Atoi:
; Description: Converts a null-terminated string of digits to an integer.
; Receives: arg1 = address of the string
; Returns:  EAX = the integer value
;----------------------------
atoi:
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi
    mov         ESI, [ebp+8] ; ESI = source (address of the string)
    xor         eax, eax    ; Clear eax (accumulator)
    xor         ecx, ecx    ; Clear ecx (used for character code)
    mov         ebx, .factor ; Load the factor (10) into ebx

.loop:
    mov         cl, [esi]   ; Load the first character from the string
    test        cl, cl      ; Check if it's a null terminator (end of string)
    jz          .wend       ; If yes, exit the loop
    sub         cl, .offset ; Calculate the integer value from ASCII
    mul         ebx         ; Multiply eax by 10
    add         eax, ecx    ; Add the next character's integer value to eax
    inc         esi         ; Calculate the effective address of the next character
    jmp         .loop       ; Repeat the loop

.wend:
    pop         esi         ; Restore esi
    pop         ebp         ; Restore the base pointer
    ret                     ; Return

;----------------------------
; Sum_array:
; Description: Sums the values of a double word array.
; Receives: arg1 = Address of the array
;           arg2 = Number of elements
; Returns:  EAX = sum
; Requires: Preserve ebp
;----------------------------
sum_array:
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi

    mov         esi, [ebp+8] ; ESI = array source address
    xor         eax, eax    ; Clear eax (accumulator) to store the sum
    mov         ecx, [ebp+12] ; ECX = counter (number of elements)

.loop:
    add         eax, [esi]  ; Add the value at the current array element to the sum
    add         esi, 4      ; Move to the next element (4 bytes ahead)
    loop        .loop       ; Repeat the loop for all elements

    pop         esi         ; Restore esi
    pop         ebp         ; Restore the base pointer
    ret                     ; Return

;----------------------------
; Print_char:
; Description: Prints a single character.
; Receives: arg1 = char value
; Requires: Preserve ebp
; Requires: char_buff (a buffer provided below)
;----------------------------
print_char:
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame

    mov         [char_buff], [ebp + 11] ; Move the character value to the buffer
    mov         eax, 4      ; System call number for write
    mov         ebx, 1      ; File descriptor (stdout)
    mov         ecx, char_buff ; Pointer to the character buffer
    mov         edx, 1      ; Number of bytes to write
    int         80h         ; Invoke the system call to write the character

    pop         ebp         ; Restore ebp
    ret                     ; Return

;----------------------------
; Rev_str:
; Description: Reverses a string.
; Receives: arg1 = Address of the string
;           arg2 = Size in bytes of the string
; Returns:  EAX = Address of the reversed array
;----------------------------
rev_str:
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi
    push        edi         ; Preserve edi

    test        [ebp + 12], [ebp + 12] ; Check if the string size is zero
    jnz         .start      ; If not zero, start the reversal process

    mov         dword [errno], -1       ; Set errno to -1 (error code)
    jmp         .exit       ; Jump to exit

.start:
    mov         ecx, [ebp + 12]     ; Set the counter to the string size
    mov         esi, [ebp + 8]      ; Set the source pointer (string address)
    mov         edi, [ebp + 8]      ; Set the destination pointer (same as source)

.pushloop:
    movzx       edx, byte [esi]     ; Load the next character into edx
    push        edx                 ; Push the character onto the stack
    inc         esi                 ; Move to the next character
    loop        .pushloop           ; Repeat for all characters in the string

    mov         ecx, [ebp + 12]     ; Reset the counter

.poploop:
    pop         edx                 ; Pop a character from the stack
    mov         [edi], dl           ; Store the character in the destination
    inc         edi                 ; Move to the next position in the destination
    loop        .poploop            ; Repeat for all characters

.exit:
    pop         edi                 ; Restore edi
    pop         esi                 ; Restore esi
    pop         ebp                 ; Restore the base pointer
    ret                             ; Return

;----------------------------
; Get_input:
; Description: Gets input from the user.
; Receives: arg1 = Address of the prompt
;           arg2 = Prompt size
;           arg3 = Input buffer address
;           arg4 = Input buffer size
; Returns:  EAX = Inputted string
;----------------------------
get_input:
    push        ebp                 ; Preserve the base pointer
    mov         ebp, esp            ; Set up a new stack frame

    push        [ebp + 12]          ; Preserve arg2 (prompt size)
    push        edi                 ; Preserve edi
    push        [ebp + 16]          ; Preserve arg3 (input buffer address)
    push        [ebp + 20]          ; Preserve arg4 (input buffer size)
    mov         edi, [ebp + 16]     ; Move the input buffer address into edi

    call        print_str           ; Call the print_str function to print the prompt

    mov         eax, 3              ; System call number for read
    mov         ebx, 0              ; File descriptor (stdin)
    pop         edx                 ; Restore edx (input buffer size)
    pop         ecx                 ; Restore ecx (input buffer)
    int         80h                 ; Invoke the system call to read user input

    mov         BYTE [edi + eax], 0 ; Null-terminate the input string
    pop         edi                 ; Restore edi
    pop         ebx                 ; Restore ebx
    pop         ebp                 ; Restore the base pointer

    ret                             ; Return

;----------------------------
; Print_str:
; Description: Prints a string.
; Receives: arg1 = Address of the string
;           arg2 = Size of the string
;----------------------------
print_str:
    push        ebp                 ; Preserve the base pointer
    mov         ebp, esp            ; Set up a new stack frame

    push        [ebp + 12]          ; Preserve arg2 (size of the string)
    mov         ecx, [ebp + 8]      ; Move the address of the string into ecx
    mov         edx, [ebp + 12]     ; Move the size of the string into edx
    mov         eax, 4              ; System call number for write
    mov         ebx, 1              ; File descriptor (stdout)
    int         80h                 ; Invoke the system call to write the string

    dec         eax                 ; Adjust eax to remove the newline character
    mov         byte [edi + eax], 0 ; Null-terminate the string

    pop         ebx                 ; Restore ebx
    pop         ebp                 ; Restore the base pointer
    ret                             ; Return

;----------------------------
; Itoa:
; Description: Converts an integer to a string.
; Receives: arg1 = Integer value
;           arg2 = String buffer address (assume it's big enough)
; Returns:  Nothing
; Requires: Nothing
;----------------------------
itoa:
    push        ebp             ; Preserve the base pointer
    mov         ebp, esp        ; Set up a new stack frame
    push        edi             ; Preserve edi
    push        ebx             ; Preserve ebx

    mov         ebx, 10         ; Load the factor (10) into ebx
    mov         eax, [ebp + 8]  ; Load the integer value into eax
    mov         edi, [ebp + 12] ; Load the address of the string buffer into edi

.do:
    xor         edx, edx        ; Clear edx (used for remainder)
    div         ebx             ; Divide eax by 10, result in eax, remainder in edx
    add         dl, 48          ; Convert the remainder to ASCII
    mov         [edi], dl       ; Store the ASCII character in the buffer
    inc         edi             ; Move to the next position in the buffer
    test        eax, eax        ; Check if eax is zero (end of conversion)
    jnz         .do             ; If not zero, repeat the loop

.wend:
    mov         byte [edi], 0   ; Null-terminate the string
    pop         ebx             ; Restore ebx
    pop         edi             ; Restore edi
    leave                       ; Clean up the stack frame
    ret                         ; Return

section .bss
    char_buff_sz:   equ 255
    char_buff:      resb char_buff_sz

    input_sz:       equ 255
    input:          resb input_sz

section .data
    errno: dd 0

    next:   dd    1
    const1: equ   1103515245
    const2: equ   12345
    const3: equ   65536
    const4: equ   32768
