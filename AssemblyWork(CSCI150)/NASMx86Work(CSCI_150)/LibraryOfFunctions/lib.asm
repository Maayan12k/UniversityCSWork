; who: Maayan Israel
; what: Library for all assembly functions used in CSCI 150
; why: because I am an engineer!
; when: Spring 2023

section .text

; This section contains various global functions and procedures. It also passes arguments via the stack

global atoi         ; Function to convert a string to an integer
global sum_array    ; Function to sum the values of an array
global print_char   ; Function to print a single character
global rev_str      ; Function to reverse a string
global get_input    ; Function to get user input
global print_str    ; Function to print a string
global errno        ; Global variable for error handling
global current_time ; Function to get the current time
global seed_random  ; Function to seed the random number generator
global gen_random   ; Function to generate random numbers
global print_uint   ; Function to print an unsigned integer
global print_nl     ; Function to print a newline character
global strlen       ; Function to calculate the length of a string
global swap         ; Function to swap two values
global malloc       ; Function to allocate memory dynamically

malloc:
; Description: Allocates a specified amount of memory on the heap
; Receives: arg1: quantity of bytes to allocate
; Returns: EAX = address of allocated space
; Algorithm: Calls the "break" system call to allocate memory
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame
    push    ebx             ; Preserve ebx

    mov     edx, [ebp + 8]  ; Get the number of bytes in edx

    mov     eax, 0x2d       ; System call code for "break"
    xor     ebx, ebx        ; Clear ebx (initial heap address)
    int     80h             ; Call the kernel

    mov     ebx, eax        ; Store the result in ebx

    mov     eax, 0x2d       ; System call code for "break"
    lea     ebx, [ebx + edx]; Load a new effective address with edx bytes
    int     80h             ; Call the kernel

    mov     eax, ebx        ; Return the allocated address in eax

    pop     ebx             ; Restore ebx
    leave                   ; Restore the previous stack frame
    ret                     ; Return

strlen:
; Description: Compute the length of a null-terminated string
; Receives: arg1: string address
;          arg2: max buffer size
; Returns: EAX = length of the string
; Algorithm: Uses the "scasb" instruction to find the null terminator
;            and calculates the length based on the difference.
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame

    mov     edi, [ebp + 8]  ; Set edi to point to the string
    xor     eax, eax        ; Clear eax
    mov     ecx, [ebp + 12] ; Set ecx to the max buffer size
    cld                     ; Set direction to forward (searching for null terminator)

    repne   scasb           ; Search for the null terminator
    cmp     ecx, 0          ; Check if the string length is equal to the buffer size
    jz      .default         ; If true, use the buffer size as the length

    dec     edi             ; Set edi to point to the character right before null
    mov     eax, edi        ; Copy the address to eax for subtraction
    sub     eax, [ebp + 8]  ; Calculate the length in eax
    jmp     .done

.default:
    mov     eax, [ebp + 12] ; Default is to use the buffer size as the length
    dec     eax             ; Adjust for null termination

.done:
    pop     ebp             ; Restore the base pointer
    ret                     ; Return with the result in eax

print_nl:
; Description: Prints a new line character
; Receives: Nothing
; Returns: Nothing
; Requires: print_char function

    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame

    push        0x0a        ; Push the newline character (ASCII 0x0a)
    call        print_char  ; Call the print_char function to print it

    leave                   ; Restore the previous stack frame
    ret                     ; Return

print_uint:
; Description: Prints an unsigned integer
; Receives: arg1 = uint value
; Requires: character buffer (char_buff)
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame

    push    char_buff       ; Push the address of the character buffer
    push    DWORD [ebp + 8] ; Push the unsigned integer value
    call    itoa            ; Call the itoa function to convert integer to string
    add     esp, 8          ; Deallocate the pushed arguments

    push    char_buff_sz    ; Push the size of the character buffer
    push    char_buff       ; Push the address of the character buffer
    call    strlen          ; Call the strlen function to calculate the string length
    add     esp, 4          ; Deallocate the pushed arguments

    push    eax             ; Push the length of the string
    push    char_buff       ; Push the address of the character buffer
    call    print_str       ; Call the print_str function to print the string

    pop     ebx             ; Restore ebx

    leave                   ; Restore the previous stack frame
    ret                     ; Return

seed_random:
; Description: Seeds the random number generator
; Receives: arg1: seed value
    push    ebp             ; Preserve the base pointer
    mov     ebp, esp        ; Set up a new stack frame

    mov     eax, [ebp + 8]  ; Get the seed value
    mov     [next], eax     ; Store it in the "next" variable

    pop     ebp             ; Restore the base pointer
    ret                     ; Return

gen_random:
; Description: Generates a random number
; Returns: EAX = random value
    mov     eax, [next]     ; Move the current "next" value into eax
    mov     ecx, const1     ; Move constant1 into ecx
    mul     ecx             ; Multiply eax by ecx (1103515245)

    add     eax, const2     ; Add constant2 (12345)
    mov     [next], eax     ; Store the new "next" value

    xor     edx, edx        ; Clear edx
    mov     ecx, const3     ; Move constant3 into ecx
    div     ecx             ; Divide edx:eax by ecx (65536)

    mov     ecx, const4     ; Move constant4 into ecx
    xor     edx, edx        ; Clear edx
    div     ecx             ; Divide edx:eax by ecx (32768)
    mov     eax, edx        ; Return the result in eax

    ret

current_time:
; Description: Gets the number of seconds since 1/1/1970 (Unix timestamp)
    push        ebx         ; Preserve ebx
    mov         eax, 13     ; Set eax to the "time" system call number
    xor         ebx, ebx    ; Clear ebx
    int         80h         ; Call the kernel interrupt

    pop         ebx         ; Restore ebx
    ret

atoi:
; Description: Converts a null-terminated string of digits to an integer
; Receives: arg1: address of the string
; Returns: EAX = the integer value
; Algorithm: Horner's method
.factor:    equ 10
.offset:    equ 48

    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi
    mov         ESI, [ebp+8]; ESI = source
    xor         eax, eax    ; Clear eax
    xor         ecx, ecx    ; Clear ecx
    mov         ebx, .factor; ebx = 10

    .while:
    mov         cl, [esi]   ; Load the first character
    test        cl, cl      ; Check if it's null-terminated
    jz          .wend       ; If null-terminated, exit the loop
    sub         cl, .offset ; Calculate the integer value from ASCII
    mul         ebx         ; Multiply eax by 10
    add         eax, ecx    ; Add the next character to eax

    inc         esi         ; Move to the next character
    jmp         .while      ; Repeat the loop

    .wend:
    pop         esi         ; Restore esi
    pop         ebp         ; Restore the base pointer
    ret                     ; Return with the result in eax

sum_array:
; Description: Sums the values of a double word array
; Receives: arg1: array address
;          arg2: number of elements
; Returns: EAX = sum
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi

    mov         esi, [ebp+8]; ESI = array source address
    xor         eax, eax    ; Clear eax
    mov         ecx, [ebp+12]; ECX = counter

    .loop:
    add         eax, [esi]  ; Add the current element to eax
    add         esi, 4      ; Move to the next element (4 bytes)
    loop    .loop

    pop         esi         ; Restore esi
    pop         ebp         ; Restore the base pointer
    ret                     ; Return with the sum in eax

print_char:
; Description: Prints a single character
; Receives: arg1: char value
; Requires: a buffer provided below

    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame

    mov         al,  [ebp +11] ; Move the char value to al
    mov         [char_buff], al; Store it in the character buffer
    mov         eax, 4      ; Write system call
    mov         ebx, 1      ; Standard output
    mov         ecx, char_buff ; Load the address of the character buffer
    mov         edx, 1      ; Length of 1 character
    int         80h         ; Call the kernel

    pop         ebp         ; Restore the base pointer
    ret                     ; Return

rev_str:
; Description: Reverses a string
; Receives: arg1: address of the string
;          arg2: size in bytes of the string
; Returns: None
    push        ebp         ; Preserve the base pointer
    mov         ebp, esp    ; Set up a new stack frame
    push        esi         ; Preserve esi
    push        edi         ; Preserve edi

    mov         eax, [ebp +12] ; Load the size in bytes
    test        [ebp + 12], eax ; Check if the size is zero
    jnz         .start      ; If not zero, proceed

    mov         dword [errno], -1 ; Set errno to -1 for zero-size string
    jmp         .exit

.start:
    mov         ecx, [ebp + 12] ; Set the counter to the size
    mov         esi, [ebp + 8]  ; Load the source address
    mov         edi, [ebp + 8]  ; Load the destination address

.pushloop:
    movzx       edx, byte [esi] ; Load a character into edx
    push        edx             ; Push the character onto the stack
    inc         esi             ; Calculate the next effective address
    loop    .pushloop

    mov         ecx, [ebp + 12] ; Reset the counter

.poploop:
    pop         edx             ; Pop a character from the stack
    mov         [edi], dl       ; Store it in the destination
    inc         edi             ; Calculate the next effective address
    loop    .poploop

.exit:
    pop         edi             ; Restore edi
    pop         esi             ; Restore esi
    pop         ebp             ; Restore the base pointer
    ret

get_input:
; Description: Gets user input
; Receives: arg1: prompt address
;          arg2: prompt size
;          arg3: input buffer address
;          arg4: input buffer size
; Returns: EAX = inputted string
    push        ebp             ; Preserve the base pointer
    mov         ebp, esp        ; Set up a new stack frame

    push        dword [ebp + 16]; Preserve the input buffer address
    push        dword [ebp + 20]; Preserve the input buffer size
    push        dword [ebp + 12]; Preserve the prompt size
    push        dword [ebp + 8] ; Preserve the prompt address
    mov         edi, [ebp + 16] ; Move the buffer address into edi

    call        print_str       ; Call the print_str function to print the prompt

    mov         eax, 3          ; Read operation
    mov         ebx, 0          ; Read from stdin
    pop         edx             ; Load the input buffer size
    pop         ecx             ; Load the input buffer address
    int         80h             ; Call the kernel

    mov         BYTE [edi + eax], 0 ; Null-terminate the input string

    pop         edi             ; Restore registers
    pop         ebx
    pop         ebp             ; Restore the base pointer
    ret

print_str:
; Description: Prints a string
; Receives: arg1: address of the string
;          arg2: size of the string
    push        ebp             ; Preserve the base pointer
    mov         ebp, esp        ; Set up a new stack frame

    mov         ecx, [ebp + 8]  ; Load the address of the string
    mov         edx, [ebp + 12] ; Load the size of the string
    mov         eax, 4          ; Write operation
    mov         ebx, 1          ; Standard output
    int         80h             ; Call the kernel

    ; EAX holds the quantity of characters printed
    dec         eax             ; Remove the newline character
    mov         byte [edi + eax], 0 ; Null-terminate the string

    pop         ebp             ; Restore the base pointer
    ret                         ; Return

swap:
; Description: Swaps two values
; Receives: arg1: address of the first value
;          arg2: address of the second value
    push        ebp             ; Preserve base pointer
    mov         ebp, esp        ; Set up a new stack frame

    push        esi             ; Preserve registers
    push        edi

    mov         esi, [ebp + 8]  ; ESI holds the first address
    mov         edi, [ebp + 12] ; EDI holds the second address
    mov         eax, [esi]      ; EAX = temp = value at the first address
    xchg        eax, [edi]      ; Swap EAX with the value at the second address
    mov         [esi], eax      ; Store the temp value at the first address

    pop         edi             ; Restore registers
    pop         esi

    mov         esp, ebp        ; Restore the stack pointer
    pop         ebp             ; Restore the caller's base pointer

    ret                         ; Return

itoa:
; Description: Converts an integer to a string
; Receives: arg1: integer
;          arg2: string buffer address (assume it's big enough)
; Returns: Nothing
    push        ebp             ; Preserve caller's base pointer
    mov         ebp, esp        ; Set base of frame
    push        edi             ; Preserve registers
    push        ebx

    mov         ebx, 10         ; EBX = 10 (for base 10)
    mov         eax, [ebp + 8]  ; Store integer in EAX
    mov         edi, [ebp + 12] ; Store buffer address in EDI

.do:
    xor         edx, edx        ; Clear EDX (remainder)
    div         ebx             ; Divide EAX by 10, result in EAX, remainder in EDX
    add         dl, 48          ; Convert remainder to ASCII
    mov         [edi], dl       ; Store the character in the buffer
    inc         edi             ; Move to the next character

    test        eax, eax        ; Check if EAX is zero (quotient)
    jnz         .do             ; If not zero, repeat

.wend:
    mov         byte [edi], 0   ; Null-terminate the string

    pop         ebx             ; Restore registers
    pop         edi
    leave                       ; Restore the stack frame
    ret                         ; Return

section .bss
    char_buff_sz:   equ 255     ; Size of the character buffer
    char_buff:      resb char_buff_sz

    input_sz: equ 255           ; Size of the input buffer
    input: resb input_sz

section .data
    errno: dd 0                 ; Global variable for error handling

    next:   dd    1             ; Seed value for random number generator
    const1: equ 1103515245      ; Constants for random number generation
    const2: equ 12345
    const3: equ 65536
    const4: equ 32768
