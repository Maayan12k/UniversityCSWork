; who: Maayan Israel
; what: get and print
; why: discover how to print 
; when: today

section .text

global _start

_start:
    ; Prompt for name
    mov eax, 4              ; Write operation (syscall number)
    mov ebx, 1              ; Write to stdout
    mov ecx, get_name       ; Address of the name prompt
    mov edx, g_name_sz      ; Size of the prompt
    int 0x80                ; Invoke syscall to print the prompt

    ; Get name from user
    mov eax, 3              ; Read operation (syscall number)
    mov edx, 0              ; Read from stdin
    mov ecx, name_buff      ; Address to store the user's input
    mov edx, buff_sz        ; Size of the input buffer
    int 0x80                ; Invoke syscall to read user input

    mov [ninput_sz], eax    ; Store the size of the input

    ; Prompt for address
    mov eax, 4              ; Write operation
    mov ebx, 1              ; Write to stdout
    mov ecx, get_addr       ; Address of the address prompt
    mov edx, g_addr_sz      ; Size of the prompt
    int 0x80                ; Invoke syscall to print the address prompt

    ; Get address from user
    mov eax, 3              ; Read operation
    mov edx, 0              ; Read from stdin
    mov ecx, addr_buff      ; Address to store the user's input
    mov edx, buff_sz        ; Size of the input buffer
    int 0x80                ; Invoke syscall to read user input

    mov [ainput_sz], eax    ; Store the size of the input

    ; Print name
    mov edx, [ninput_sz]    ; Size of the name input
    mov eax, 4              ; Write operation
    mov ebx, 1              ; Write to stdout
    mov ecx, name_buff      ; Address of the name input
    int 0x80                ; Invoke syscall to print the name

    ; Print address
    mov edx, [ainput_sz]    ; Size of the address input
    mov eax, 4              ; Write operation
    mov ebx, 1              ; Write to stdout
    mov ecx, addr_buff      ; Address of the address input
    int 0x80                ; Invoke syscall to print the address

exit:
    mov ebx, 0              ; Return 0 status on exit - 'No Errors'
    mov eax, 1              ; Invoke SYS_EXIT (kernel opcode 1)
    int 0x80                ; Trigger the kernel interrupt to exit the program

section .bss
    buff_sz:    equ 255     ; Define a constant for buffer size
    name_buff:  resb buff_sz ; Reserve space for the name input buffer
    addr_buff:  resb buff_sz ; Reserve space for the address input buffer
    ninput_sz:  resd 1      ; Reserve space to store the size of the name input
    ainput_sz:  resd 1      ; Reserve space to store the size of the address input

section .data
    get_name:   db "Enter name: "    ; Define the name prompt
    g_name_sz:  equ $ - get_name     ; Calculate the size of the name prompt

    get_addr:   db "Enter address: " ; Define the address prompt
    g_addr_sz:  equ $ - get_addr     ; Calculate the size of the address prompt



