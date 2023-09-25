; who: Maayan Israel
; what: make a copy of a file in assembly
; why: because I am an engineer!
; when: Spring 2023

section .text

global _start

_start:

    ; Open source file (syscall: open)
    mov     eax, 5              ; syscall number for open
    mov     ebx, src_path       ; address of source file path
    mov     ecx, 0              ; file flags (O_RDONLY)
    mov     edx, 0777           ; file permission mode (read, write, execute for owner)
    int     80h                  ; invoke syscall

    ; Store the file descriptor of the source file
    mov     esi, eax            ; store the file descriptor in esi

    ; Create destination file (syscall: creat)
    mov     eax, 8              ; syscall number for creat
    mov     ebx, dst_path       ; address of destination file path
    mov     ecx, 0777           ; file permission mode (read, write, execute for owner)
    int     80h                  ; invoke syscall

    ; Store the file descriptor of the destination file
    mov     edi, eax            ; store the file descriptor in edi

copy_loop:
    ; Read from source file to buffer (syscall: read)
    mov     eax, 3              ; syscall number for read
    mov     ebx, esi            ; file descriptor of source file
    mov     ecx, buffer         ; address of buffer
    mov     edx, buff_sz        ; maximum number of bytes to read
    int     80h                  ; invoke syscall

    ; Check if bytesRead is 0, which indicates end of file
    test    eax, eax            ; test if bytesRead is zero
    jz      end_copy_loop       ; if so, exit the loop

    ; Write from buffer to destination file (syscall: write)
    mov     edx, eax            ; number of bytes to write (same as bytesRead)
    mov     eax, 4              ; syscall number for write
    mov     ebx, edi            ; file descriptor of destination file
    mov     ecx, buffer         ; address of buffer
    int     80h                  ; invoke syscall

    ; Repeat the copy loop
    jmp     copy_loop

end_copy_loop:
    ; Close source file (syscall: close)
    mov     eax, 6              ; syscall number for close
    mov     ebx, esi            ; file descriptor of source file
    int     80h                  ; invoke syscall

    ; Close destination file (syscall: close)
    mov     eax, 6              ; syscall number for close
    mov     ebx, edi            ; file descriptor of destination file
    int     80h                  ; invoke syscall

exit:
    ; Exit the program
    mov     ebx, 0              ; return 0 status on exit (no errors)
    mov     eax, 1              ; invoke SYS_EXIT (kernel opcode 1)
    int     80h                  ; invoke syscall

section .bss
    buff_sz: equ 4096            ; buffer size
    buffer: resb buff_sz         ; buffer for copying data

section .data

    src_path: db "einstein.jpeg", 0     ; source file path
    dst_path: db "einstein_copy.jpeg", 0 ; destination file path
