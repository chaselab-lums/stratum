/* RV32 bare-metal layout: .text.init @ 0x8000_0000, HTIF compatibility page. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
  . = 0x80000000;
  .text.init : { *(.text.init) }

  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }

  . = ALIGN(0x1000);
  .text : {
    *(.text .text.*)
    *(.rodata .rodata.*)
  }

  . = ALIGN(16);
  .data : { *(.data .data.*) }

  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.srodata .srodata.*)
    *(.sdata .sdata.*)
  }

  .sbss : {
    *(.sbss .sbss.*)
    *(.scommon)
  }

  .bss : { *(.bss .bss.*) *(COMMON) }

  .tdata : {
    _tdata_begin = .;
    *(.tdata .tdata.*)
    _tdata_end = .;
  }

  .tbss : {
    *(.tbss .tbss.*)
    _tbss_end = .;
  }

  _end = .;
}
