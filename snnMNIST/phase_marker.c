/* Phase cycle markers using rdcycle (RV32: 32-bit counter, accumulate in 64-bit). */
#include <stdint.h>

volatile uint64_t snn_phase_cycles[5]; /* 0=scan, 1=gather, 2=lif1, 3=fc2, 4=other */

static uint32_t phase_last_cycle;
static int phase_have_last;

static inline uint32_t read_cycle32(void) {
    uint32_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

static inline uint32_t cycle_delta(uint32_t now, uint32_t last) {
    return now - last; /* unsigned wrap is correct for rdcycle on RV32 */
}

void snn_phase_marker(int phase_id) {
    uint32_t now = read_cycle32();
    if (phase_have_last && phase_id >= 0 && phase_id < 5) {
        snn_phase_cycles[phase_id] += cycle_delta(now, phase_last_cycle);
    }
    phase_last_cycle = now;
    phase_have_last = 1;
}

void snn_phase_reset(void) {
    for (int i = 0; i < 5; i++) {
        snn_phase_cycles[i] = 0;
    }
    phase_last_cycle = read_cycle32();
    phase_have_last = 0;
}
