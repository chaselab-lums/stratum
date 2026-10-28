#ifndef SNN_GEM5_PROFILE_H
#define SNN_GEM5_PROFILE_H

#include <stdint.h>

#ifdef ENABLE_GEM5_M5OPS
#include <gem5/m5ops.h>

static inline void snn_gem5_roi_begin(uint64_t work_id)
{
    (void)work_id;
    m5_reset_stats(0, 0);
}

static inline void snn_gem5_roi_end(uint64_t work_id)
{
    (void)work_id;
    m5_dump_reset_stats(0, 0);
}
#else
static inline void snn_gem5_roi_begin(uint64_t work_id)
{
    (void)work_id;
}

static inline void snn_gem5_roi_end(uint64_t work_id)
{
    (void)work_id;
}
#endif

#endif