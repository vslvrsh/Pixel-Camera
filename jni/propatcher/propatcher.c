#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <android/log.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <arm_neon.h>

#define LOG_TAG "ProPatcher"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Known HAL / AHardwareBuffer 8-bit formats
#define FMT_RGBA_8888 1
#define FMT_RGBX_8888 2
#define FMT_RGB_888   3
#define FMT_NV21      0x11
#define FMT_IMPL_DEF  0x22
#define FMT_YUV_420   0x23
#define FMT_YV12      0x32315659

static pthread_mutex_t s_lut_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t s_selfie_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int s_grain_enabled = 0;

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeSetGrainEnabled(
    JNIEnv *env, jclass clazz, jboolean enabled) {
    (void)env; (void)clazz;
    s_grain_enabled = (enabled == JNI_TRUE) ? 1 : 0;
    LOGI("ProPatcher: Film Grain state set to %s", s_grain_enabled ? "ENABLED" : "DISABLED");
}

JNIEXPORT jboolean JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeIsProPatcherLoaded(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeApplyTuning(
    JNIEnv *env, jclass clazz, jfloat sharpenScale, jint denoiseMode) {
    (void)env; (void)clazz; (void)sharpenScale; (void)denoiseMode;
    // Bypassed: Tuning is applied via real-time NEON SIMD algorithms on HardwareBuffers
    // and InterleavedWriteViewU8 frames (nativeSetFineTuning).
    // Direct .rodata/text segment mprotect is strictly bypassed for GrapheneOS W^X safety.
    LOGI("nativeApplyTuning: bypassed for GOS and Pixel memory safety");
}

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// ---------------------------------------------------------------------------
// Photographic Tone Curve & Chroma Calibration Engine (Google Tomte Aligned)
// ---------------------------------------------------------------------------

typedef struct {
    float contrast_p;    // Sigmoid exponent (>1.0 = punchy S-curve, <1.0 = soft/log)
    float black_lift;    // Black pedestal in [0.0, 1.0] (lifted film blacks)
    float white_comp;    // White shoulder point in [0.0, 1.0] (highlight roll-off)
    float saturation;    // Chroma saturation factor (0.0 = monochrome, >1.0 = punch)
    float shift_u;       // Blue/Yellow chroma shift (- = yellow/warm, + = blue/cool)
    float shift_v;       // Red/Cyan chroma shift (+ = red/amber/magenta, - = green/cyan)
    int is_monochrome;   // 1 for Black Tie (zero chroma)
} LookProfile;

static const LookProfile s_look_profiles[13] = {
    { 1.00f, 0.0f, 1.0f, 1.00f, 0.0f, 0.0f, 0 },                              // 0: Original
    { 1.02f, 0.0f, 1.0f, 1.02f, 0.0f, 0.0f, 0 },                              // 1: Natural (qvb)
    { 0.92f, 12.0f/255.0f, 248.0f/255.0f, 0.72f, 0.0f, 0.0f, 0 },             // 2: Minimal (qva)
    { 1.18f, 5.0f/255.0f, 252.0f/255.0f, 1.04f, +2.5f, -1.5f, 0 },            // 3: Shadows (qvf)
    { 0.98f, 4.0f/255.0f, 252.0f/255.0f, 1.05f, -4.0f, +3.0f, 0 },            // 4: Vanilla (qvc)
    { 1.14f, 2.0f/255.0f, 253.0f/255.0f, 1.14f, +3.0f, -1.5f, 0 },            // 5: Digi (quy)
    { 1.26f, 0.0f, 1.0f, 0.00f, 0.0f, 0.0f, 1 },                              // 6: Black Tie (quw)
    { 0.95f, 12.0f/255.0f, 245.0f/255.0f, 1.03f, -4.5f, -2.0f, 0 },           // 7: Dijon (qvi)
    { 1.15f, 2.0f/255.0f, 252.0f/255.0f, 1.08f, -3.0f, +3.5f, 0 },            // 8: Buffalo (qvg)
    { 0.86f, 18.0f/255.0f, 238.0f/255.0f, 0.70f, 0.0f, 0.0f, 0 },             // 9: Flat (qvd)
    { 1.06f, 6.0f/255.0f, 250.0f/255.0f, 1.02f, -2.5f, +2.5f, 0 },            // 10: Classic (qve)
    { 1.16f, 0.0f, 1.0f, 1.10f, +1.5f, +2.5f, 0 },                            // 11: Editorial (quz)
    { 1.12f, 2.0f/255.0f, 252.0f/255.0f, 1.18f, 0.0f, +2.0f, 0 }              // 12: Velvet (qvh)
};

static int s_static_tables_inited = 0;
static int8_t s_noise_table[512];
static uint8_t s_tint_weight[256];

// Protected by s_lut_mutex
static float s_ft_contrast = 0.0f;
static float s_ft_shadows = 0.0f;
static float s_ft_highlights = 0.0f;
static float s_ft_vibrance = 0.0f;
static float s_ft_tint = 0.0f;
static float s_ft_warmth = 0.0f;
static float s_ft_sharpness = 1.0f;

static uint8_t s_active_lut_y[256];
static float s_active_saturation = 1.0f;
static float s_active_shift_u = 0.0f;
static float s_active_shift_v = 0.0f;
static int s_active_look_id = -999;
static float s_last_c = -999.0f, s_last_s = -999.0f, s_last_h = -999.0f;
static float s_last_v = -999.0f, s_last_t = -999.0f, s_last_w = -999.0f;
static int s_active_lut_valid = 0;

// Caller must hold s_lut_mutex
static void init_static_tables_locked(void) {
    if (s_static_tables_inited) return;

    // Initialize 512-byte cyclic film grain noise pattern
    uint32_t s = 0x19385721u;
    for (int i = 0; i < 512; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s_noise_table[i] = (int8_t)((s & 0x1Fu) - 16); // [-16, 15]
    }

    // Luminance-weighted split-toning weights:
    // Pure shadows (< 24): ramp 0..255 (keeps blacks ink-rich)
    // Midtones (24..205): 255
    // Highlights (> 205): ramp 255..0 (specular whites stay clean)
    for (int i = 0; i < 256; i++) {
        int w = 255;
        if (i < 24) {
            w = (i * 255) / 24;
        } else if (i > 205) {
            w = ((255 - i) * 255) / 50;
        }
        s_tint_weight[i] = clamp_u8(w);
    }

    s_static_tables_inited = 1;
    LOGI("ProPatcher: Photographic tables initialized (thread-safe, 512-byte grain).");
}

// Caller must hold s_lut_mutex
static void update_active_luts_locked(int lookId) {
    if (s_active_lut_valid && s_active_look_id == lookId &&
        s_last_c == s_ft_contrast && s_last_s == s_ft_shadows &&
        s_last_h == s_ft_highlights && s_last_v == s_ft_vibrance &&
        s_last_t == s_ft_tint && s_last_w == s_ft_warmth) {
        return;
    }

    const LookProfile* base_p = (lookId >= 0 && lookId < 13) ? &s_look_profiles[lookId] : &s_look_profiles[0];

    // 1. Contrast: sigmoid exponent adjustment
    float eff_contrast_p = base_p->contrast_p;
    if (s_ft_contrast > 0.0f) {
        eff_contrast_p += s_ft_contrast * 0.55f;
    } else if (s_ft_contrast < 0.0f) {
        eff_contrast_p += s_ft_contrast * 0.40f;
    }
    if (eff_contrast_p < 0.45f) eff_contrast_p = 0.45f;
    if (eff_contrast_p > 2.50f) eff_contrast_p = 2.50f;

    // Build Y (Luma) Tone Curve with continuous photographic zone adjustment
    for (int i = 0; i < 256; i++) {
        float x = (float)i / 255.0f;
        float sig;
        if (x <= 0.0001f) {
            sig = 0.0f;
        } else if (x >= 0.9999f) {
            sig = 1.0f;
        } else {
            float xp = powf(x, eff_contrast_p);
            float omxp = powf(1.0f - x, eff_contrast_p);
            sig = xp / (xp + omxp);
        }
        float y_base = base_p->black_lift + sig * (base_p->white_comp - base_p->black_lift);

        // 2. Shadows photographic bell curve (0.0..0.60, peak at 0.30, anchored at 0)
        float delta_s = 0.0f;
        if (x < 0.60f) {
            float ts = x / 0.60f;
            float ws = 4.0f * ts * (1.0f - ts);
            delta_s = s_ft_shadows * 0.28f * ws;
        }

        // 3. Highlights photographic bell curve (0.40..1.0, peak at 0.70, anchored at 1)
        float delta_h = 0.0f;
        if (x > 0.40f) {
            float th = (x - 0.40f) / 0.60f;
            float wh = 4.0f * th * (1.0f - th);
            delta_h = -s_ft_highlights * 0.28f * wh;
        }

        float y_out = y_base + delta_s + delta_h;
        if (y_out < 0.0f) y_out = 0.0f;
        if (y_out > 1.0f) y_out = 1.0f;
        s_active_lut_y[i] = clamp_u8((int)roundf(y_out * 255.0f));
    }

    // 4. Vibrance (Saturation)
    float eff_saturation = base_p->saturation;
    if (!base_p->is_monochrome) {
        eff_saturation *= (1.0f + 0.70f * s_ft_vibrance);
        if (eff_saturation < 0.0f) eff_saturation = 0.0f;
        if (eff_saturation > 2.50f) eff_saturation = 2.50f;
    }

    // 5. Warmth & Tint (BT.601 YUV chroma shift):
    // +Warmth (Yellow/Amber): -U (-24), +V (+12)
    // +Tint (Magenta vs Green): +U (+14), +V (+18)  [since +R,+B,-G requires +U and +V]
    float eff_shift_u = base_p->shift_u - (s_ft_warmth * 24.0f) + (s_ft_tint * 14.0f);
    float eff_shift_v = base_p->shift_v + (s_ft_warmth * 12.0f) + (s_ft_tint * 18.0f);

    s_active_saturation = eff_saturation;
    s_active_shift_u = eff_shift_u;
    s_active_shift_v = eff_shift_v;

    s_active_look_id = lookId;
    s_last_c = s_ft_contrast;
    s_last_s = s_ft_shadows;
    s_last_h = s_ft_highlights;
    s_last_v = s_ft_vibrance;
    s_last_t = s_ft_tint;
    s_last_w = s_ft_warmth;
    s_active_lut_valid = 1;
}

typedef struct {
    uint8_t lut_y[256];
    uint8_t tint_weight[256];
    float saturation;
    float shift_u;
    float shift_v;
    int is_monochrome;
    int has_fine_tuning;
} GradingSnapshot;

static void capture_grading_snapshot(int lookId, GradingSnapshot* out) {
    pthread_mutex_lock(&s_lut_mutex);
    init_static_tables_locked();

    int has_ft = (fabsf(s_ft_contrast) > 0.01f ||
                  fabsf(s_ft_shadows) > 0.01f ||
                  fabsf(s_ft_highlights) > 0.01f ||
                  fabsf(s_ft_vibrance) > 0.01f ||
                  fabsf(s_ft_tint) > 0.01f ||
                  fabsf(s_ft_warmth) > 0.01f);
    out->has_fine_tuning = has_ft;

    if (lookId != 0 || has_ft || s_grain_enabled) {
        update_active_luts_locked(lookId);
        const LookProfile* prof = (lookId >= 0 && lookId < 13) ? &s_look_profiles[lookId] : &s_look_profiles[0];
        memcpy(out->lut_y, s_active_lut_y, 256);
        memcpy(out->tint_weight, s_tint_weight, 256);
        out->saturation = s_active_saturation;
        out->shift_u = s_active_shift_u;
        out->shift_v = s_active_shift_v;
        out->is_monochrome = prof->is_monochrome;
    } else {
        out->saturation = 1.0f;
        out->shift_u = 0.0f;
        out->shift_v = 0.0f;
        out->is_monochrome = 0;
    }

    pthread_mutex_unlock(&s_lut_mutex);
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeSetFineTuning(
    JNIEnv *env, jclass clazz, jfloat contrast, jfloat shadows, jfloat highlights, jfloat vibrance, jfloat tint, jfloat warmth, jfloat sharpness) {
    (void)env; (void)clazz;
    pthread_mutex_lock(&s_lut_mutex);
    if (s_ft_contrast != contrast || s_ft_shadows != shadows || s_ft_highlights != highlights ||
        s_ft_vibrance != vibrance || s_ft_tint != tint || s_ft_warmth != warmth || s_ft_sharpness != sharpness) {
        s_ft_contrast = contrast;
        s_ft_shadows = shadows;
        s_ft_highlights = highlights;
        s_ft_vibrance = vibrance;
        s_ft_tint = tint;
        s_ft_warmth = warmth;
        s_ft_sharpness = sharpness;
        s_active_lut_valid = 0;
        LOGI("ProPatcher: nativeSetFineTuning: C=%.2f, S=%.2f, H=%.2f, V=%.2f, T=%.2f, W=%.2f, Sh=%.2f",
             contrast, shadows, highlights, vibrance, tint, warmth, sharpness);
    }
    pthread_mutex_unlock(&s_lut_mutex);
}

// 128-bit dual-burst NEON SIMD vector grading for Y plane (Luma)
static inline void neon_grade_y(uint8_t* y_data, uint32_t width, uint32_t height, uint32_t stride,
                                const uint8_t lut[256]) {
    uint8x16x4_t tab0 = vld1q_u8_x4(lut + 0);
    uint8x16x4_t tab1 = vld1q_u8_x4(lut + 64);
    uint8x16x4_t tab2 = vld1q_u8_x4(lut + 128);
    uint8x16x4_t tab3 = vld1q_u8_x4(lut + 192);
    uint8x16_t v64 = vdupq_n_u8(64);
    uint8x16_t v128 = vdupq_n_u8(128);
    uint8x16_t v192 = vdupq_n_u8(192);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = y_data + (size_t)y * stride;
        uint32_t x = 0;

        for (; x + 32 <= width; x += 32) {
            uint8x16_t vy0 = vld1q_u8(row + x);
            uint8x16_t vy1 = vld1q_u8(row + x + 16);

            uint8x16_t r0_0 = vqtbl4q_u8(tab0, vy0);
            uint8x16_t r0_1 = vqtbl4q_u8(tab1, vsubq_u8(vy0, v64));
            uint8x16_t r0_2 = vqtbl4q_u8(tab2, vsubq_u8(vy0, v128));
            uint8x16_t r0_3 = vqtbl4q_u8(tab3, vsubq_u8(vy0, v192));
            uint8x16_t out0 = vorrq_u8(vorrq_u8(r0_0, r0_1), vorrq_u8(r0_2, r0_3));

            uint8x16_t r1_0 = vqtbl4q_u8(tab0, vy1);
            uint8x16_t r1_1 = vqtbl4q_u8(tab1, vsubq_u8(vy1, v64));
            uint8x16_t r1_2 = vqtbl4q_u8(tab2, vsubq_u8(vy1, v128));
            uint8x16_t r1_3 = vqtbl4q_u8(tab3, vsubq_u8(vy1, v192));
            uint8x16_t out1 = vorrq_u8(vorrq_u8(r1_0, r1_1), vorrq_u8(r1_2, r1_3));

            vst1q_u8(row + x, out0);
            vst1q_u8(row + x + 16, out1);
        }

        for (; x + 16 <= width; x += 16) {
            uint8x16_t vy = vld1q_u8(row + x);
            uint8x16_t r0 = vqtbl4q_u8(tab0, vy);
            uint8x16_t r1 = vqtbl4q_u8(tab1, vsubq_u8(vy, v64));
            uint8x16_t r2 = vqtbl4q_u8(tab2, vsubq_u8(vy, v128));
            uint8x16_t r3 = vqtbl4q_u8(tab3, vsubq_u8(vy, v192));
            uint8x16_t out = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
            vst1q_u8(row + x, out);
        }

        for (; x < width; x++) {
            row[x] = lut[row[x]];
        }
    }
}

// 128-bit dual-burst NEON SIMD vector grading for semi-planar NV12/NV21 UV plane
static inline void neon_grade_uv_nv12(uint8_t* uv_data, uint32_t width, uint32_t uv_h, uint32_t stride,
                                      int16_t scale_q8, int16_t shift_even, int16_t shift_odd) {
    uint32_t uv_bytes = width & ~1u;
    if (uv_bytes > stride) uv_bytes = stride & ~1u;
    if (uv_bytes == 0) return;

    if (scale_q8 == 0 && shift_even == 0 && shift_odd == 0) {
        for (uint32_t y = 0; y < uv_h; y++) {
            memset(uv_data + (size_t)y * stride, 128, uv_bytes);
        }
        return;
    }
    if (scale_q8 == 256 && shift_even == 0 && shift_odd == 0) return;

    int16x4_t v_scale = vdup_n_s16(scale_q8);
    int16x8_t v_128 = vdupq_n_s16(128);
    int16_t shifts[8] = { shift_even, shift_odd, shift_even, shift_odd, shift_even, shift_odd, shift_even, shift_odd };
    int16x8_t v_shifts = vld1q_s16(shifts);
    int16x4_t v_shift_lo = vget_low_s16(v_shifts);
    int16x4_t v_shift_hi = vget_high_s16(v_shifts);

    for (uint32_t y = 0; y < uv_h; y++) {
        uint8_t* row = uv_data + (size_t)y * stride;
        uint32_t x = 0;

        for (; x + 32 <= uv_bytes; x += 32) {
            uint8x16_t v0 = vld1q_u8(row + x);
            uint8x16_t v1 = vld1q_u8(row + x + 16);

            int16x8_t w0_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v0))), v_128);
            int16x8_t w0_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v0))), v_128);
            int16x8_t w1_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1))), v_128);
            int16x8_t w1_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1))), v_128);

            int32x4_t p0_lo0 = vmull_s16(vget_low_s16(w0_lo), v_scale);
            int32x4_t p0_lo1 = vmull_s16(vget_high_s16(w0_lo), v_scale);
            int32x4_t s0_lo0 = vaddw_s16(vshrq_n_s32(p0_lo0, 8), v_shift_lo);
            int32x4_t s0_lo1 = vaddw_s16(vshrq_n_s32(p0_lo1, 8), v_shift_hi);
            int16x8_t r0_lo = vaddq_s16(vcombine_s16(vqmovn_s32(s0_lo0), vqmovn_s32(s0_lo1)), v_128);

            int32x4_t p0_hi0 = vmull_s16(vget_low_s16(w0_hi), v_scale);
            int32x4_t p0_hi1 = vmull_s16(vget_high_s16(w0_hi), v_scale);
            int32x4_t s0_hi0 = vaddw_s16(vshrq_n_s32(p0_hi0, 8), v_shift_lo);
            int32x4_t s0_hi1 = vaddw_s16(vshrq_n_s32(p0_hi1, 8), v_shift_hi);
            int16x8_t r0_hi = vaddq_s16(vcombine_s16(vqmovn_s32(s0_hi0), vqmovn_s32(s0_hi1)), v_128);

            int32x4_t p1_lo0 = vmull_s16(vget_low_s16(w1_lo), v_scale);
            int32x4_t p1_lo1 = vmull_s16(vget_high_s16(w1_lo), v_scale);
            int32x4_t s1_lo0 = vaddw_s16(vshrq_n_s32(p1_lo0, 8), v_shift_lo);
            int32x4_t s1_lo1 = vaddw_s16(vshrq_n_s32(p1_lo1, 8), v_shift_hi);
            int16x8_t r1_lo = vaddq_s16(vcombine_s16(vqmovn_s32(s1_lo0), vqmovn_s32(s1_lo1)), v_128);

            int32x4_t p1_hi0 = vmull_s16(vget_low_s16(w1_hi), v_scale);
            int32x4_t p1_hi1 = vmull_s16(vget_high_s16(w1_hi), v_scale);
            int32x4_t s1_hi0 = vaddw_s16(vshrq_n_s32(p1_hi0, 8), v_shift_lo);
            int32x4_t s1_hi1 = vaddw_s16(vshrq_n_s32(p1_hi1, 8), v_shift_hi);
            int16x8_t r1_hi = vaddq_s16(vcombine_s16(vqmovn_s32(s1_hi0), vqmovn_s32(s1_hi1)), v_128);

            vst1q_u8(row + x, vcombine_u8(vqmovun_s16(r0_lo), vqmovun_s16(r0_hi)));
            vst1q_u8(row + x + 16, vcombine_u8(vqmovun_s16(r1_lo), vqmovun_s16(r1_hi)));
        }

        for (; x + 16 <= uv_bytes; x += 16) {
            uint8x16_t v = vld1q_u8(row + x);
            int16x8_t w_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))), v_128);
            int16x8_t w_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))), v_128);

            int32x4_t p0 = vmull_s16(vget_low_s16(w_lo), v_scale);
            int32x4_t p1 = vmull_s16(vget_high_s16(w_lo), v_scale);
            int32x4_t s0 = vaddw_s16(vshrq_n_s32(p0, 8), v_shift_lo);
            int32x4_t s1 = vaddw_s16(vshrq_n_s32(p1, 8), v_shift_hi);
            int16x8_t r_lo = vaddq_s16(vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1)), v_128);

            int32x4_t p2 = vmull_s16(vget_low_s16(w_hi), v_scale);
            int32x4_t p3 = vmull_s16(vget_high_s16(w_hi), v_scale);
            int32x4_t s2 = vaddw_s16(vshrq_n_s32(p2, 8), v_shift_lo);
            int32x4_t s3 = vaddw_s16(vshrq_n_s32(p3, 8), v_shift_hi);
            int16x8_t r_hi = vaddq_s16(vcombine_s16(vqmovn_s32(s2), vqmovn_s32(s3)), v_128);

            vst1q_u8(row + x, vcombine_u8(vqmovun_s16(r_lo), vqmovun_s16(r_hi)));
        }

        for (; x + 1 < uv_bytes; x += 2) {
            int u = (((int)row[x] - 128) * (int)scale_q8 >> 8) + (int)shift_even + 128;
            int v = (((int)row[x + 1] - 128) * (int)scale_q8 >> 8) + (int)shift_odd + 128;
            row[x] = clamp_u8(u);
            row[x + 1] = clamp_u8(v);
        }
    }
}

// Grading for a single planar chroma plane (I420 / YV12 where pixelStride == 1)
static inline void grade_chroma_planar(uint8_t* plane_data, uint32_t plane_w, uint32_t plane_h, uint32_t stride,
                                       int16_t scale_q8, int16_t shift_c) {
    if (plane_w > stride) plane_w = stride;
    if (plane_w == 0) return;

    if (scale_q8 == 0 && shift_c == 0) {
        for (uint32_t y = 0; y < plane_h; y++) {
            memset(plane_data + (size_t)y * stride, 128, plane_w);
        }
        return;
    }
    if (scale_q8 == 256 && shift_c == 0) return;

    uint8_t lut_c[256];
    for (int i = 0; i < 256; i++) {
        int val = (((i - 128) * (int)scale_q8) >> 8) + (int)shift_c + 128;
        lut_c[i] = clamp_u8(val);
    }
    neon_grade_y(plane_data, plane_w, plane_h, stride, lut_c);
}

// Photographic color grading for RGB (3 channels) or RGBA (4 channels) captured photos
static inline void lut_grade_raster(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t stride,
                                    const uint8_t tint_weight[256],
                                    float sat, float shift_u, float shift_v, int is_monochrome) {
    int s_q8 = (int)roundf(sat * 256.0f);
    int s_r = (int)roundf(shift_v * 1.402f);
    int s_g = (int)roundf(-0.344136f * shift_u - 0.714136f * shift_v);
    int s_b = (int)roundf(shift_u * 1.772f);
    int grain_on = s_grain_enabled;

    if (!is_monochrome && !grain_on && s_q8 == 256 && s_r == 0 && s_g == 0 && s_b == 0) {
        return;
    }

    uint8x16x4_t tw0 = vld1q_u8_x4(tint_weight + 0);
    uint8x16x4_t tw1 = vld1q_u8_x4(tint_weight + 64);
    uint8x16x4_t tw2 = vld1q_u8_x4(tint_weight + 128);
    uint8x16x4_t tw3 = vld1q_u8_x4(tint_weight + 192);
    uint8x16_t v64 = vdupq_n_u8(64);
    uint8x16_t v128 = vdupq_n_u8(128);
    uint8x16_t v192 = vdupq_n_u8(192);

    int16x8_t v_sq8_base = vdupq_n_s16((int16_t)s_q8);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + (size_t)y * stride;
        uint32_t y_offset = (y * 79u) & 0x1FFu;
        uint32_t x = 0;

        if (grain_on) {
            if (is_monochrome) {
                for (; x < width; x++) {
                    uint8_t* pix = row + (size_t)x * channels;
                    uint8_t gray = (uint8_t)((pix[0] * 77 + pix[1] * 150 + pix[2] * 29) >> 8);
                    int n = s_noise_table[(y_offset + x) & 0x1FFu];
                    int mw = 255 - abs((int)gray * 2 - 255);
                    uint8_t out = clamp_u8((int)gray + ((n * mw) >> 10));
                    pix[0] = out; pix[1] = out; pix[2] = out;
                }
            } else {
                for (; x < width; x++) {
                    uint8_t* pix = row + (size_t)x * channels;
                    int r = pix[0], g = pix[1], b = pix[2];
                    int luma = (r * 77 + g * 150 + b * 29) >> 8;
                    int r_sat = luma + (((r - luma) * s_q8) >> 8);
                    int g_sat = luma + (((g - luma) * s_q8) >> 8);
                    int b_sat = luma + (((b - luma) * s_q8) >> 8);
                    int tw = tint_weight[luma];
                    int n = s_noise_table[(y_offset + x) & 0x1FFu];
                    int mw = 255 - abs(luma * 2 - 255);
                    int gr = (n * mw) >> 10;
                    pix[0] = clamp_u8(r_sat + ((s_r * tw) >> 8) + gr);
                    pix[1] = clamp_u8(g_sat + ((s_g * tw) >> 8) + gr);
                    pix[2] = clamp_u8(b_sat + ((s_b * tw) >> 8) + gr);
                }
            }
            continue;
        }

        if (is_monochrome) {
            if (channels == 3) {
                for (; x + 16 <= width; x += 16) {
                    uint8x16x3_t pix = vld3q_u8(row + (size_t)x * 3);
                    uint16x8_t luma_lo16 = vmull_u8(vget_low_u8(pix.val[0]), vdup_n_u8(77));
                    luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(pix.val[1]), vdup_n_u8(150));
                    luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(pix.val[2]), vdup_n_u8(29));
                    uint8x8_t y_lo = vshrn_n_u16(luma_lo16, 8);

                    uint16x8_t luma_hi16 = vmull_u8(vget_high_u8(pix.val[0]), vdup_n_u8(77));
                    luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(pix.val[1]), vdup_n_u8(150));
                    luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(pix.val[2]), vdup_n_u8(29));
                    uint8x8_t y_hi = vshrn_n_u16(luma_hi16, 8);
                    uint8x16_t y16 = vcombine_u8(y_lo, y_hi);

                    pix.val[0] = y16;
                    pix.val[1] = y16;
                    pix.val[2] = y16;
                    vst3q_u8(row + (size_t)x * 3, pix);
                }
            } else {
                for (; x + 16 <= width; x += 16) {
                    uint8x16x4_t pix = vld4q_u8(row + (size_t)x * 4);
                    uint16x8_t luma_lo16 = vmull_u8(vget_low_u8(pix.val[0]), vdup_n_u8(77));
                    luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(pix.val[1]), vdup_n_u8(150));
                    luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(pix.val[2]), vdup_n_u8(29));
                    uint8x8_t y_lo = vshrn_n_u16(luma_lo16, 8);

                    uint16x8_t luma_hi16 = vmull_u8(vget_high_u8(pix.val[0]), vdup_n_u8(77));
                    luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(pix.val[1]), vdup_n_u8(150));
                    luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(pix.val[2]), vdup_n_u8(29));
                    uint8x8_t y_hi = vshrn_n_u16(luma_hi16, 8);
                    uint8x16_t y16 = vcombine_u8(y_lo, y_hi);

                    pix.val[0] = y16;
                    pix.val[1] = y16;
                    pix.val[2] = y16;
                    vst4q_u8(row + (size_t)x * 4, pix);
                }
            }
            for (; x < width; x++) {
                uint8_t* pix = row + (size_t)x * channels;
                uint8_t gray = (uint8_t)((pix[0] * 77 + pix[1] * 150 + pix[2] * 29) >> 8);
                pix[0] = gray; pix[1] = gray; pix[2] = gray;
            }
        } else {
            for (; x + 16 <= width; x += 16) {
                uint8x16_t r, g, b;
                uint8x16_t a = vdupq_n_u8(255);
                if (channels == 3) {
                    uint8x16x3_t p3 = vld3q_u8(row + (size_t)x * 3);
                    r = p3.val[0]; g = p3.val[1]; b = p3.val[2];
                } else {
                    uint8x16x4_t p4 = vld4q_u8(row + (size_t)x * 4);
                    r = p4.val[0]; g = p4.val[1]; b = p4.val[2]; a = p4.val[3];
                }

                uint16x8_t luma_lo16 = vmull_u8(vget_low_u8(r), vdup_n_u8(77));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(g), vdup_n_u8(150));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(b), vdup_n_u8(29));
                uint8x8_t y_lo = vshrn_n_u16(luma_lo16, 8);

                uint16x8_t luma_hi16 = vmull_u8(vget_high_u8(r), vdup_n_u8(77));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(g), vdup_n_u8(150));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(b), vdup_n_u8(29));
                uint8x8_t y_hi = vshrn_n_u16(luma_hi16, 8);
                uint8x16_t y16 = vcombine_u8(y_lo, y_hi);

                int16x8_t eff_sq8_lo = v_sq8_base;
                int16x8_t eff_sq8_hi = v_sq8_base;

                uint8x16_t tw_r0 = vqtbl4q_u8(tw0, y16);
                uint8x16_t tw_r1 = vqtbl4q_u8(tw1, vsubq_u8(y16, v64));
                uint8x16_t tw_r2 = vqtbl4q_u8(tw2, vsubq_u8(y16, v128));
                uint8x16_t tw_r3 = vqtbl4q_u8(tw3, vsubq_u8(y16, v192));
                uint8x16_t tw16 = vorrq_u8(vorrq_u8(tw_r0, tw_r1), vorrq_u8(tw_r2, tw_r3));

                int16x8_t tw_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(tw16)));
                int16x8_t tw_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(tw16)));

                int32x4_t tint_r_lo0 = vmull_n_s16(vget_low_s16(tw_lo), (int16_t)s_r);
                int32x4_t tint_r_lo1 = vmull_n_s16(vget_high_s16(tw_lo), (int16_t)s_r);
                int16x8_t tint_r_lo = vcombine_s16(vshrn_n_s32(tint_r_lo0, 8), vshrn_n_s32(tint_r_lo1, 8));

                int32x4_t tint_r_hi0 = vmull_n_s16(vget_low_s16(tw_hi), (int16_t)s_r);
                int32x4_t tint_r_hi1 = vmull_n_s16(vget_high_s16(tw_hi), (int16_t)s_r);
                int16x8_t tint_r_hi = vcombine_s16(vshrn_n_s32(tint_r_hi0, 8), vshrn_n_s32(tint_r_hi1, 8));

                int32x4_t tint_g_lo0 = vmull_n_s16(vget_low_s16(tw_lo), (int16_t)s_g);
                int32x4_t tint_g_lo1 = vmull_n_s16(vget_high_s16(tw_lo), (int16_t)s_g);
                int16x8_t tint_g_lo = vcombine_s16(vshrn_n_s32(tint_g_lo0, 8), vshrn_n_s32(tint_g_lo1, 8));

                int32x4_t tint_g_hi0 = vmull_n_s16(vget_low_s16(tw_hi), (int16_t)s_g);
                int32x4_t tint_g_hi1 = vmull_n_s16(vget_high_s16(tw_hi), (int16_t)s_g);
                int16x8_t tint_g_hi = vcombine_s16(vshrn_n_s32(tint_g_hi0, 8), vshrn_n_s32(tint_g_hi1, 8));

                int32x4_t tint_b_lo0 = vmull_n_s16(vget_low_s16(tw_lo), (int16_t)s_b);
                int32x4_t tint_b_lo1 = vmull_n_s16(vget_high_s16(tw_lo), (int16_t)s_b);
                int16x8_t tint_b_lo = vcombine_s16(vshrn_n_s32(tint_b_lo0, 8), vshrn_n_s32(tint_b_lo1, 8));

                int32x4_t tint_b_hi0 = vmull_n_s16(vget_low_s16(tw_hi), (int16_t)s_b);
                int32x4_t tint_b_hi1 = vmull_n_s16(vget_high_s16(tw_hi), (int16_t)s_b);
                int16x8_t tint_b_hi = vcombine_s16(vshrn_n_s32(tint_b_hi0, 8), vshrn_n_s32(tint_b_hi1, 8));

                int16x8_t y_lo_16 = vreinterpretq_s16_u16(vmovl_u8(y_lo));
                int16x8_t y_hi_16 = vreinterpretq_s16_u16(vmovl_u8(y_hi));

                int16x8_t r_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r))), y_lo_16);
                int16x8_t r_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(r))), y_hi_16);
                int32x4_t r_p_lo0 = vmull_s16(vget_low_s16(r_diff_lo), vget_low_s16(eff_sq8_lo));
                int32x4_t r_p_lo1 = vmull_s16(vget_high_s16(r_diff_lo), vget_high_s16(eff_sq8_lo));
                int16x8_t r_delta_lo = vcombine_s16(vshrn_n_s32(r_p_lo0, 8), vshrn_n_s32(r_p_lo1, 8));
                int16x8_t r_out_lo = vaddq_s16(vaddq_s16(y_lo_16, r_delta_lo), tint_r_lo);
                int32x4_t r_p_hi0 = vmull_s16(vget_low_s16(r_diff_hi), vget_low_s16(eff_sq8_hi));
                int32x4_t r_p_hi1 = vmull_s16(vget_high_s16(r_diff_hi), vget_high_s16(eff_sq8_hi));
                int16x8_t r_delta_hi = vcombine_s16(vshrn_n_s32(r_p_hi0, 8), vshrn_n_s32(r_p_hi1, 8));
                int16x8_t r_out_hi = vaddq_s16(vaddq_s16(y_hi_16, r_delta_hi), tint_r_hi);
                uint8x16_t r_final = vcombine_u8(vqmovun_s16(r_out_lo), vqmovun_s16(r_out_hi));

                int16x8_t g_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g))), y_lo_16);
                int16x8_t g_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(g))), y_hi_16);
                int32x4_t g_p_lo0 = vmull_s16(vget_low_s16(g_diff_lo), vget_low_s16(eff_sq8_lo));
                int32x4_t g_p_lo1 = vmull_s16(vget_high_s16(g_diff_lo), vget_high_s16(eff_sq8_lo));
                int16x8_t g_delta_lo = vcombine_s16(vshrn_n_s32(g_p_lo0, 8), vshrn_n_s32(g_p_lo1, 8));
                int16x8_t g_out_lo = vaddq_s16(vaddq_s16(y_lo_16, g_delta_lo), tint_g_lo);
                int32x4_t g_p_hi0 = vmull_s16(vget_low_s16(g_diff_hi), vget_low_s16(eff_sq8_hi));
                int32x4_t g_p_hi1 = vmull_s16(vget_high_s16(g_diff_hi), vget_high_s16(eff_sq8_hi));
                int16x8_t g_delta_hi = vcombine_s16(vshrn_n_s32(g_p_hi0, 8), vshrn_n_s32(g_p_hi1, 8));
                int16x8_t g_out_hi = vaddq_s16(vaddq_s16(y_hi_16, g_delta_hi), tint_g_hi);
                uint8x16_t g_final = vcombine_u8(vqmovun_s16(g_out_lo), vqmovun_s16(g_out_hi));

                int16x8_t b_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))), y_lo_16);
                int16x8_t b_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b))), y_hi_16);
                int32x4_t b_p_lo0 = vmull_s16(vget_low_s16(b_diff_lo), vget_low_s16(eff_sq8_lo));
                int32x4_t b_p_lo1 = vmull_s16(vget_high_s16(b_diff_lo), vget_high_s16(eff_sq8_lo));
                int16x8_t b_delta_lo = vcombine_s16(vshrn_n_s32(b_p_lo0, 8), vshrn_n_s32(b_p_lo1, 8));
                int16x8_t b_out_lo = vaddq_s16(vaddq_s16(y_lo_16, b_delta_lo), tint_b_lo);
                int32x4_t b_p_hi0 = vmull_s16(vget_low_s16(b_diff_hi), vget_low_s16(eff_sq8_hi));
                int32x4_t b_p_hi1 = vmull_s16(vget_high_s16(b_diff_hi), vget_high_s16(eff_sq8_hi));
                int16x8_t b_delta_hi = vcombine_s16(vshrn_n_s32(b_p_hi0, 8), vshrn_n_s32(b_p_hi1, 8));
                int16x8_t b_out_hi = vaddq_s16(vaddq_s16(y_hi_16, b_delta_hi), tint_b_hi);
                uint8x16_t b_final = vcombine_u8(vqmovun_s16(b_out_lo), vqmovun_s16(b_out_hi));

                if (channels == 3) {
                    uint8x16x3_t out3;
                    out3.val[0] = r_final; out3.val[1] = g_final; out3.val[2] = b_final;
                    vst3q_u8(row + (size_t)x * 3, out3);
                } else {
                    uint8x16x4_t out4;
                    out4.val[0] = r_final; out4.val[1] = g_final; out4.val[2] = b_final; out4.val[3] = a;
                    vst4q_u8(row + (size_t)x * 4, out4);
                }
            }

            for (; x < width; x++) {
                uint8_t* pix = row + (size_t)x * channels;
                int r = pix[0], g = pix[1], b = pix[2];
                int luma = (r * 77 + g * 150 + b * 29) >> 8;
                int r_sat = luma + (((r - luma) * s_q8) >> 8);
                int g_sat = luma + (((g - luma) * s_q8) >> 8);
                int b_sat = luma + (((b - luma) * s_q8) >> 8);
                int tw = tint_weight[luma];
                pix[0] = clamp_u8(r_sat + ((s_r * tw) >> 8));
                pix[1] = clamp_u8(g_sat + ((s_g * tw) >> 8));
                pix[2] = clamp_u8(b_sat + ((s_b * tw) >> 8));
            }
        }
    }
}

// 128-bit ARM64 NEON vector grading for RGBA/RGBX viewfinder buffers (4 bytes/pixel)
static inline void neon_grade_rgba(uint8_t* data, uint32_t width, uint32_t height, uint32_t stride,
                                   const uint8_t lut[256], int16_t scale_uv, int16_t shift_u, int16_t shift_v,
                                   int is_monochrome) {
    if (stride < width * 4u) return;

    uint8x16x4_t tab0 = vld1q_u8_x4(lut + 0);
    uint8x16x4_t tab1 = vld1q_u8_x4(lut + 64);
    uint8x16x4_t tab2 = vld1q_u8_x4(lut + 128);
    uint8x16x4_t tab3 = vld1q_u8_x4(lut + 192);
    uint8x16_t v64 = vdupq_n_u8(64);
    uint8x16_t v128 = vdupq_n_u8(128);
    uint8x16_t v192 = vdupq_n_u8(192);

    if (is_monochrome) {
        for (uint32_t y = 0; y < height; y++) {
            uint8_t* row = data + (size_t)y * stride;
            uint32_t x = 0;
            for (; x + 16 <= width; x += 16) {
                uint8x16x4_t rgba = vld4q_u8(row + (size_t)x * 4);

                uint16x8_t luma_lo16 = vmull_u8(vget_low_u8(rgba.val[0]), vdup_n_u8(77));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(rgba.val[1]), vdup_n_u8(150));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(rgba.val[2]), vdup_n_u8(29));
                uint8x8_t y_lo = vshrn_n_u16(luma_lo16, 8);

                uint16x8_t luma_hi16 = vmull_u8(vget_high_u8(rgba.val[0]), vdup_n_u8(77));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(rgba.val[1]), vdup_n_u8(150));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(rgba.val[2]), vdup_n_u8(29));
                uint8x8_t y_hi = vshrn_n_u16(luma_hi16, 8);
                uint8x16_t y16 = vcombine_u8(y_lo, y_hi);

                uint8x16_t y0 = vqtbl4q_u8(tab0, y16);
                uint8x16_t y1 = vqtbl4q_u8(tab1, vsubq_u8(y16, v64));
                uint8x16_t y2 = vqtbl4q_u8(tab2, vsubq_u8(y16, v128));
                uint8x16_t y3 = vqtbl4q_u8(tab3, vsubq_u8(y16, v192));
                uint8x16_t y_out = vorrq_u8(vorrq_u8(y0, y1), vorrq_u8(y2, y3));

                rgba.val[0] = y_out;
                rgba.val[1] = y_out;
                rgba.val[2] = y_out;
                vst4q_u8(row + (size_t)x * 4, rgba);
            }
            for (; x < width; x++) {
                uint8_t* pix = row + (size_t)x * 4;
                uint8_t gray = (uint8_t)((pix[0] * 77 + pix[1] * 150 + pix[2] * 29) >> 8);
                uint8_t out = lut[gray];
                pix[0] = out; pix[1] = out; pix[2] = out;
            }
        }
        return;
    }

    int s_r = (int)roundf((float)shift_v * 1.402f);
    int s_g = (int)roundf(-0.344136f * (float)shift_u - 0.714136f * (float)shift_v);
    int s_b = (int)roundf((float)shift_u * 1.772f);
    int apply_chroma = (scale_uv != 256 || s_r != 0 || s_g != 0 || s_b != 0);

    int16x8_t v_sr = vdupq_n_s16((int16_t)s_r);
    int16x8_t v_sg = vdupq_n_s16((int16_t)s_g);
    int16x8_t v_sb = vdupq_n_s16((int16_t)s_b);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + (size_t)y * stride;
        uint32_t x = 0;

        for (; x + 16 <= width; x += 16) {
            uint8x16x4_t rgba = vld4q_u8(row + (size_t)x * 4);

            uint8x16_t r0 = vqtbl4q_u8(tab0, rgba.val[0]);
            uint8x16_t r1 = vqtbl4q_u8(tab1, vsubq_u8(rgba.val[0], v64));
            uint8x16_t r2 = vqtbl4q_u8(tab2, vsubq_u8(rgba.val[0], v128));
            uint8x16_t r3 = vqtbl4q_u8(tab3, vsubq_u8(rgba.val[0], v192));
            rgba.val[0] = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));

            uint8x16_t g0 = vqtbl4q_u8(tab0, rgba.val[1]);
            uint8x16_t g1 = vqtbl4q_u8(tab1, vsubq_u8(rgba.val[1], v64));
            uint8x16_t g2 = vqtbl4q_u8(tab2, vsubq_u8(rgba.val[1], v128));
            uint8x16_t g3 = vqtbl4q_u8(tab3, vsubq_u8(rgba.val[1], v192));
            rgba.val[1] = vorrq_u8(vorrq_u8(g0, g1), vorrq_u8(g2, g3));

            uint8x16_t b0 = vqtbl4q_u8(tab0, rgba.val[2]);
            uint8x16_t b1 = vqtbl4q_u8(tab1, vsubq_u8(rgba.val[2], v64));
            uint8x16_t b2 = vqtbl4q_u8(tab2, vsubq_u8(rgba.val[2], v128));
            uint8x16_t b3 = vqtbl4q_u8(tab3, vsubq_u8(rgba.val[2], v192));
            rgba.val[2] = vorrq_u8(vorrq_u8(b0, b1), vorrq_u8(b2, b3));

            vst4q_u8(row + (size_t)x * 4, rgba);
        }
        for (; x < width; x++) {
            uint8_t* pix = row + (size_t)x * 4;
            pix[0] = lut[pix[0]];
            pix[1] = lut[pix[1]];
            pix[2] = lut[pix[2]];
        }

        if (apply_chroma) {
            x = 0;
            for (; x + 16 <= width; x += 16) {
                uint8x16x4_t rgba = vld4q_u8(row + (size_t)x * 4);

                uint16x8_t luma_lo16 = vmull_u8(vget_low_u8(rgba.val[0]), vdup_n_u8(77));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(rgba.val[1]), vdup_n_u8(150));
                luma_lo16 = vmlal_u8(luma_lo16, vget_low_u8(rgba.val[2]), vdup_n_u8(29));
                uint8x8_t y_lo = vshrn_n_u16(luma_lo16, 8);

                uint16x8_t luma_hi16 = vmull_u8(vget_high_u8(rgba.val[0]), vdup_n_u8(77));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(rgba.val[1]), vdup_n_u8(150));
                luma_hi16 = vmlal_u8(luma_hi16, vget_high_u8(rgba.val[2]), vdup_n_u8(29));
                uint8x8_t y_hi = vshrn_n_u16(luma_hi16, 8);

                int16x8_t y_lo_16 = vreinterpretq_s16_u16(vmovl_u8(y_lo));
                int16x8_t y_hi_16 = vreinterpretq_s16_u16(vmovl_u8(y_hi));

                int16x8_t r_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba.val[0]))), y_lo_16);
                int16x8_t r_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgba.val[0]))), y_hi_16);
                int32x4_t r_p_lo0 = vmull_n_s16(vget_low_s16(r_diff_lo), scale_uv);
                int32x4_t r_p_lo1 = vmull_n_s16(vget_high_s16(r_diff_lo), scale_uv);
                int16x8_t r_delta_lo = vcombine_s16(vshrn_n_s32(r_p_lo0, 8), vshrn_n_s32(r_p_lo1, 8));
                int16x8_t r_out_lo = vaddq_s16(vaddq_s16(y_lo_16, r_delta_lo), v_sr);
                int32x4_t r_p_hi0 = vmull_n_s16(vget_low_s16(r_diff_hi), scale_uv);
                int32x4_t r_p_hi1 = vmull_n_s16(vget_high_s16(r_diff_hi), scale_uv);
                int16x8_t r_delta_hi = vcombine_s16(vshrn_n_s32(r_p_hi0, 8), vshrn_n_s32(r_p_hi1, 8));
                int16x8_t r_out_hi = vaddq_s16(vaddq_s16(y_hi_16, r_delta_hi), v_sr);
                rgba.val[0] = vcombine_u8(vqmovun_s16(r_out_lo), vqmovun_s16(r_out_hi));

                int16x8_t g_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba.val[1]))), y_lo_16);
                int16x8_t g_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgba.val[1]))), y_hi_16);
                int32x4_t g_p_lo0 = vmull_n_s16(vget_low_s16(g_diff_lo), scale_uv);
                int32x4_t g_p_lo1 = vmull_n_s16(vget_high_s16(g_diff_lo), scale_uv);
                int16x8_t g_delta_lo = vcombine_s16(vshrn_n_s32(g_p_lo0, 8), vshrn_n_s32(g_p_lo1, 8));
                int16x8_t g_out_lo = vaddq_s16(vaddq_s16(y_lo_16, g_delta_lo), v_sg);
                int32x4_t g_p_hi0 = vmull_n_s16(vget_low_s16(g_diff_hi), scale_uv);
                int32x4_t g_p_hi1 = vmull_n_s16(vget_high_s16(g_diff_hi), scale_uv);
                int16x8_t g_delta_hi = vcombine_s16(vshrn_n_s32(g_p_hi0, 8), vshrn_n_s32(g_p_hi1, 8));
                int16x8_t g_out_hi = vaddq_s16(vaddq_s16(y_hi_16, g_delta_hi), v_sg);
                rgba.val[1] = vcombine_u8(vqmovun_s16(g_out_lo), vqmovun_s16(g_out_hi));

                int16x8_t b_diff_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba.val[2]))), y_lo_16);
                int16x8_t b_diff_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgba.val[2]))), y_hi_16);
                int32x4_t b_p_lo0 = vmull_n_s16(vget_low_s16(b_diff_lo), scale_uv);
                int32x4_t b_p_lo1 = vmull_n_s16(vget_high_s16(b_diff_lo), scale_uv);
                int16x8_t b_delta_lo = vcombine_s16(vshrn_n_s32(b_p_lo0, 8), vshrn_n_s32(b_p_lo1, 8));
                int16x8_t b_out_lo = vaddq_s16(vaddq_s16(y_lo_16, b_delta_lo), v_sb);
                int32x4_t b_p_hi0 = vmull_n_s16(vget_low_s16(b_diff_hi), scale_uv);
                int32x4_t b_p_hi1 = vmull_n_s16(vget_high_s16(b_diff_hi), scale_uv);
                int16x8_t b_delta_hi = vcombine_s16(vshrn_n_s32(b_p_hi0, 8), vshrn_n_s32(b_p_hi1, 8));
                int16x8_t b_out_hi = vaddq_s16(vaddq_s16(y_hi_16, b_delta_hi), v_sb);
                rgba.val[2] = vcombine_u8(vqmovun_s16(b_out_lo), vqmovun_s16(b_out_hi));

                vst4q_u8(row + (size_t)x * 4, rgba);
            }
            for (; x < width; x++) {
                uint8_t* pix = row + (size_t)x * 4;
                int r = pix[0], g = pix[1], b = pix[2];
                int luma = (r * 77 + g * 150 + b * 29) >> 8;
                int r_sat = luma + (((r - luma) * (int)scale_uv) >> 8);
                int g_sat = luma + (((g - luma) * (int)scale_uv) >> 8);
                int b_sat = luma + (((b - luma) * (int)scale_uv) >> 8);
                pix[0] = clamp_u8(r_sat + s_r);
                pix[1] = clamp_u8(g_sat + s_g);
                pix[2] = clamp_u8(b_sat + s_b);
            }
        }
    }
}

// Safe 3-channel RGB888 viewfinder path (Format AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM = 3)
static inline void grade_rgb888(uint8_t* data, uint32_t width, uint32_t height, uint32_t stride,
                                const uint8_t lut[256], int16_t scale_uv, int16_t shift_u, int16_t shift_v,
                                int is_monochrome) {
    if (stride < width * 3u) return;
    int s_r = (int)roundf((float)shift_v * 1.402f);
    int s_g = (int)roundf(-0.344136f * (float)shift_u - 0.714136f * (float)shift_v);
    int s_b = (int)roundf((float)shift_u * 1.772f);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + (size_t)y * stride;
        for (uint32_t x = 0; x < width; x++) {
            uint8_t* pix = row + (size_t)x * 3;
            if (is_monochrome) {
                uint8_t gray = (uint8_t)((pix[0] * 77 + pix[1] * 150 + pix[2] * 29) >> 8);
                uint8_t out = lut[gray];
                pix[0] = out; pix[1] = out; pix[2] = out;
            } else {
                int r = lut[pix[0]];
                int g = lut[pix[1]];
                int b = lut[pix[2]];
                if (scale_uv != 256 || s_r != 0 || s_g != 0 || s_b != 0) {
                    int luma = (r * 77 + g * 150 + b * 29) >> 8;
                    r = clamp_u8(luma + (((r - luma) * (int)scale_uv) >> 8) + s_r);
                    g = clamp_u8(luma + (((g - luma) * (int)scale_uv) >> 8) + s_g);
                    b = clamp_u8(luma + (((b - luma) * (int)scale_uv) >> 8) + s_b);
                }
                pix[0] = (uint8_t)r; pix[1] = (uint8_t)g; pix[2] = (uint8_t)b;
            }
        }
    }
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeProcessHardwareBuffer(
    JNIEnv *env, jclass clazz, jobject hwBuffer, jint lookId) {
    (void)clazz;
    if (!hwBuffer || lookId < 0 || lookId > 12) return;

    GradingSnapshot snap;
    capture_grading_snapshot(lookId, &snap);

    if (lookId == 0 && !snap.has_fine_tuning) {
        return;
    }

    int16_t scale_uv = snap.is_monochrome ? 0 : (int16_t)roundf(snap.saturation * 256.0f);
    if (scale_uv < 0) scale_uv = 0;
    int16_t shift_u = (int16_t)roundf(snap.shift_u);
    int16_t shift_v = (int16_t)roundf(snap.shift_v);

    AHardwareBuffer* buffer = AHardwareBuffer_fromHardwareBuffer(env, hwBuffer);
    if (!buffer) return;

    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(buffer, &desc);
    if (desc.width == 0 || desc.height == 0) return;

    int is_rgba = (desc.format == FMT_RGBA_8888 || desc.format == FMT_RGBX_8888);
    int is_rgb888 = (desc.format == FMT_RGB_888);
    int is_known_yuv8 = (desc.format == FMT_YUV_420 || desc.format == FMT_NV21 ||
                         desc.format == FMT_YV12 || desc.format == FMT_IMPL_DEF);
    if (!is_rgba && !is_rgb888 && !is_known_yuv8) {
        // Unsupported format (e.g. P010 10-bit HDR, FP16, RAW, BLOB) — skip safely
        return;
    }

    // Always request CPU read + write usage so in-place reads are cache-coherent
    AHardwareBuffer_Planes planes;
    memset(&planes, 0, sizeof(planes));
    int res = AHardwareBuffer_lockPlanes(buffer,
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        -1, NULL, &planes);

    if (res != 0) {
        res = AHardwareBuffer_lockPlanes(buffer,
            AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
            -1, NULL, &planes);
    }

    uint32_t width = desc.width;
    uint32_t height = desc.height;

    if (res != 0) {
        // Fallback single-plane lock ONLY for explicit RGBA/RGB888/NV21 (never ambiguous IMPL_DEF)
        if (desc.format == FMT_IMPL_DEF || desc.format == FMT_YUV_420 || desc.format == FMT_YV12) {
            return;
        }
        void* virt = NULL;
        res = AHardwareBuffer_lock(buffer,
            AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
            -1, NULL, &virt);
        if (res != 0) {
            res = AHardwareBuffer_lock(buffer,
                AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                -1, NULL, &virt);
        }
        if (res == 0 && virt) {
            if (is_rgba && desc.stride >= width) {
                neon_grade_rgba((uint8_t*)virt, width, height, desc.stride * 4u,
                                snap.lut_y, scale_uv, shift_u, shift_v, snap.is_monochrome);
            } else if (is_rgb888 && desc.stride >= width) {
                grade_rgb888((uint8_t*)virt, width, height, desc.stride * 3u,
                             snap.lut_y, scale_uv, shift_u, shift_v, snap.is_monochrome);
            } else if (desc.format == FMT_NV21 && desc.stride >= width) {
                size_t y_size = (size_t)desc.stride * height;
                neon_grade_y((uint8_t*)virt, width, height, desc.stride, snap.lut_y);
                uint8_t* uv_data = (uint8_t*)virt + y_size;
                uint32_t uv_h = height / 2u;
                // NV21 has V first (even), U second (odd)
                if (snap.is_monochrome) {
                    uint32_t uv_w = width & ~1u;
                    for (uint32_t y = 0; y < uv_h; y++) {
                        memset(uv_data + (size_t)y * desc.stride, 128, uv_w);
                    }
                } else {
                    neon_grade_uv_nv12(uv_data, width, uv_h, desc.stride, scale_uv, shift_v, shift_u);
                }
            }
            AHardwareBuffer_unlock(buffer, NULL);
        }
        return;
    }

    if (planes.planeCount < 1 || !planes.planes[0].data || planes.planes[0].rowStride == 0) {
        AHardwareBuffer_unlock(buffer, NULL);
        return;
    }

    if (is_rgba) {
        if (planes.planes[0].rowStride >= width * 4u) {
            neon_grade_rgba((uint8_t*)planes.planes[0].data, width, height,
                            planes.planes[0].rowStride, snap.lut_y, scale_uv, shift_u, shift_v, snap.is_monochrome);
        }
    } else if (is_rgb888) {
        if (planes.planes[0].rowStride >= width * 3u) {
            grade_rgb888((uint8_t*)planes.planes[0].data, width, height,
                         planes.planes[0].rowStride, snap.lut_y, scale_uv, shift_u, shift_v, snap.is_monochrome);
        }
    } else if (planes.planeCount >= 2 && planes.planes[0].pixelStride == 1 && planes.planes[0].rowStride >= width) {
        // 8-bit 4:2:0 YUV (NV12, NV21, I420, YV12)
        neon_grade_y((uint8_t*)planes.planes[0].data, width, height,
                     planes.planes[0].rowStride, snap.lut_y);

        uint32_t uv_h = (height + 1u) / 2u;
        uint32_t uv_plane_w = (width + 1u) / 2u;

        if (planes.planes[1].data != NULL && planes.planes[1].rowStride > 0) {
            int is_interleaved = (planes.planes[1].pixelStride == 2) || (planes.planeCount == 2);
            if (is_interleaved) {
                uint32_t uv_stride = planes.planes[1].rowStride;
                uint8_t* uv_data = (uint8_t*)planes.planes[1].data;
                int16_t shift_even = shift_u;
                int16_t shift_odd = shift_v;

                // NV21: plane[2] (V) precedes plane[1] (U) by 1 byte
                if (planes.planeCount >= 3 && planes.planes[2].data != NULL &&
                    planes.planes[2].data < planes.planes[1].data) {
                    uv_data = (uint8_t*)planes.planes[2].data;
                    shift_even = shift_v;
                    shift_odd = shift_u;
                }

                if (snap.is_monochrome) {
                    uint32_t uv_bytes = width & ~1u;
                    if (uv_bytes > uv_stride) uv_bytes = uv_stride & ~1u;
                    for (uint32_t y = 0; y < uv_h; y++) {
                        memset(uv_data + (size_t)y * uv_stride, 128, uv_bytes);
                    }
                } else {
                    neon_grade_uv_nv12(uv_data, width, uv_h, uv_stride, scale_uv, shift_even, shift_odd);
                }
            } else if (planes.planeCount >= 3 &&
                       planes.planes[1].pixelStride == 1 &&
                       planes.planes[2].pixelStride == 1 &&
                       planes.planes[2].data != NULL &&
                       planes.planes[2].rowStride > 0) {
                // Planar 4:2:0 (I420: plane[1]=U, plane[2]=V; YV12: plane[1]=V, plane[2]=U)
                int16_t shift_p1 = (desc.format == FMT_YV12) ? shift_v : shift_u;
                int16_t shift_p2 = (desc.format == FMT_YV12) ? shift_u : shift_v;
                grade_chroma_planar((uint8_t*)planes.planes[1].data, uv_plane_w, uv_h,
                                    planes.planes[1].rowStride, scale_uv, shift_p1);
                grade_chroma_planar((uint8_t*)planes.planes[2].data, uv_plane_w, uv_h,
                                    planes.planes[2].rowStride, scale_uv, shift_p2);
            }
        }
    }

    AHardwareBuffer_unlock(buffer, NULL);
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeProcessRasterImage(
    JNIEnv *env, jclass clazz, jlong dataPtr, jint width, jint height, jint channels, jint stride, jint lookId) {
    (void)env; (void)clazz;
    if (dataPtr == 0 || width <= 0 || height <= 0 || (channels != 3 && channels != 4) ||
        (int64_t)stride < (int64_t)width * (int64_t)channels) {
        LOGE("nativeProcessRasterImage: invalid arguments (ptr=0x%lx, w=%d, h=%d, ch=%d, stride=%d)",
             (unsigned long)dataPtr, width, height, channels, stride);
        return;
    }

    if (lookId < 0 || lookId > 12) lookId = 0;

    GradingSnapshot snap;
    capture_grading_snapshot(lookId, &snap);

    if (lookId == 0 && !snap.has_fine_tuning && !s_grain_enabled) {
        return;
    }

    LOGI("ProPatcher: Grading Captured Photo: %dx%d (%d ch, stride=%d), Look=%d, Mono=%d, Sat=%.2f, U=%.1f, V=%.1f",
         width, height, channels, stride, lookId, snap.is_monochrome, snap.saturation, snap.shift_u, snap.shift_v);

    lut_grade_raster((uint8_t*)(uintptr_t)dataPtr, (uint32_t)width, (uint32_t)height,
                     (uint32_t)channels, (uint32_t)stride, snap.tint_weight,
                     snap.saturation, snap.shift_u, snap.shift_v, snap.is_monochrome);
}

// -----------------------------------------------------------------------------
// Front Camera Backlight & Gentle Warm Skin Tone Engine (Opt-In)
// -----------------------------------------------------------------------------
static uint8_t s_selfie_luma_lut[256];
static int s_selfie_lut_inited = 0;

static void init_selfie_luma_lut(void) {
    pthread_mutex_lock(&s_selfie_mutex);
    if (!s_selfie_lut_inited) {
        for (int y = 0; y < 256; y++) {
            // Subtle backlit shadow lift: preserves deep blacks (<16), peaks +8 around y=80, tapers to 0 at y=185
            float lift = 0.0f;
            if (y > 16 && y < 185) {
                float norm = (float)(y - 16) / (185.0f - 16.0f);
                lift = 8.0f * sinf(norm * 3.14159265f);
            }
            s_selfie_luma_lut[y] = clamp_u8((int)roundf((float)y + lift));
        }
        s_selfie_lut_inited = 1;
    }
    pthread_mutex_unlock(&s_selfie_mutex);
}

static inline float smoothstep01(float edge0, float edge1, float x) {
    if (x <= edge0) return 0.0f;
    if (x >= edge1) return 1.0f;
    float t = (x - edge0) / (edge1 - edge0);
    return t * t * (3.0f - 2.0f * t);
}

static void grade_front_backlight_raster(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t stride) {
    if (!data || width == 0 || height == 0 || (channels != 3 && channels != 4) ||
        (uint64_t)stride < (uint64_t)width * (uint64_t)channels) {
        return;
    }
    init_selfie_luma_lut();

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + (size_t)y * stride;
        for (uint32_t x = 0; x < width; x++) {
            uint8_t* pix = row + (size_t)x * channels;
            int r = pix[0];
            int g = pix[1];
            int b = pix[2];

            // 1. Gentle shadow recovery in backlit mid-shadows
            int luma = (r * 77 + g * 150 + b * 29) >> 8;
            if (luma > 16) {
                int luma_lifted = (int)s_selfie_luma_lut[luma];
                if (luma_lifted != luma) {
                    r = clamp_u8((r * luma_lifted + (luma >> 1)) / luma);
                    g = clamp_u8((g * luma_lifted + (luma >> 1)) / luma);
                    b = clamp_u8((b * luma_lifted + (luma >> 1)) / luma);
                    luma = (r * 77 + g * 150 + b * 29) >> 8;
                }
            }

            // 2. Continuous soft skin-tone mask requiring genuine warm chroma
            // Neutral greys/whites (walls, shirts, clouds) have w_skin = 0 -> untouched
            int max_c = r > g ? (r > b ? r : b) : (g > b ? g : b);
            int min_c = r < g ? (r < b ? r : b) : (g < b ? g : b);
            int chroma = max_c - min_c;

            if (r > g && r > b && chroma >= 12 && luma >= 28 && luma <= 232) {
                float w_luma = smoothstep01(28.0f, 52.0f, (float)luma) *
                               (1.0f - smoothstep01(205.0f, 232.0f, (float)luma));
                float w_rg = smoothstep01(4.0f, 18.0f, (float)(r - g));
                float w_rb = smoothstep01(10.0f, 28.0f, (float)(r - b));
                float w_gb = smoothstep01(-6.0f, 10.0f, (float)(g - b));
                float w_chroma = smoothstep01(12.0f, 26.0f, (float)chroma);
                float w_skin = w_luma * w_rg * w_rb * w_gb * w_chroma;

                if (w_skin > 0.01f) {
                    // Subtle warm peach bias + gentle 6% saturation boost (no hard jumps, no double warming)
                    float sat = 1.0f + 0.06f * w_skin;
                    float rf = (float)luma + ((float)(r - luma) * sat) + (4.0f * w_skin);
                    float gf = (float)luma + ((float)(g - luma) * sat) + (1.0f * w_skin);
                    float bf = (float)luma + ((float)(b - luma) * sat) - (2.0f * w_skin);
                    r = clamp_u8((int)roundf(rf));
                    g = clamp_u8((int)roundf(gf));
                    b = clamp_u8((int)roundf(bf));
                }
            }

            // 3. Soft highlight shoulder above 232
            if (luma > 228) {
                if (r > 236) r = 236 + (((r - 236) * 15) / 20);
                if (g > 236) g = 236 + (((g - 236) * 15) / 20);
                if (b > 236) b = 236 + (((b - 236) * 15) / 20);
            }

            pix[0] = (uint8_t)r;
            pix[1] = (uint8_t)g;
            pix[2] = (uint8_t)b;
        }
    }
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeProcessFrontBacklightImage(
    JNIEnv *env, jclass clazz, jlong dataPtr, jint width, jint height, jint channels, jint stride) {
    (void)env; (void)clazz;
    if (dataPtr == 0 || width <= 0 || height <= 0 || (channels != 3 && channels != 4) ||
        (int64_t)stride < (int64_t)width * (int64_t)channels) {
        LOGE("nativeProcessFrontBacklightImage: invalid arguments (ptr=0x%lx, w=%d, h=%d, ch=%d, stride=%d)",
             (unsigned long)dataPtr, width, height, channels, stride);
        return;
    }

    LOGI("ProPatcher: Applying Gentle Front Backlight / Skin Tone Grading: %dx%d (%d ch, stride=%d)",
         width, height, channels, stride);

    grade_front_backlight_raster((uint8_t*)(uintptr_t)dataPtr, (uint32_t)width, (uint32_t)height,
                                 (uint32_t)channels, (uint32_t)stride);
}

// -----------------------------------------------------------------------------
// Bento HDR+ Highlight Recovery Green-Cast & Black-Level Pedestal Normalizer
// -----------------------------------------------------------------------------
// On pre-Pixel-8 sensors (e.g., Pixel 7 Pro 'cheetah' Samsung GN1), forcing
// ShotParams_bento_enabled_set(true) activates Gcam's linear highlight recovery
// stage without p23+ Bento green-channel equilibrium coefficients in gcam::Tuning,
// which can leave an uncompensated green gain + shadow pedestal (G > (R+B)/2).
// This self-calibrating two-pass filter samples the near-neutral locus:
// - If the frame is already neutral (delta_g <= 2.5 or ratio_g <= 1.025), it is a no-op.
// - If a green cast is present on the neutral locus, it normalizes G via an integer
//   LUT bounded by 0.918 * max(0, G - (R+B)/2) so highlights never shift magenta.
static void grade_bento_awb_raster(uint8_t* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t stride) {
    if (!data || width < 16 || height < 16 || (channels != 3 && channels != 4) ||
        (uint64_t)stride < (uint64_t)width * (uint64_t)channels) {
        return;
    }

    uint64_t total_samples = 0;
    uint64_t match_count = 0;
    int64_t sum_rb = 0;
    int64_t sum_g = 0;
    int64_t sum_excess = 0;
    uint64_t shadow_count = 0;
    int64_t sum_shadow_excess = 0;

    for (uint32_t y = 0; y < height; y += 4) {
        const uint8_t* row = data + (size_t)y * stride;
        for (uint32_t x = 0; x < width; x += 4) {
            const uint8_t* pix = row + (size_t)x * channels;
            int r = (int)pix[0];
            int g = (int)pix[1];
            int b = (int)pix[2];
            int rb = (r + b) >> 1;
            int rb_diff = r >= b ? (r - b) : (b - r);
            int g_excess = g - rb;
            total_samples++;

            if (rb_diff <= 18 && rb >= 25 && rb <= 210 &&
                g_excess >= -4 && g_excess <= 26 && (g * 100 <= r * 132)) {
                match_count++;
                sum_rb += rb;
                sum_g += g;
                sum_excess += g_excess;
                if (rb < 60) {
                    shadow_count++;
                    sum_shadow_excess += g_excess;
                }
            }
        }
    }

    if (match_count < 64 || (match_count * 100ULL) < (total_samples * 8ULL)) {
        return;
    }

    float mean_rb = (float)sum_rb / (float)match_count;
    float mean_g = (float)sum_g / (float)match_count;
    float delta_g = (float)sum_excess / (float)match_count;
    float ratio_g = mean_g / (mean_rb > 1.0f ? mean_rb : 1.0f);

    if (delta_g <= 2.5f || ratio_g <= 1.025f) {
        return;
    }

    float shadow_dg = (shadow_count >= 16)
        ? ((float)sum_shadow_excess / (float)shadow_count)
        : delta_g;

    float clamped_ratio = ratio_g < 1.18f ? ratio_g : 1.18f;
    float inv_gain = 1.0f / clamped_ratio;
    float drop_factor = 1.0f - inv_gain;
    float ped_amp = shadow_dg - (45.0f * drop_factor);
    if (ped_amp < 0.0f) ped_amp = 0.0f;

    int lut[256];
    for (int gv = 0; gv < 256; gv++) {
        float gain_drop = (float)gv * drop_factor;
        float t1 = (float)gv / 24.0f;
        if (t1 < 0.0f) t1 = 0.0f;
        if (t1 > 1.0f) t1 = 1.0f;
        float t2 = (110.0f - (float)gv) / 70.0f;
        if (t2 < 0.0f) t2 = 0.0f;
        if (t2 > 1.0f) t2 = 1.0f;
        float ped = ped_amp * t1 * t2;
        int d = (int)(gain_drop + ped + 0.5f);
        if (d < 0) d = 0;
        if (d > 64) d = 64;
        lut[gv] = d;
    }

    LOGI("ProPatcher: Bento AWB green-cast normalizer active (delta_g=+%.2f, ratio_g=%.4f, shadow_dg=+%.2f)",
         delta_g, ratio_g, shadow_dg);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + (size_t)y * stride;
        for (uint32_t x = 0; x < width; x++) {
            uint8_t* pix = row + (size_t)x * channels;
            int g = (int)pix[1];
            int rb = ((int)pix[0] + (int)pix[2]) >> 1;
            if (g > rb) {
                int diff = g - rb;
                int max_drop = (diff * 235) >> 8;
                int des_drop = lut[g];
                int act_drop = des_drop < max_drop ? des_drop : max_drop;
                pix[1] = clamp_u8(g - act_drop);
            }
        }
    }
}

JNIEXPORT void JNICALL
Java_com_google_android_patch_cameralooks_TomteInitHelper_nativeProcessBentoAwbImage(
    JNIEnv *env, jclass clazz, jlong dataPtr, jint width, jint height, jint channels, jint stride) {
    (void)env; (void)clazz;
    if (dataPtr == 0 || width <= 0 || height <= 0 || (channels != 3 && channels != 4) ||
        (int64_t)stride < (int64_t)width * (int64_t)channels) {
        return;
    }
    grade_bento_awb_raster((uint8_t*)(uintptr_t)dataPtr, (uint32_t)width, (uint32_t)height,
                           (uint32_t)channels, (uint32_t)stride);
}

