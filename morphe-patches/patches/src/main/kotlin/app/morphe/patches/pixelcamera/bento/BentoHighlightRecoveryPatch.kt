package app.morphe.patches.pixelcamera.bento

import app.morphe.patcher.patch.bytecodePatch
import app.morphe.patches.pixelcamera.PixelCameraPatchUtils
import app.morphe.patches.pixelcamera.looks.cameraLooksPatch

val bentoHighlightRecoveryPatch = bytecodePatch(
    name = "Bento HDR+ Highlight Recovery (Experimental)",
    description = "Unlocks Google's Bento HDR+ highlight recovery pipeline and configurable Max HDR Ratio AE override on Pixel 7 Pro and other Pixel devices via ShotParams and AeShotParams, with in-app toggles in Advanced Settings and safe EdgeTPU bento_edamame isolation.",
    default = true
) {
    extendWith("TomteInitHelper.dex")
    dependsOn(cameraLooksPatch)

    compatibleWith(
        "com.google.android.GoogleCamera" to setOf("11.1.040.982810059.19")
    )

    execute {
        // Bento HDR+ highlight recovery (ShotParams_bento_enabled_set), EdgeTPU bento_edamame isolation,
        // and Max HDR Ratio AE override (AeShotParams_max_hdr_ratio_override_set) are applied safely
        // inside TomteInitHelper.applyShotParamsDenoiseAndSharpen(ShotParams), which CameraLooksPatch
        // invokes at the end of mtt.K(ShotParams). Avoiding additional instruction insertions inside
        // mtt.K / mtt.L preserves exact Dalvik register verification across all Android ART versions.
        mutableClassDefByOrNull("Lksf;")?.let { clazz ->
            PixelCameraPatchUtils.hookKlmFlags(clazz)
        }
    }
}
