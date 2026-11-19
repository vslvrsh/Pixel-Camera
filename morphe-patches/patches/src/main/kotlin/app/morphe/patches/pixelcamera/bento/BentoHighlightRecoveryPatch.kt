package app.morphe.patches.pixelcamera.bento

import app.morphe.patcher.patch.bytecodePatch
import app.morphe.patcher.util.smali.toInstructions
import app.morphe.patches.pixelcamera.PixelCameraPatchUtils
import app.morphe.patches.pixelcamera.looks.cameraLooksPatch
import com.android.tools.smali.dexlib2.Opcode
import com.android.tools.smali.dexlib2.iface.instruction.FiveRegisterInstruction
import com.android.tools.smali.dexlib2.iface.instruction.OneRegisterInstruction
import com.android.tools.smali.dexlib2.iface.instruction.ReferenceInstruction
import com.android.tools.smali.dexlib2.iface.instruction.RegisterRangeInstruction

val bentoHighlightRecoveryPatch = bytecodePatch(
    name = "Bento HDR+ Highlight Recovery (Experimental)",
    description = "Unlocks Google's Bento HDR+ highlight recovery pipeline and configurable Max HDR Ratio AE override on Pixel 7 Pro and other Pixel devices, with in-app toggles in Advanced Settings and safe EdgeTPU bento_edamame isolation.",
    default = true
) {
    extendWith("TomteInitHelper.dex")
    dependsOn(cameraLooksPatch)

    compatibleWith(
        "com.google.android.GoogleCamera" to setOf("11.1.040.982810059.19")
    )

    execute {
        // 1. Ensure feature flag hooks are present on Lksf;
        mutableClassDefByOrNull("Lksf;")?.let { clazz ->
            PixelCameraPatchUtils.hookKlmFlags(clazz)
            PixelCameraPatchUtils.hookKlmFlagR(clazz)
        }

        // 2. Hook ShotParams_bento_enabled_set & AeShotParams factory methods in Lmtt; (11.1) / Lmla; (11.0)
        for (shotBuilderDesc in listOf("Lmtt;", "Lmla;")) {
            mutableClassDefByOrNull(shotBuilderDesc)?.let { clazz ->
                for (method in clazz.methods) {
                    val impl = method.implementation ?: continue

                    // 2a. Route every ShotParams_bento_enabled_set(J, ShotParams, Z) boolean through filterBentoEnabled(Z)Z
                    val bentoCallIndices = impl.instructions.mapIndexedNotNull { idx, ins ->
                        val ref = (ins as? ReferenceInstruction)?.reference?.toString() ?: ""
                        if ((ins.opcode == Opcode.INVOKE_STATIC || ins.opcode == Opcode.INVOKE_STATIC_RANGE) &&
                            ref.contains("GcamModuleJNI;->ShotParams_bento_enabled_set(JLcom/google/googlex/gcam/ShotParams;Z)V")
                        ) {
                            idx to ins
                        } else null
                    }
                    for ((callIdx, callIns) in bentoCallIndices.reversed()) {
                        val boolReg = when (callIns) {
                            is FiveRegisterInstruction -> callIns.registerF
                            is RegisterRangeInstruction -> callIns.startRegister + 3
                            else -> null
                        } ?: continue

                        val invokeStr = PixelCameraPatchUtils.invokeStaticSingleReg(
                            boolReg,
                            "Lcom/google/android/patch/cameralooks/TomteInitHelper;->filterBentoEnabled(Z)Z"
                        )
                        val hookSmali = """
                            $invokeStr
                            move-result v$boolReg
                        """.trimIndent()
                        try {
                            val hookIns = hookSmali.toInstructions(method)
                            var pos = callIdx
                            for (ni in hookIns) {
                                impl.addInstruction(pos++, ni)
                            }
                        } catch (_: Throwable) {}
                    }

                    // 2b. Apply Max HDR Ratio override before returning AeShotParams
                    if (method.returnType == "Lcom/google/googlex/gcam/AeShotParams;") {
                        val retIndices = impl.instructions.mapIndexedNotNull { idx, ins ->
                            if (ins.opcode == Opcode.RETURN_OBJECT) idx to ins else null
                        }
                        for ((retIdx, retIns) in retIndices.reversed()) {
                            val aeReg = (retIns as? OneRegisterInstruction)?.registerA ?: continue
                            val invokeStr = PixelCameraPatchUtils.invokeStaticSingleReg(
                                aeReg,
                                "Lcom/google/android/patch/cameralooks/TomteInitHelper;->applyBentoAeShotParams(Lcom/google/googlex/gcam/AeShotParams;)V"
                            )
                            try {
                                val hookIns = invokeStr.toInstructions(method)
                                var pos = retIdx
                                for (ni in hookIns) {
                                    impl.addInstruction(pos++, ni)
                                }
                            } catch (_: Throwable) {}
                        }
                    }
                }
            }
        }
    }
}
