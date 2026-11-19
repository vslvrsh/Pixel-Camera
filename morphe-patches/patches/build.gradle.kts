group = "app.morphe.patches.pixelcamera"
version = "1.1.4-p7p-bento"

patches {
    about {
        name = "Pixel Camera Looks, Bento HDR+ & Creator Patches"
        description = "Unlocks Camera Looks, Experimental Bento HDR+ Highlight Recovery, Viewfinder Controls, 10x Zoom, Pro 50MP & Creator Suite on Pixel 6-10 (Pixel 7 Pro tuned)"
        source = "https://github.com/vslvrsh/Pixel-Camera"
        author = "Akshay Kadam / vslvrsh fork"
        contact = "na"
        website = "https://github.com/vslvrsh/Pixel-Camera"
        license = "GPLv3"
    }
}

val patchListGeneratorClasspath = configurations.create("patchListGeneratorClasspath")

dependencies {
    compileOnly(libs.gson)
    compileOnly(libs.morphe.patcher)
    compileOnly(libs.smali)
    compileOnly("com.github.REAndroid:arsclib:a28c6fb2a7")
    patchListGeneratorClasspath(libs.gson)

    testImplementation(kotlin("test"))
    testImplementation(libs.morphe.patcher)
    testImplementation(libs.smali)
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
}

tasks.named("jar") {
    finalizedBy("buildAndroid")
}

tasks.named("assemble") {
    dependsOn("buildAndroid")
}

