[app]
title = AlphaTrainer
package.name = alphatrainer
package.domain = org.alphatechnologies
source.dir = src
source.include_exts = py,txt,json,csv,md
version = 1.2.0
requirements = python3,kivy,pyjnius
orientation = portrait
fullscreen = 0
android.api = 35
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a
android.accept_sdk_license = True
android.permissions = INTERNET

[buildozer]
log_level = 2
warn_on_root = 1
