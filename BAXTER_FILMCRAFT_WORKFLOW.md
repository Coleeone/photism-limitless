# Baxter & Layne Video FX Bridge: Photism Limitless CLI

`photism-cli.exe` is a high-performance headless GPU tool that runs all 21 Photism Limitless effect shaders on video files using your NVIDIA GPU (RTX 5080/4090/3080).

It allows **AI agents (like Baxter or Layne's agent)** and automated editing scripts to apply audio-reactive visual effects, transitions, and distortions directly to video clips before or after importing them into **FilmCraft**, **CapCut**, **Resolve**, or **Premiere Pro**.

---

## 1. Quick Example Commands

### Basic Effect Application
Process a clip using the **Droste** spiral effect:
```bash
photism-cli -i input.mp4 -o output.mp4 --mode droste
```

### Liquid Fire Smear with Echo Trails
```bash
photism-cli -i dance_clip.mp4 -o smear_output.mp4 --mode smear --trails 90 --heat 75 --chaos 40
```

### Glitch / Slice / VHS Look
```bash
photism-cli -i intro.mp4 -o vhs_glitch.mp4 --mode vhs --slice 80 --twist 30 --contrast 90
```

### Kaleidoscope Symmetry
```bash
photism-cli -i drop.mp4 -o kaleidoscope.mp4 --mode infinite --symmetry 6 --tiles 2
```

---

## 2. All Available Modes (`--mode <name>`)

1. `none` - Clean canvas with global color & lens tuning
2. `smear` - Liquid fire advection flow field
3. `droste` - Recursive logarithmic spiral
4. `infinite` - Zoom tunnel feedback recursion
5. `corridor` - Perspective geometry depth tunnel
6. `cube` - 3D isometric voxel reflection
7. `planet` - Spherical fisheye warp & rim glow
8. `windows` - Nested UI frame cascading feedback
9. `melt` - Viscous thermal gravity drip
10. `shatter` - Voronoi cellular glass fracture
11. `prism` - Chromatic aberration dispersion
12. `terrain` - Height-mapped scan displacement
13. `breathe` - Radial sine wave expansion
14. `mosh` - Datamosh macroblock vector tearing
15. `sort` - Pixel luma sorting drift
16. `coral` - Reaction-diffusion organic cellular growth
17. `scanline` - CRT cathode raster lines
18. `thresh` - High-contrast black & white comic threshold
19. `solar` - Sabattier edge inversion & neon fringes
20. `halftone` - CMYK retro print dot raster
21. `vhs` - Magnetic tape jitter, head-switching noise & bleed

---

## 3. How Baxter & AI Agents Use It in FilmCraft

Because **FilmCraft CLI** supports headless project manipulation (`filmcraft-cli import`, `filmcraft-cli export`), Baxter can execute an automated styling recipe:

1. **Baxter selects a clip** in the FilmCraft timeline that needs an intro hook or transition.
2. **Baxter runs `photism-cli`**:
   ```bash
   photism-cli -i raw_clip.mp4 -o stylized_clip.mp4 --mode droste --tiles 2
   ```
3. **Baxter imports the stylized clip back into FilmCraft**:
   ```bash
   filmcraft-cli --project my_project.fcproj import stylized_clip.mp4
   ```
4. **Baxter places the stylized cut** on the beat or across a transition.

---

## 4. Hardware Acceleration
- Automatically detects and runs on NVIDIA CUDA (Ampere, Ada, Blackwell sm_120).
- Uses NVENC hardware encoding (`h264_nvenc`) for export speeds of 150+ FPS on 1080p vertical video.
