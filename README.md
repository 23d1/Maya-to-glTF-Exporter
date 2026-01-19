# Maya GLTF/GLB Exporter v1.1.1

GLTF 2.0 exporter for Autodesk Maya 2026+

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Maya](https://img.shields.io/badge/Maya-2026%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

![screenshot](Screenshot.png)

## Features

### 🎨 Materials
- **openPBRSurface** - Full support for OpenPBR Surface shader
  - Base color, metalness, roughness, coat, fuzz
  - Normal maps
  - Emission
  - Opacity/transparency
- **standardSurface** - Maya's standard surface shader
- **Legacy Materials** - Lambert, Blinn, Phong

### 🎬 Animation
- **Transform Animation** - Position, rotation, scale keyframes
- **Frame Baking** - Bake animation at any frame rate
- **Timeline Control** - Use Maya timeline or custom frame ranges
- **Skeletal Animation** - Full skinCluster support with up to 4 bone influences per vertex
- **Correct Pivot Handling** - Properly exports Maya pivot points, even with rotated parent hierarchies

### 📦 Export Options
- **GLB Format** - Single binary file with embedded textures
- **GLTF Format** - JSON + separate .bin and texture files
- **Texture Embedding** - Automatic texture embedding in GLB format
- **Smooth Normals** - Proper vertex normal export for smooth shading
- **Selection Export** - Export selected objects only

## Installation

### Quick Start

1. Download `maya_gltf_exporter_v1.1.1.py`
2. Open Maya
3. Open the **Script Editor** (Windows > General Editors > Script Editor)
4. Switch to the **Python** tab
5. **Copy and paste** the entire script
6. **Execute** (Ctrl+Enter or Command+Return)
7. The exporter UI will appear

### Optional: Add to Maya Shelf

1. Follow steps 1-5 above
2. In the Script Editor, select all the script text
3. **Middle-mouse drag** the selected text to your shelf
4. A new shelf button will be created
5. Click the button anytime to launch the exporter

## Usage

### Basic Export

1. Launch the exporter (run the script)
2. Choose **GLB** or **GLTF** format
3. Click **Browse** to choose output location
4. Click **EXPORT**

### Animation Export

1. Check **"Export Animation"**
2. Choose timeline or custom range:
   - **Use Timeline Range** - Uses Maya's playback range
   - **Custom Range** - Specify start/end frames manually
3. *Optional:* Check **"Bake Animation"** if you want to force baking without keyframes
4. Set **Sample Every N Frames** (default: 1 = every frame)
5. Click **EXPORT**

### Advanced Options

**Export Selected Only**
- Only exports currently selected objects
- Useful for exporting specific parts of your scene

**Force Bake All**
- Bakes animation for all transforms, even without keyframes
- Use when animation is driven by expressions, constraints, etc.

**Sample Rate**
- Controls how often to sample animation
- 1 = every frame (smooth, larger file)
- 2 = every other frame (smaller file, slightly less smooth)
- Higher values = smaller files but choppier animation

## Material Mapping

### openPBRSurface → GLTF PBR

| Maya (openPBRSurface) | GLTF (pbrMetallicRoughness) |
|----------------------|----------------------------|
| baseColor × baseWeight | baseColorFactor / baseColorTexture |
| baseMetalness | metallicFactor |
| specularRoughness | roughnessFactor |
| geometryNormal | normalTexture |
| emissionColor × emissionLuminance | emissiveFactor / emissiveTexture |
| geometryOpacity | alpha channel + alphaMode |

### standardSurface → GLTF PBR

| Maya (standardSurface) | GLTF (pbrMetallicRoughness) |
|----------------------|----------------------------|
| baseColor | baseColorFactor / baseColorTexture |
| metalness | metallicFactor |
| specularRoughness | roughnessFactor |

### Legacy Materials

Lambert, Blinn, and Phong materials are converted to basic PBR:
- Color → baseColorFactor
- Metalness set to 0
- Roughness: 1.0 (Lambert), 0.5 (Blinn/Phong)

## Technical Details

### File Structure

**GLB (Binary)**
```
file.glb (single file)
├─ JSON chunk (GLTF structure)
└─ Binary chunk (geometry + textures)
```

**GLTF (JSON)**
```
file.gltf (JSON structure)
file.bin (binary geometry data)
texture1.png (external texture)
texture2.jpg (external texture)
...
```

### Pivot Point Handling

The exporter correctly handles Maya's pivot points by:
1. Reading the world transform matrix (accurate with rotated parents)
2. Offsetting mesh vertices by the negative local pivot
3. Positioning the GLTF node at world_position + local_pivot
4. This ensures rotation happens around the correct point in GLTF viewers

### Coordinate System

Maya uses a **Y-up, right-handed** coordinate system, which matches GLTF 2.0 specifications. No coordinate conversion is needed.

## Compatibility

### Tested With
- **Maya 2026** (primary target)
- **Three.js** - Full compatibility
- **Babylon.js** - Full compatibility
- **glTF Viewer** (https://gltf-viewer.donmccurdy.com/)
- **Blender** - Import tested successfully

### GLTF Version
- Exports **GLTF 2.0** specification
- Uses core GLTF features (no extensions required)

## Troubleshooting

### Issue: Textures not showing
**Solution:** 
- Ensure texture files exist at the paths specified in Maya
- For GLB format, textures must be PNG or JPEG
- Check that file paths don't contain special characters

### Issue: Animation not exporting
**Solution:**
- Make sure "Export Animation" is checked
- Verify objects have keyframes (or use "Force Bake All")
- Check that the timeline range is correct
- Ensure animated objects are actually exported (not filtered out)

### Issue: Mesh appears in wrong position
**Solution:**
- Check for frozen transforms in Maya
- Verify parent hierarchy doesn't have unexpected rotations
- Try centering pivot (Modify > Center Pivot) if pivot is intentionally offset

### Issue: Normals look faceted/flat
**Solution:**
- In Maya: Mesh Display > Soften Edge
- The exporter exports smooth vertex normals when available

### Issue: Scale warnings during export
**Solution:**
- These warnings are harmless - Maya falls back to relative scale
- The export will work correctly despite the warnings

## Limitations

### Not Currently Supported
- **Blend shapes** - Planned for future version
- **Multiple UV sets** - Only UV set 0 is exported
- **Vertex colors** - Not yet implemented
- **GLTF Extensions** - Uses core GLTF only (no KHR_materials_clearcoat, etc.)
- **Cameras and Lights** - Geometry and materials only

### Material Limitations
Some openPBRSurface features cannot be represented in core GLTF:
- Subsurface scattering
- Coat/clearcoat (requires extension)
- Fuzz/sheen (requires extension)
- Thin film

## Performance Tips

### For Faster Exports
- Export selected objects only when possible
- Use lower sample rates for animation (2-3 frames)
- Keep texture resolutions reasonable (2K max for real-time use)

### For Smaller Files
- Use JPEG textures instead of PNG where possible
- Reduce animation sample rate
- Remove unused UVs and vertex data in Maya before export

## Examples

### Basic Static Model
```python
# 1. Create or open your model in Maya
# 2. Run the exporter script
# 3. Select GLB format
# 4. Browse to output location
# 5. Click EXPORT
```

### Animated Character
```python
# 1. Rig and animate your character in Maya
# 2. Run the exporter script
# 3. Check "Export Animation"
# 4. Select "Use Timeline Range"
# 5. Set sample rate to 1 (every frame)
# 6. Click EXPORT
```

### Custom Frame Range
```python
# Export only frames 50-100 of a longer animation
# 1. Run exporter
# 2. Check "Export Animation"  
# 3. Select "Custom Range"
# 4. Set Start Frame: 50, End Frame: 100
# 5. Click EXPORT
```

## Credits

**Created with assistance from:** Claude (Anthropic AI)  
**Development Date:** January 2026  
**License:** MIT License  

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

For issues, questions, or contributions, please refer to the script's comments and documentation.

## Version History

- [**CHANGELOG.md**](CHANGELOG.md)
