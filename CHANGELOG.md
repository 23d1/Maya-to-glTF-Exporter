# Changelog

All notable changes to Maya GLTF Exporter will be documented in this file.

## [1.1.1] - January 2026

### Fixed
- **GLTF Validation Errors** - Fixed EMPTY_ENTITY errors
  - Removed empty `animations` array when no animation is exported
  - Removed empty `skins` array when no skeletal animation is present
  - Removed empty `extensionsUsed` array when no extensions are used
  - Files now pass Khronos GLTF Validator without errors

### Technical
- Added `cleanup_empty_arrays()` function called before file write
- Improved GLTF spec compliance
- Normal maps use runtime tangent generation (standard practice, supported by all major viewers)

## [1.1.0] - January 2026

### Added
- **KHR_materials_clearcoat extension support**
  - Exports coat/clearcoat from openPBRSurface (coatWeight, coatRoughness, coatColor)
  - Exports coat from standardSurface (coat, coatRoughness, coatColor, coatNormal)
  - Includes clearcoat texture support
  - Includes clearcoat normal map support
  
- **KHR_materials_sheen extension support**
  - Exports fuzz/sheen from openPBRSurface (fuzzWeight, fuzzColor, fuzzRoughness)
  - Exports sheen from standardSurface (sheen, sheenColor, sheenRoughness)
  - Includes sheen texture support

- **Extension tracking**
  - Automatically adds `extensionsUsed` to GLTF output
  - Console output shows which extensions are being used

### Fixed
- Normal maps now export correctly (was missing in v1.0.0)
- Metallic/roughness textures now export (was only exporting values)
- Emission textures and values now export correctly
- Opacity/transparency now exports with proper alpha blend mode

### Technical
- Clearcoat maps to GLTF KHR_materials_clearcoat extension
- Sheen/fuzz maps to GLTF KHR_materials_sheen extension
- Both openPBRSurface and standardSurface support these extensions

## [1.0.1] - January 2026

### Fixed
- Restored full texture channel export (normal, metallic, roughness, emission)
- Fixed missing normal map export
- Fixed missing metallic/roughness texture export
- Enhanced standardSurface material conversion with all texture channels

## [1.0.0] - January 2026

### Added
- Initial release
- openPBRSurface material support
- standardSurface material support
- Legacy material support (Lambert, Blinn, Phong)
- Full animation export with frame baking
- Skeletal animation (skinClusters)
- Texture embedding in GLB format
- Correct pivot point handling
- Timeline and custom frame range controls
- GLB and GLTF format export
- Selection-only export
- Force bake all option

### Features
- Transform animation (translation, rotation, scale)
- Smooth vertex normals
- Material with PBR workflow
- Texture support (base color, metallic, roughness, normal, emission, opacity)