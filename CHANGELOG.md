# Changelog

## [4.0.8] - January 2026

### Fixed
- **Local pivot handling** - Correctly compensates for parent and self pivots in hierarchies
- Fixed issue where local pivots shifted to parent object's pivot upon export
- Updated translation calculation: `glTF Translation = Maya Local Translation - Parent Pivot + Self Pivot`
- Applied pivot compensation to both static meshes and animated transforms (frame baking)

---

## [3.0.1] - January 2026 ⭐ STABLE RELEASE

### Fixed
- **Animation rotation axis corrected** - Animated meshes now rotate on correct axis regardless of parent transforms
- Vertices un-rotated to identity orientation for animated objects
- Animation applies world-space rotation from clean state

### Technical
- Un-rotation matrix applied to animated mesh vertices
- Enables correct world-space rotation animation
- Static meshes unchanged (world-space baking only)

---

## [3.0.0] - January 2026

### Changed
- **Pure world-space animation baking** - Samples mesh center + rotation at every frame
- No more pivot offset math
- Captures actual motion as seen in Maya

---

## [2.0.x] - January 2026

### Added
- World-space vertex baking for any hierarchy
- Automatic parent transform handling

### Fixed
- 90° rotation artifacts with parent groups
- World-space rotation extraction from matrices

---

## [1.1.x] - January 2026

### Added
- KHR_materials_clearcoat extension
- KHR_materials_sheen extension
- Normal map, metallic/roughness texture export

### Fixed
- GLTF validation errors
- Frozen transform handling
- Animation export with selections

---

## [1.0.0] - January 2026

### Initial Release
- openPBRSurface & standardSurface materials
- Animation with frame baking
- Skeletal animation (skinClusters)
- GLB texture embedding
- Timeline controls