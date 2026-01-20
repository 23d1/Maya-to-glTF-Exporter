"""
Maya GLTF/GLB Exporter v4.0.8
GLTF 2.0 exporter for Autodesk Maya 2026+

Features:
- openPBRSurface and standardSurface material support
- Full animation export with frame baking
- Skeletal animation support (skinClusters)
- Texture embedding in GLB format
- Proper hierarchy support with parent/child relationships
- Timeline and custom frame range control
- Fixed axis and pivot point handling
- Accurate quaternion rotation export using Maya API
- Correct transform inheritance for animated meshes

Author: Created with assistance from Claude (Anthropic)
License: MIT
Version: 4.0.3
Date: January 2026

USAGE:
1. Copy this entire script
2. Paste into Maya Script Editor (Python tab)
3. Execute to open the exporter UI
"""


import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import json
import struct
import os
import traceback
import math

# Version information
VERSION = "4.0.8"
VERSION_DATE = "January 2026"

# ============================================================================
# EXPORTER CLASS WITH ANIMATION
# ============================================================================

class GLTFExporter:
    """GLTF exporter with full animation support"""
    
    def __init__(self):
        self.gltf_data = {
            "asset": {"version": "2.0", "generator": f"Maya GLTF Exporter v{VERSION}"},
            "scene": 0,
            "scenes": [],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "animations": [],
            "skins": []
        }
        self.binary_data = bytearray()
        self.material_index_map = {}
        self.texture_index_map = {}
        self.node_index_map = {}  # Maya node name -> GLTF node index
        self.joint_index_map = {}  # Joint name -> GLTF node index
        self.pivot_map = {}  # Maya node name -> rotatePivot [x, y, z]
        self.is_glb = False
        self.extensions_used = set()  # Track which extensions are used
        
        # Animation settings
        self.export_animation = False
        self.bake_animation = False
        self.force_bake_all = False  # Bake all transforms even without keyframes
        self.use_timeline = True
        self.start_frame = 1
        self.end_frame = 100
        self.sample_rate = 1  # Sample every N frames
        
    def export(self, filepath, export_format='glb', selection_only=False, 
               export_anim=False, bake_anim=True, force_bake_all=False, use_timeline=True, 
               start_frame=None, end_frame=None, sample_rate=1):
        """Main export function with animation options"""
        print("\n" + "="*60)
        print(f"GLTF EXPORT v{VERSION}")
        print("="*60)
        print(f"Output: {filepath}")
        print(f"Format: {export_format}")
        print(f"Animation: {'Yes' if export_anim else 'No'}")
        
        self.is_glb = (export_format == 'glb')
        self.export_animation = export_anim
        self.bake_animation = bake_anim
        self.force_bake_all = force_bake_all
        self.use_timeline = use_timeline
        self.sample_rate = sample_rate
        self.pivot_map = {}
        
        # Get animation range
        if export_anim:
            if use_timeline:
                self.start_frame = int(cmds.playbackOptions(query=True, minTime=True))
                self.end_frame = int(cmds.playbackOptions(query=True, maxTime=True))
            else:
                self.start_frame = start_frame if start_frame is not None else 1
                self.end_frame = end_frame if end_frame is not None else 100
            
            print(f"Animation Range: {self.start_frame} - {self.end_frame}")
            print(f"Sample Rate: Every {sample_rate} frame(s)")
            print(f"Baking: {'Yes' if bake_anim else 'No'}")
        
        # Validate filepath
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            print(f"ERROR: Directory does not exist: {output_dir}")
            return False
        
        # Get objects
        if selection_only:
            objects = cmds.ls(selection=True, dag=True, shapes=True, type='mesh')
            # Get transforms for animation (with LONG names to match node_index_map)
            # Use dag=True to get children of selected parents
            self.export_transforms = cmds.ls(selection=True, dag=True, transforms=True, long=True)
        else:
            objects = cmds.ls(dag=True, shapes=True, type='mesh')
            self.export_transforms = cmds.ls(dag=True, transforms=True, long=True)
        
        if not objects:
            print("ERROR: No meshes found")
            return False
        
        # Filter intermediate objects
        objects = [obj for obj in objects if not cmds.getAttr(f"{obj}.intermediateObject")]
        objects = list(set(objects))
        
        print(f"Exporting {len(objects)} mesh(es)")
        
        # Store current time to restore later
        current_time = cmds.currentTime(query=True)
        
        try:
            # Create scene
            scene_data = {"name": "Scene", "nodes": []}
            
            # Process meshes and collect all node indices
            # Build hierarchy properly - preserve parent/child relationships
            root_nodes = self.build_hierarchy(objects)
            
            processed = 0
            
            # Process all meshes first (creates nodes)
            for mesh_shape in objects:
                try:
                    print(f"\n{mesh_shape}")
                    node_idx = self.process_mesh(mesh_shape)
                    if node_idx is not None:
                        processed += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
            
            # Process parent transforms (empty nodes)
            self.process_parent_transforms(objects)
            
            # Build parent-child relationships in GLTF
            self.build_gltf_hierarchy()
            
            # Add only ROOT nodes to scene (children are referenced by parents)
            scene_data["nodes"] = [self.node_index_map[node] for node in root_nodes if node in self.node_index_map]
            
            if processed == 0:
                return False
            
            print(f"\n✓ Processed {processed} mesh(es)")
            print(f"✓ Scene hierarchy: {len(scene_data['nodes'])} root node(s)")
            
            self.gltf_data["scenes"].append(scene_data)
            
            # Add extensions if any were used
            if self.extensions_used:
                self.gltf_data["extensionsUsed"] = sorted(list(self.extensions_used))
                print(f"\nGLTF Extensions: {', '.join(sorted(self.extensions_used))}")
            
            # Add sampler
            self.gltf_data["samplers"].append({
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497
            })
            
            # Export animation if enabled
            if self.export_animation:
                print("\nExporting animation...")
                self.export_animations()
            
            # Write
            if export_format == 'glb':
                success = self.write_glb(filepath)
            else:
                success = self.write_gltf(filepath)
            
            if success:
                print("\n" + "="*60)
                print("✓ EXPORT COMPLETE")
                print("="*60)
                if os.path.exists(filepath):
                    print(f"File: {filepath}")
                    print(f"Size: {os.path.getsize(filepath):,} bytes")
                print("="*60)
            
            return success
            
        finally:
            # Restore original time
            cmds.currentTime(current_time)
    
    def build_hierarchy(self, mesh_shapes):
        """Build proper node hierarchy preserving parent/child relationships"""
        print("\nBuilding hierarchy...")
        
        # Get all transforms that need to be exported
        transforms_to_export = set()
        
        # Add all mesh transforms
        for mesh_shape in mesh_shapes:
            transform = cmds.listRelatives(mesh_shape, parent=True, fullPath=True)
            if transform:
                transforms_to_export.add(transform[0])
        
        # Add all parent transforms up to root
        for transform in list(transforms_to_export):
            parent = cmds.listRelatives(transform, parent=True, fullPath=True)
            while parent:
                transforms_to_export.add(parent[0])
                parent = cmds.listRelatives(parent[0], parent=True, fullPath=True)
        
        print(f"  Total transforms to export: {len(transforms_to_export)}")
        
        # Store for later use
        self.transforms_to_export = transforms_to_export
        
        # Find root nodes
        root_nodes = []
        for transform in transforms_to_export:
            parent = cmds.listRelatives(transform, parent=True, fullPath=True)
            if not parent or parent[0] not in transforms_to_export:
                root_nodes.append(transform)
        
        print(f"  Root nodes: {len(root_nodes)}")
        for root in root_nodes:
            print(f"    - {root.split('|')[-1]}")
            
        # Store pivots for all transforms
        for transform in transforms_to_export:
            pivot = cmds.xform(transform, query=True, rotatePivot=True, worldSpace=False)
            self.pivot_map[transform] = pivot
        
        return root_nodes
    
    def process_parent_transforms(self, mesh_shapes):
        """Create nodes for parent transforms (empty nodes without meshes)"""
        print("\nProcessing parent transforms...")
        
        # Collect all parent transforms
        for transform in self.transforms_to_export:
            # Skip if already processed (mesh nodes)
            if transform in self.node_index_map:
                continue
            
            # This is an empty transform (no mesh)
            short_name = transform.split('|')[-1]
            print(f"  {short_name} (empty transform)")
            
            # Get LOCAL transform
            local_trans, local_quat, local_scale = self.get_transform_with_pivot(transform)
            
            # Get pivot and parent pivot for compensation
            self_pivot = self.pivot_map.get(transform, [0, 0, 0])
            parent = cmds.listRelatives(transform, parent=True, fullPath=True)
            parent_pivot = [0, 0, 0]
            if parent and parent[0] in self.pivot_map:
                parent_pivot = self.pivot_map[parent[0]]
            
            # Compensate for parent pivot and move node to self pivot
            # glTF_trans = local_trans - parent_pivot + self_pivot
            final_trans = [
                local_trans[0] - parent_pivot[0] + self_pivot[0],
                local_trans[1] - parent_pivot[1] + self_pivot[1],
                local_trans[2] - parent_pivot[2] + self_pivot[2]
            ]
            
            # Create node
            node_data = {"name": short_name}
            
            # Add non-identity transforms
            if not all(abs(t) < 0.0001 for t in final_trans):
                node_data["translation"] = final_trans
            
            if not all(abs(r - 1.0 if i == 3 else 0.0) < 0.0001 for i, r in enumerate(local_quat)):
                node_data["rotation"] = local_quat
            
            if not all(abs(s - 1.0) < 0.0001 for s in local_scale):
                node_data["scale"] = local_scale
            
            node_index = len(self.gltf_data["nodes"])
            self.gltf_data["nodes"].append(node_data)
            self.node_index_map[transform] = node_index
    
    def build_gltf_hierarchy(self):
        """Build parent-child relationships in GLTF nodes"""
        print("\nBuilding GLTF hierarchy...")
        
        for transform in self.transforms_to_export:
            if transform not in self.node_index_map:
                continue
            
            node_idx = self.node_index_map[transform]
            
            # Get Maya children
            children = cmds.listRelatives(transform, children=True, type='transform', fullPath=True)
            if children:
                child_indices = []
                for child in children:
                    if child in self.node_index_map:
                        child_indices.append(self.node_index_map[child])
                
                if child_indices:
                    self.gltf_data["nodes"][node_idx]["children"] = child_indices
                    short_name = transform.split('|')[-1]
                    print(f"  {short_name} → {len(child_indices)} child(ren)")
    
    def process_mesh(self, mesh_shape):
        """Process mesh with animation support"""
        transform = cmds.listRelatives(mesh_shape, parent=True, fullPath=True)
        if not transform:
            return None
        
        transform = transform[0]
        mesh_name = transform.split('|')[-1]
        
        # Check if already processed
        if transform in self.node_index_map:
            return self.node_index_map[transform]
        
        # Get mesh data
        sel_list = om.MSelectionList()
        sel_list.add(mesh_shape)
        dag_path = om.MDagPath()
        sel_list.getDagPath(0, dag_path)
        mesh_fn = om.MFnMesh(dag_path)
        
        points = om.MPointArray()
        mesh_fn.getPoints(points, om.MSpace.kObject)
        
        u_array = om.MFloatArray()
        v_array = om.MFloatArray()
        mesh_fn.getUVs(u_array, v_array)
        
        # Check for skinning
        skin_cluster = self.get_skin_cluster(mesh_shape)
        
        # Get pivot and local translation
        rotate_pivot = cmds.xform(transform, query=True, rotatePivot=True, worldSpace=False)
        local_trans = cmds.xform(transform, query=True, translation=True, worldSpace=False)
        
        # Check if transforms are frozen (local translation is zero)
        transforms_frozen = all(abs(t) < 0.0001 for t in local_trans)
        pivot_at_origin = all(abs(p) < 0.0001 for p in rotate_pivot)
        
        # In Maya, vertices are stored relative to the pivot point.
        # For GLTF, we want vertices relative to the object origin.
        # So we offset vertices by -pivot to make them relative to origin,
        # and add +pivot to translation to compensate.
        should_offset_vertices = not pivot_at_origin
        
        if should_offset_vertices:
            print(f"  ℹ Pivot point: [{rotate_pivot[0]:.4f}, {rotate_pivot[1]:.4f}, {rotate_pivot[2]:.4f}]")
            print(f"  ℹ Offsetting vertices by -pivot for GLTF compatibility")
        
        # Build vertex data
        vertices = []
        vertex_normals = []
        uvs = []
        indices = []
        joints = []  # Joint indices for skinning
        weights = []  # Joint weights for skinning
        vertex_map = {}
        unique_index = 0
        
        poly_iter = om.MItMeshPolygon(dag_path)
        
        while not poly_iter.isDone():
            point_array = om.MPointArray()
            vertex_list = om.MIntArray()
            poly_iter.getTriangles(point_array, vertex_list)
            
            num_triangles = len(vertex_list) // 3
            
            for tri in range(num_triangles):
                for i in range(3):
                    vert_index = vertex_list[tri * 3 + i]
                    
                    # Get smooth vertex normal
                    normal = om.MVector()
                    mesh_fn.getVertexNormal(vert_index, True, normal, om.MSpace.kObject)
                    
                    # Get UV
                    uv_index = -1
                    try:
                        script_util = om.MScriptUtil()
                        script_util.createFromInt(0)
                        uv_ptr = script_util.asIntPtr()
                        
                        vertex_count = poly_iter.polygonVertexCount()
                        for pv in range(vertex_count):
                            if poly_iter.vertexIndex(pv) == vert_index:
                                poly_iter.getUVIndex(pv, uv_ptr)
                                uv_index = script_util.getInt(uv_ptr)
                                break
                    except:
                        pass
                    
                    # Unique vertex key
                    normal_key = (round(normal.x, 6), round(normal.y, 6), round(normal.z, 6))
                    vertex_key = (vert_index, normal_key, uv_index)
                    
                    if vertex_key not in vertex_map:
                        vertex_map[vertex_key] = unique_index
                        
                        point = points[vert_index]
                        
                        # Offset vertices by -pivot to make them relative to object origin
                        if should_offset_vertices:
                            vertices.extend([
                                point.x - rotate_pivot[0],
                                point.y - rotate_pivot[1],
                                point.z - rotate_pivot[2]
                            ])
                        else:
                            vertices.extend([point.x, point.y, point.z])
                        
                        vertex_normals.extend([normal.x, normal.y, normal.z])
                        
                        if uv_index >= 0 and uv_index < len(u_array):
                            uvs.extend([u_array[uv_index], 1.0 - v_array[uv_index]])
                        else:
                            uvs.extend([0.0, 0.0])
                        
                        # Get skin weights if skinned
                        if skin_cluster:
                            joint_indices, joint_weights = self.get_vertex_weights(skin_cluster, vert_index)
                            joints.extend(joint_indices)
                            weights.extend(joint_weights)
                        
                        unique_index += 1
                    
                    indices.append(vertex_map[vertex_key])
            
            poly_iter.next()
        
        if len(vertices) == 0 or len(indices) == 0:
            return None
        
        # Pack geometry data
        position_bytes = struct.pack(f'{len(vertices)}f', *vertices)
        normal_bytes = struct.pack(f'{len(vertex_normals)}f', *vertex_normals)
        uv_bytes = struct.pack(f'{len(uvs)}f', *uvs)
        index_bytes = struct.pack(f'{len(indices)}I', *indices)
        
        # Add to buffer
        def align():
            while len(self.binary_data) % 4 != 0:
                self.binary_data.append(0)
        
        align()
        pos_offset = len(self.binary_data)
        self.binary_data.extend(position_bytes)
        
        align()
        norm_offset = len(self.binary_data)
        self.binary_data.extend(normal_bytes)
        
        align()
        uv_offset = len(self.binary_data)
        self.binary_data.extend(uv_bytes)
        
        align()
        idx_offset = len(self.binary_data)
        self.binary_data.extend(index_bytes)
        
        # Skin data if present
        joints_offset = None
        weights_offset = None
        if skin_cluster and len(joints) > 0:
            joints_bytes = struct.pack(f'{len(joints)}H', *joints)  # Unsigned short
            weights_bytes = struct.pack(f'{len(weights)}f', *weights)
            
            align()
            joints_offset = len(self.binary_data)
            self.binary_data.extend(joints_bytes)
            
            align()
            weights_offset = len(self.binary_data)
            self.binary_data.extend(weights_bytes)
        
        # Bounds
        min_pos = [min(vertices[i::3]) for i in range(3)]
        max_pos = [max(vertices[i::3]) for i in range(3)]
        
        # Create accessors
        pos_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5126,
            "count": len(vertices) // 3,
            "type": "VEC3",
            "min": min_pos,
            "max": max_pos
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": len(position_bytes),
            "target": 34962
        })
        
        norm_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5126,
            "count": len(vertex_normals) // 3,
            "type": "VEC3"
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": norm_offset,
            "byteLength": len(normal_bytes),
            "target": 34962
        })
        
        uv_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5126,
            "count": len(uvs) // 2,
            "type": "VEC2"
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": len(uv_bytes),
            "target": 34962
        })
        
        idx_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5125,
            "count": len(indices),
            "type": "SCALAR"
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": idx_offset,
            "byteLength": len(index_bytes),
            "target": 34963
        })
        
        # Skin accessors
        joints_acc = None
        weights_acc = None
        if joints_offset is not None:
            joints_acc = len(self.gltf_data["accessors"])
            self.gltf_data["accessors"].append({
                "bufferView": len(self.gltf_data["bufferViews"]),
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(joints) // 4,
                "type": "VEC4"
            })
            self.gltf_data["bufferViews"].append({
                "buffer": 0,
                "byteOffset": joints_offset,
                "byteLength": len(joints_bytes),
                "target": 34962
            })
            
            weights_acc = len(self.gltf_data["accessors"])
            self.gltf_data["accessors"].append({
                "bufferView": len(self.gltf_data["bufferViews"]),
                "componentType": 5126,
                "count": len(weights) // 4,
                "type": "VEC4"
            })
            self.gltf_data["bufferViews"].append({
                "buffer": 0,
                "byteOffset": weights_offset,
                "byteLength": len(weights_bytes),
                "target": 34962
            })
        
        # Material
        material_index = self.process_material(mesh_shape)
        
        # Create mesh
        mesh_attributes = {
            "POSITION": pos_acc,
            "NORMAL": norm_acc,
            "TEXCOORD_0": uv_acc
        }
        
        if joints_acc is not None:
            mesh_attributes["JOINTS_0"] = joints_acc
            mesh_attributes["WEIGHTS_0"] = weights_acc
        
        mesh_data = {
            "name": mesh_name,
            "primitives": [{
                "attributes": mesh_attributes,
                "indices": idx_acc,
                "mode": 4
            }]
        }
        
        if material_index is not None:
            mesh_data["primitives"][0]["material"] = material_index
        
        mesh_index = len(self.gltf_data["meshes"])
        self.gltf_data["meshes"].append(mesh_data)
        
        # Create node with transform
        node_data = {
            "name": mesh_name,
            "mesh": mesh_index
        }
        
        # Get LOCAL transform (relative to parent)
        translation, rotation, scale = self.get_transform_with_pivot(transform)
        
        # Get pivot and parent pivot for compensation
        self_pivot = self.pivot_map.get(transform, [0, 0, 0])
        parent = cmds.listRelatives(transform, parent=True, fullPath=True)
        parent_pivot = [0, 0, 0]
        if parent and parent[0] in self.pivot_map:
            parent_pivot = self.pivot_map[parent[0]]
            
        # Compensate for parent pivot and move node to self pivot
        # glTF_trans = local_trans - parent_pivot + self_pivot
        final_translation = [
            translation[0] - parent_pivot[0] + self_pivot[0],
            translation[1] - parent_pivot[1] + self_pivot[1],
            translation[2] - parent_pivot[2] + self_pivot[2]
        ]
        
        # Debug output
        print(f"  ℹ Local transform: T[{final_translation[0]:.3f}, {final_translation[1]:.3f}, {final_translation[2]:.3f}] " +
              f"R[{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}]")
        
        # Add non-identity transforms
        if not all(abs(t) < 0.0001 for t in final_translation):
            node_data["translation"] = final_translation
        
        # Export local rotation for mesh nodes
        if not all(abs(r - 1.0 if i == 3 else 0.0) < 0.0001 for i, r in enumerate(rotation)):
            node_data["rotation"] = rotation
        
        if not all(abs(s - 1.0) < 0.0001 for s in scale):
            node_data["scale"] = scale
        
        # Add skin reference if skinned
        if skin_cluster:
            skin_index = self.create_skin(skin_cluster, mesh_shape)
            if skin_index is not None:
                node_data["skin"] = skin_index
                print(f"  ✓ Skinned mesh")
        
        node_index = len(self.gltf_data["nodes"])
        self.gltf_data["nodes"].append(node_data)
        self.node_index_map[transform] = node_index
        
        print(f"  ✓ {unique_index} verts, {len(indices)//3} tris")
        print(f"  ✓ Added to node_index_map: {transform} → Node {node_index}")
        
        return node_index
    
    def get_skin_cluster(self, mesh_shape):
        """Find skin cluster attached to mesh"""
        history = cmds.listHistory(mesh_shape, pruneDagObjects=True)
        if history:
            skin_clusters = cmds.ls(history, type='skinCluster')
            if skin_clusters:
                return skin_clusters[0]
        return None
    
    def get_vertex_weights(self, skin_cluster, vertex_index):
        """Get joint indices and weights for a vertex"""
        # Get influences for this vertex
        influences = cmds.skinPercent(skin_cluster, f"{skin_cluster}.vtx[{vertex_index}]", 
                                     query=True, value=True)
        
        # Get joint names
        joints = cmds.skinCluster(skin_cluster, query=True, influence=True)
        
        # Collect non-zero weights (up to 4 for GLTF)
        joint_weight_pairs = []
        for i, weight in enumerate(influences):
            if weight > 0.0001 and i < len(joints):
                joint_weight_pairs.append((joints[i], weight))
        
        # Sort by weight (descending) and take top 4
        joint_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        joint_weight_pairs = joint_weight_pairs[:4]
        
        # Normalize weights to sum to 1.0
        total_weight = sum(w for _, w in joint_weight_pairs)
        if total_weight > 0:
            joint_weight_pairs = [(j, w/total_weight) for j, w in joint_weight_pairs]
        
        # Pad to 4 entries
        while len(joint_weight_pairs) < 4:
            joint_weight_pairs.append((None, 0.0))
        
        # Convert to indices and weights
        joint_indices = []
        joint_weights = []
        
        for joint_name, weight in joint_weight_pairs:
            if joint_name and joint_name in self.joint_index_map:
                joint_indices.append(self.joint_index_map[joint_name])
            else:
                joint_indices.append(0)
            joint_weights.append(weight)
        
        return joint_indices, joint_weights
    
    def create_skin(self, skin_cluster, mesh_shape):
        """Create GLTF skin"""
        joints = cmds.skinCluster(skin_cluster, query=True, influence=True)
        
        if not joints:
            return None
        
        # Map joints to node indices
        joint_node_indices = []
        for joint in joints:
            if joint not in self.joint_index_map:
                # Create joint node
                joint_idx = self.create_joint_node(joint)
                self.joint_index_map[joint] = joint_idx
            joint_node_indices.append(self.joint_index_map[joint])
        
        # Get inverse bind matrices
        bind_pre_matrix = cmds.getAttr(f"{skin_cluster}.bindPreMatrix")
        inverse_bind_matrices = []
        
        for joint in joints:
            # Get world inverse matrix at bind pose
            matrix = cmds.getAttr(f"{joint}.worldInverseMatrix[0]")
            # Convert to column-major
            gltf_matrix = [
                matrix[0], matrix[4], matrix[8], matrix[12],
                matrix[1], matrix[5], matrix[9], matrix[13],
                matrix[2], matrix[6], matrix[10], matrix[14],
                matrix[3], matrix[7], matrix[11], matrix[15]
            ]
            inverse_bind_matrices.extend(gltf_matrix)
        
        # Pack inverse bind matrices
        ibm_bytes = struct.pack(f'{len(inverse_bind_matrices)}f', *inverse_bind_matrices)
        
        while len(self.binary_data) % 4 != 0:
            self.binary_data.append(0)
        
        ibm_offset = len(self.binary_data)
        self.binary_data.extend(ibm_bytes)
        
        # Create accessor
        ibm_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5126,
            "count": len(joints),
            "type": "MAT4"
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": ibm_offset,
            "byteLength": len(ibm_bytes)
        })
        
        # Create skin
        skin_index = len(self.gltf_data["skins"])
        self.gltf_data["skins"].append({
            "joints": joint_node_indices,
            "inverseBindMatrices": ibm_acc
        })
        
        return skin_index
    
    def create_joint_node(self, joint):
        """Create GLTF node for joint"""
        node_data = {"name": joint}
        
        # Get transform
        matrix = cmds.xform(joint, query=True, matrix=True, worldSpace=False)
        gltf_matrix = [
            matrix[0], matrix[4], matrix[8], matrix[12],
            matrix[1], matrix[5], matrix[9], matrix[13],
            matrix[2], matrix[6], matrix[10], matrix[14],
            matrix[3], matrix[7], matrix[11], matrix[15]
        ]
        
        node_data["matrix"] = gltf_matrix
        
        node_index = len(self.gltf_data["nodes"])
        self.gltf_data["nodes"].append(node_data)
        
        return node_index
    
    def export_animations(self):
        """Export baked animation data"""
        if not self.export_animation:
            print("  Animation export disabled")
            return
        
        print(f"\n{'='*60}")
        print(f"EXPORTING ANIMATION")
        print(f"{'='*60}")
        print(f"  Baking animation from frame {self.start_frame} to {self.end_frame}...")
        print(f"  Total transforms in scene: {len(self.export_transforms)}")
        print(f"  Transforms with node indices: {len([t for t in self.export_transforms if t in self.node_index_map])}")
        
        # Debug: show node_index_map
        print(f"\n  Node Index Map ({len(self.node_index_map)} entries):")
        for transform, node_idx in self.node_index_map.items():
            print(f"    {transform} → Node {node_idx}")
        
        # Collect all animated transforms
        animated_nodes = []
        
        print(f"\n  Checking for animation...")
        
        if self.force_bake_all:
            # Bake ALL transforms that were exported (have node indices)
            exported_transforms = [t for t in self.export_transforms if t in self.node_index_map]
            print(f"  Force bake mode: checking {len(exported_transforms)} exported transforms")
            animated_nodes = exported_transforms
        else:
            # Only bake transforms with keyframes AND that were exported
            for transform in self.export_transforms:
                print(f"    Checking: {transform}")
                
                in_map = transform in self.node_index_map
                is_anim = self.is_animated(transform) if in_map else False
                
                print(f"      In node_index_map: {in_map}")
                if in_map:
                    print(f"      Is animated: {is_anim}")
                
                if in_map and is_anim:
                    animated_nodes.append(transform)
                    print(f"      ✓ Will export animation for this transform")
        
        if not animated_nodes:
            print(f"\n  ❌ No animated transforms found to export")
            print(f"  Reasons this could happen:")
            print(f"    1. Animated transforms not in node_index_map")
            print(f"    2. No keyframes detected (try Force Bake All)")
            return
        
        print(f"\n  ✓ Found {len(animated_nodes)} animated transforms:")
        for t in animated_nodes:
            print(f"    - {t}")
        
        # Create animation
        animation_data = {
            "name": "BakedAnimation",
            "channels": [],
            "samplers": []
        }
        
        # Time values (in seconds)
        fps = self.get_fps()
        frames = range(int(self.start_frame), int(self.end_frame) + 1, self.sample_rate)
        time_values = [(f - self.start_frame) / fps for f in frames]
        
        # Pack time data
        time_bytes = struct.pack(f'{len(time_values)}f', *time_values)
        
        while len(self.binary_data) % 4 != 0:
            self.binary_data.append(0)
        
        time_offset = len(self.binary_data)
        self.binary_data.extend(time_bytes)
        
        # Create time accessor
        time_acc = len(self.gltf_data["accessors"])
        self.gltf_data["accessors"].append({
            "bufferView": len(self.gltf_data["bufferViews"]),
            "componentType": 5126,
            "count": len(time_values),
            "type": "SCALAR",
            "min": [min(time_values)],
            "max": [max(time_values)]
        })
        self.gltf_data["bufferViews"].append({
            "buffer": 0,
            "byteOffset": time_offset,
            "byteLength": len(time_bytes)
        })
        
        # Bake each transform
        for transform in animated_nodes:
            if transform not in self.node_index_map:
                continue
            
            node_idx = self.node_index_map[transform]
            
            print(f"\n  Baking {transform.split('|')[-1]}...")
            
            # Sample LOCAL transforms at every frame (relative to parent)
            translations = []
            rotations = []
            scales = []
            
            # Get pivot and parent pivot for compensation
            self_pivot = self.pivot_map.get(transform, [0, 0, 0])
            parent = cmds.listRelatives(transform, parent=True, fullPath=True)
            parent_pivot = [0, 0, 0]
            if parent and parent[0] in self.pivot_map:
                parent_pivot = self.pivot_map[parent[0]]
            
            for frame in frames:
                cmds.currentTime(frame)
                
                # Get LOCAL translation, rotation, scale
                local_trans = cmds.xform(transform, query=True, translation=True, worldSpace=False)
                
                # Compensate for parent pivot and move node to self pivot
                final_trans = [
                    local_trans[0] - parent_pivot[0] + self_pivot[0],
                    local_trans[1] - parent_pivot[1] + self_pivot[1],
                    local_trans[2] - parent_pivot[2] + self_pivot[2]
                ]
                translations.extend(final_trans)
                
                # Get LOCAL rotation as quaternion
                local_quat = self.get_transform_quaternion(transform)
                rotations.extend(local_quat)
                
                # Get LOCAL scale
                local_scale = cmds.xform(transform, query=True, scale=True, relative=True)
                scales.extend(local_scale)
            
            print(f"    Sampled {len(frames)} frames (local transforms)")
            
            # Pack and add translation
            if len(translations) > 0:
                trans_bytes = struct.pack(f'{len(translations)}f', *translations)
                
                while len(self.binary_data) % 4 != 0:
                    self.binary_data.append(0)
                
                trans_offset = len(self.binary_data)
                self.binary_data.extend(trans_bytes)
                
                trans_acc = len(self.gltf_data["accessors"])
                self.gltf_data["accessors"].append({
                    "bufferView": len(self.gltf_data["bufferViews"]),
                    "componentType": 5126,
                    "count": len(translations) // 3,
                    "type": "VEC3"
                })
                self.gltf_data["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": trans_offset,
                    "byteLength": len(trans_bytes)
                })
                
                # Add sampler and channel for translation
                sampler_idx = len(animation_data["samplers"])
                animation_data["samplers"].append({
                    "input": time_acc,
                    "output": trans_acc,
                    "interpolation": "LINEAR"
                })
                animation_data["channels"].append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": node_idx,
                        "path": "translation"
                    }
                })
            
            # Pack and add rotation
            if len(rotations) > 0:
                rot_bytes = struct.pack(f'{len(rotations)}f', *rotations)
                
                while len(self.binary_data) % 4 != 0:
                    self.binary_data.append(0)
                
                rot_offset = len(self.binary_data)
                self.binary_data.extend(rot_bytes)
                
                rot_acc = len(self.gltf_data["accessors"])
                self.gltf_data["accessors"].append({
                    "bufferView": len(self.gltf_data["bufferViews"]),
                    "componentType": 5126,
                    "count": len(rotations) // 4,
                    "type": "VEC4"
                })
                self.gltf_data["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": rot_offset,
                    "byteLength": len(rot_bytes)
                })
                
                sampler_idx = len(animation_data["samplers"])
                animation_data["samplers"].append({
                    "input": time_acc,
                    "output": rot_acc,
                    "interpolation": "LINEAR"
                })
                animation_data["channels"].append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": node_idx,
                        "path": "rotation"
                    }
                })
            
            # Pack and add scale
            if len(scales) > 0:
                scale_bytes = struct.pack(f'{len(scales)}f', *scales)
                
                while len(self.binary_data) % 4 != 0:
                    self.binary_data.append(0)
                
                scale_offset = len(self.binary_data)
                self.binary_data.extend(scale_bytes)
                
                scale_acc = len(self.gltf_data["accessors"])
                self.gltf_data["accessors"].append({
                    "bufferView": len(self.gltf_data["bufferViews"]),
                    "componentType": 5126,
                    "count": len(scales) // 3,
                    "type": "VEC3"
                })
                self.gltf_data["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": scale_offset,
                    "byteLength": len(scale_bytes)
                })
                
                sampler_idx = len(animation_data["samplers"])
                animation_data["samplers"].append({
                    "input": time_acc,
                    "output": scale_acc,
                    "interpolation": "LINEAR"
                })
                animation_data["channels"].append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": node_idx,
                        "path": "scale"
                    }
                })
        
        if len(animation_data["channels"]) > 0:
            self.gltf_data["animations"].append(animation_data)
            print(f"  ✓ Exported {len(animation_data['channels'])} animation channels")
        else:
            print("  No animation channels exported")
    
    def get_transform_with_pivot(self, transform):
        """Get LOCAL translation/rotation/scale for hierarchical export"""
        # Get local-space transforms (relative to parent)
        # These already account for pivot points correctly
        local_trans = cmds.xform(transform, query=True, translation=True, worldSpace=False)
        local_quat = self.get_transform_quaternion(transform)
        local_scale = cmds.xform(transform, query=True, scale=True, relative=True)
        
        return local_trans, local_quat, local_scale
    
    def is_animated(self, transform):
        """Check if transform is animated"""
        # Check for keyframes on transform attributes
        attrs = ['translateX', 'translateY', 'translateZ', 
                'rotateX', 'rotateY', 'rotateZ',
                'scaleX', 'scaleY', 'scaleZ']
        
        for attr in attrs:
            connections = cmds.listConnections(f"{transform}.{attr}", source=True, destination=False)
            if connections:
                return True
        
        return False
    
    def get_fps(self):
        """Get current FPS setting"""
        time_unit = cmds.currentUnit(query=True, time=True)
        fps_map = {
            'game': 15,
            'film': 24,
            'pal': 25,
            'ntsc': 30,
            'show': 48,
            'palf': 50,
            'ntscf': 60
        }
        return fps_map.get(time_unit, 24)
    
    def get_transform_quaternion(self, transform):
        """Get the local quaternion rotation of a transform using Maya API"""
        try:
            # Get the MObject for the transform
            sel_list = om.MSelectionList()
            sel_list.add(transform)
            mobj = om.MObject()
            sel_list.getDependNode(0, mobj)
            
            # Create MFnTransform
            fn_transform = om.MFnTransform(mobj)
            
            # Get the local rotation as quaternion
            quat = om.MQuaternion()
            fn_transform.getRotation(quat, om.MSpace.kTransform)
            
            # Return as [x, y, z, w] list, negated to match GLTF rotation direction
            return [-quat.x, -quat.y, -quat.z, quat.w]
            
        except:
            # Fallback: convert from Euler angles
            local_rot = cmds.xform(transform, query=True, rotation=True, worldSpace=False)
            return self.euler_to_quaternion(local_rot[0], local_rot[1], local_rot[2])
    
    def matrix_inverse(self, matrix):
        """Compute inverse of 4x4 matrix (simplified for affine transforms)"""
        # For affine transforms, we can use a simplified inverse
        # This assumes the matrix is affine (last row is [0,0,0,1])
        m = matrix
        
        # Extract rotation/scale part (3x3)
        rot_scale = [
            [m[0], m[1], m[2]],
            [m[4], m[5], m[6]],
            [m[8], m[9], m[10]]
        ]
        
        # Extract translation
        trans = [m[3], m[7], m[11]]
        
        # Compute determinant of 3x3 matrix
        det = (rot_scale[0][0] * (rot_scale[1][1] * rot_scale[2][2] - rot_scale[1][2] * rot_scale[2][1]) -
               rot_scale[0][1] * (rot_scale[1][0] * rot_scale[2][2] - rot_scale[1][2] * rot_scale[2][0]) +
               rot_scale[0][2] * (rot_scale[1][0] * rot_scale[2][1] - rot_scale[1][1] * rot_scale[2][0]))
        
        if abs(det) < 1e-6:
            # Singular matrix, return identity
            return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        
        # Compute inverse of 3x3
        inv_det = 1.0 / det
        inv_rot_scale = [
            [(rot_scale[1][1] * rot_scale[2][2] - rot_scale[1][2] * rot_scale[2][1]) * inv_det,
             (rot_scale[0][2] * rot_scale[2][1] - rot_scale[0][1] * rot_scale[2][2]) * inv_det,
             (rot_scale[0][1] * rot_scale[1][2] - rot_scale[0][2] * rot_scale[1][1]) * inv_det],
            [(rot_scale[1][2] * rot_scale[2][0] - rot_scale[1][0] * rot_scale[2][2]) * inv_det,
             (rot_scale[0][0] * rot_scale[2][2] - rot_scale[0][2] * rot_scale[2][0]) * inv_det,
             (rot_scale[0][2] * rot_scale[1][0] - rot_scale[0][0] * rot_scale[1][2]) * inv_det],
            [(rot_scale[1][0] * rot_scale[2][1] - rot_scale[1][1] * rot_scale[2][0]) * inv_det,
             (rot_scale[0][1] * rot_scale[2][0] - rot_scale[0][0] * rot_scale[2][1]) * inv_det,
             (rot_scale[0][0] * rot_scale[1][1] - rot_scale[0][1] * rot_scale[1][0]) * inv_det]
        ]
        
        # Compute inverse translation: -inv_rot_scale * trans
        inv_trans = [
            -(inv_rot_scale[0][0] * trans[0] + inv_rot_scale[0][1] * trans[1] + inv_rot_scale[0][2] * trans[2]),
            -(inv_rot_scale[1][0] * trans[0] + inv_rot_scale[1][1] * trans[1] + inv_rot_scale[1][2] * trans[2]),
            -(inv_rot_scale[2][0] * trans[0] + inv_rot_scale[2][1] * trans[1] + inv_rot_scale[2][2] * trans[2])
        ]
        
        # Build inverse matrix
        return [
            inv_rot_scale[0][0], inv_rot_scale[1][0], inv_rot_scale[2][0], inv_trans[0],
            inv_rot_scale[0][1], inv_rot_scale[1][1], inv_rot_scale[2][1], inv_trans[1],
            inv_rot_scale[0][2], inv_rot_scale[1][2], inv_rot_scale[2][2], inv_trans[2],
            0, 0, 0, 1
        ]
    
    def matrix_multiply(self, a, b):
        """Multiply two 4x4 matrices"""
        return [
            a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12],
            a[0]*b[1] + a[1]*b[5] + a[2]*b[9] + a[3]*b[13],
            a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14],
            a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15],
            
            a[4]*b[0] + a[5]*b[4] + a[6]*b[8] + a[7]*b[12],
            a[4]*b[1] + a[5]*b[5] + a[6]*b[9] + a[7]*b[13],
            a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14],
            a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15],
            
            a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12],
            a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13],
            a[8]*b[2] + a[9]*b[6] + a[10]*b[10] + a[11]*b[14],
            a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]*b[15],
            
            a[12]*b[0] + a[13]*b[4] + a[14]*b[8] + a[15]*b[12],
            a[12]*b[1] + a[13]*b[5] + a[14]*b[9] + a[15]*b[13],
            a[12]*b[2] + a[13]*b[6] + a[14]*b[10] + a[15]*b[14],
            a[12]*b[3] + a[13]*b[7] + a[14]*b[11] + a[15]*b[15]
        ]
    
    def decompose_matrix(self, matrix):
        """Decompose 4x4 transformation matrix into translation, rotation (Euler), scale"""
        # Extract translation
        translation = [matrix[3], matrix[7], matrix[11]]
        
        # Extract scale and rotation
        # This is a simplified decomposition - assumes no shear
        sx = math.sqrt(matrix[0]**2 + matrix[1]**2 + matrix[2]**2)
        sy = math.sqrt(matrix[4]**2 + matrix[5]**2 + matrix[6]**2)
        sz = math.sqrt(matrix[8]**2 + matrix[9]**2 + matrix[10]**2)
        
        scale = [sx, sy, sz]
        
        # Extract rotation matrix (divide by scale)
        if sx > 1e-6:
            rot_matrix = [
                matrix[0]/sx, matrix[1]/sx, matrix[2]/sx,
                matrix[4]/sy, matrix[5]/sy, matrix[6]/sy,
                matrix[8]/sz, matrix[9]/sz, matrix[10]/sz
            ]
        else:
            # Degenerate case
            rot_matrix = [1,0,0, 0,1,0, 0,0,1]
        
        # Convert rotation matrix to Euler angles (XYZ order)
        # This is a simplified extraction - may not handle all cases perfectly
        sy_rot = math.sqrt(rot_matrix[0]**2 + rot_matrix[3]**2)
        
        if sy_rot > 1e-6:
            x = math.atan2(rot_matrix[7], rot_matrix[8])
            y = math.atan2(-rot_matrix[6], sy_rot)
            z = math.atan2(rot_matrix[3], rot_matrix[0])
        else:
            x = math.atan2(-rot_matrix[5], rot_matrix[4])
            y = math.atan2(-rot_matrix[6], sy_rot)
            z = 0
        
        # Convert to degrees
        rotation = [math.degrees(x), math.degrees(y), math.degrees(z)]
        
        return translation, rotation, scale
    
    def process_material(self, mesh_shape):
        """Process materials"""
        try:
            shading_engines = cmds.listConnections(mesh_shape, type='shadingEngine')
            if not shading_engines:
                return None
            
            shading_engine = shading_engines[0]
            if shading_engine in self.material_index_map:
                return self.material_index_map[shading_engine]
            
            materials = cmds.ls(cmds.listConnections(shading_engine), materials=True)
            if not materials:
                return None
            
            material = materials[0]
            material_type = cmds.nodeType(material)
            
            if material_type == 'openPBRSurface':
                pbr_mat = self.convert_openpbr(material)
            elif material_type == 'standardSurface':
                pbr_mat = self.convert_standard(material)
            elif material_type in ['lambert', 'blinn', 'phong']:
                pbr_mat = self.convert_legacy(material, material_type)
            else:
                pbr_mat = self.create_default(material)
            
            mat_idx = len(self.gltf_data["materials"])
            self.gltf_data["materials"].append(pbr_mat)
            self.material_index_map[shading_engine] = mat_idx
            
            return mat_idx
        except Exception as e:
            return None
    
    def convert_openpbr(self, mat):
        """Convert openPBRSurface to GLTF PBR"""
        pbr = {"name": mat, "pbrMetallicRoughness": {}}
        
        # Base Color
        base_tex = self.get_texture(mat, "baseColor")
        if base_tex is not None:
            pbr["pbrMetallicRoughness"]["baseColorTexture"] = {"index": base_tex}
        else:
            try:
                r = cmds.getAttr(f"{mat}.baseColorR")
                g = cmds.getAttr(f"{mat}.baseColorG")
                b = cmds.getAttr(f"{mat}.baseColorB")
                weight = cmds.getAttr(f"{mat}.baseWeight")
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [r*weight, g*weight, b*weight, 1.0]
            except:
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]
        
        # Metalness
        metal_tex = self.get_texture(mat, "baseMetalness")
        if metal_tex is not None:
            pbr["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": metal_tex}
        else:
            try:
                metal = cmds.getAttr(f"{mat}.baseMetalness")
                pbr["pbrMetallicRoughness"]["metallicFactor"] = float(metal)
            except:
                pbr["pbrMetallicRoughness"]["metallicFactor"] = 0.0
        
        # Roughness
        rough_attrs = ['specularRoughness', 'baseDiffuseRoughness']
        roughness_set = False
        
        for attr in rough_attrs:
            if cmds.objExists(f"{mat}.{attr}"):
                rough_tex = self.get_texture(mat, attr)
                if rough_tex is not None:
                    # Only set if metallic texture not already set
                    if "metallicRoughnessTexture" not in pbr["pbrMetallicRoughness"]:
                        pbr["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": rough_tex}
                    roughness_set = True
                    break
                else:
                    try:
                        rough = cmds.getAttr(f"{mat}.{attr}")
                        pbr["pbrMetallicRoughness"]["roughnessFactor"] = float(rough)
                        roughness_set = True
                        break
                    except:
                        pass
        
        if not roughness_set:
            pbr["pbrMetallicRoughness"]["roughnessFactor"] = 0.5
        
        # Normal Map
        # Try common normal attribute names for openPBRSurface
        normal_attrs = ['geometryNormal', 'normalCamera', 'normal']
        for attr in normal_attrs:
            if cmds.objExists(f"{mat}.{attr}"):
                norm_tex = self.get_texture(mat, attr)
                if norm_tex is not None:
                    pbr["normalTexture"] = {"index": norm_tex}
                    # Note: GLTF viewers will auto-generate tangents if missing
                    # This is standard and expected behavior
                    break
        
        # Emission
        try:
            em_lum = cmds.getAttr(f"{mat}.emissionLuminance")
            if em_lum > 0.001:  # Only export if actually emitting
                em_tex = self.get_texture(mat, "emissionColor")
                if em_tex is not None:
                    pbr["emissiveTexture"] = {"index": em_tex}
                else:
                    try:
                        em_r = cmds.getAttr(f"{mat}.emissionColorR")
                        em_g = cmds.getAttr(f"{mat}.emissionColorG")
                        em_b = cmds.getAttr(f"{mat}.emissionColorB")
                        pbr["emissiveFactor"] = [em_r * em_lum, em_g * em_lum, em_b * em_lum]
                    except:
                        pass
        except:
            pass
        
        # Opacity/Transparency
        try:
            opacity = cmds.getAttr(f"{mat}.geometryOpacity")
            if opacity < 0.999:  # Only set if actually transparent
                if "baseColorFactor" in pbr["pbrMetallicRoughness"]:
                    pbr["pbrMetallicRoughness"]["baseColorFactor"][3] = opacity
                pbr["alphaMode"] = "BLEND"
        except:
            pass
        
        # Clearcoat (KHR_materials_clearcoat extension)
        try:
            coat_weight = cmds.getAttr(f"{mat}.coatWeight")
            if coat_weight > 0.001:
                clearcoat_ext = {}
                
                # Clearcoat intensity
                clearcoat_ext["clearcoatFactor"] = float(coat_weight)
                
                # Clearcoat roughness
                try:
                    coat_roughness = cmds.getAttr(f"{mat}.coatRoughness")
                    clearcoat_ext["clearcoatRoughnessFactor"] = float(coat_roughness)
                except:
                    pass
                
                # Clearcoat texture (if any)
                coat_tex = self.get_texture(mat, "coatWeight")
                if coat_tex is not None:
                    clearcoat_ext["clearcoatTexture"] = {"index": coat_tex}
                
                # Clearcoat roughness texture
                coat_rough_tex = self.get_texture(mat, "coatRoughness")
                if coat_rough_tex is not None:
                    clearcoat_ext["clearcoatRoughnessTexture"] = {"index": coat_rough_tex}
                
                # Clearcoat normal
                coat_norm_tex = self.get_texture(mat, "geometryCoatNormal")
                if coat_norm_tex is not None:
                    clearcoat_ext["clearcoatNormalTexture"] = {"index": coat_norm_tex}
                
                if clearcoat_ext:
                    if "extensions" not in pbr:
                        pbr["extensions"] = {}
                    pbr["extensions"]["KHR_materials_clearcoat"] = clearcoat_ext
                    self.extensions_used.add("KHR_materials_clearcoat")
        except:
            pass
        
        # Sheen (KHR_materials_sheen extension)
        try:
            fuzz_weight = cmds.getAttr(f"{mat}.fuzzWeight")
            if fuzz_weight > 0.001:
                sheen_ext = {}
                
                # Sheen color (from fuzzColor)
                try:
                    fuzz_r = cmds.getAttr(f"{mat}.fuzzColorR")
                    fuzz_g = cmds.getAttr(f"{mat}.fuzzColorG")
                    fuzz_b = cmds.getAttr(f"{mat}.fuzzColorB")
                    # Scale by weight
                    sheen_ext["sheenColorFactor"] = [fuzz_r * fuzz_weight, fuzz_g * fuzz_weight, fuzz_b * fuzz_weight]
                except:
                    sheen_ext["sheenColorFactor"] = [fuzz_weight, fuzz_weight, fuzz_weight]
                
                # Sheen roughness (from fuzzRoughness)
                try:
                    fuzz_roughness = cmds.getAttr(f"{mat}.fuzzRoughness")
                    sheen_ext["sheenRoughnessFactor"] = float(fuzz_roughness)
                except:
                    pass
                
                # Sheen color texture
                fuzz_color_tex = self.get_texture(mat, "fuzzColor")
                if fuzz_color_tex is not None:
                    sheen_ext["sheenColorTexture"] = {"index": fuzz_color_tex}
                
                # Sheen roughness texture
                fuzz_rough_tex = self.get_texture(mat, "fuzzRoughness")
                if fuzz_rough_tex is not None:
                    sheen_ext["sheenRoughnessTexture"] = {"index": fuzz_rough_tex}
                
                if sheen_ext:
                    if "extensions" not in pbr:
                        pbr["extensions"] = {}
                    pbr["extensions"]["KHR_materials_sheen"] = sheen_ext
                    self.extensions_used.add("KHR_materials_sheen")
        except:
            pass
        
        return pbr
    
    def convert_standard(self, mat):
        """Convert standardSurface to GLTF PBR"""
        pbr = {"name": mat, "pbrMetallicRoughness": {}}
        
        # Base Color
        tex = self.get_texture(mat, "baseColor")
        if tex is not None:
            pbr["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex}
        else:
            try:
                color = cmds.getAttr(f"{mat}.baseColor")[0]
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [color[0], color[1], color[2], 1.0]
            except:
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]
        
        # Metalness
        metal_tex = self.get_texture(mat, "metalness")
        if metal_tex is not None:
            pbr["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": metal_tex}
        else:
            try:
                pbr["pbrMetallicRoughness"]["metallicFactor"] = float(cmds.getAttr(f"{mat}.metalness"))
            except:
                pbr["pbrMetallicRoughness"]["metallicFactor"] = 0.0
        
        # Roughness
        rough_tex = self.get_texture(mat, "specularRoughness")
        if rough_tex is not None and "metallicRoughnessTexture" not in pbr["pbrMetallicRoughness"]:
            pbr["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": rough_tex}
        else:
            try:
                pbr["pbrMetallicRoughness"]["roughnessFactor"] = float(cmds.getAttr(f"{mat}.specularRoughness"))
            except:
                pbr["pbrMetallicRoughness"]["roughnessFactor"] = 0.5
        
        # Normal Map
        norm_tex = self.get_texture(mat, "normalCamera")
        if norm_tex is not None:
            pbr["normalTexture"] = {"index": norm_tex}
        
        # Emission
        try:
            em_weight = cmds.getAttr(f"{mat}.emission")
            if em_weight > 0.001:
                em_tex = self.get_texture(mat, "emissionColor")
                if em_tex is not None:
                    pbr["emissiveTexture"] = {"index": em_tex}
                else:
                    em_color = cmds.getAttr(f"{mat}.emissionColor")[0]
                    pbr["emissiveFactor"] = [em_color[0] * em_weight, em_color[1] * em_weight, em_color[2] * em_weight]
        except:
            pass
        
        # Opacity
        try:
            opacity_tex = self.get_texture(mat, "opacity")
            if opacity_tex is not None:
                # Opacity as texture - would need to be in base color alpha channel
                # This is complex, skipping for now
                pass
            else:
                opacity_vals = cmds.getAttr(f"{mat}.opacity")[0]
                avg_opacity = (opacity_vals[0] + opacity_vals[1] + opacity_vals[2]) / 3.0
                if avg_opacity < 0.999:
                    if "baseColorFactor" in pbr["pbrMetallicRoughness"]:
                        pbr["pbrMetallicRoughness"]["baseColorFactor"][3] = avg_opacity
                    pbr["alphaMode"] = "BLEND"
        except:
            pass
        
        # Clearcoat (KHR_materials_clearcoat extension)
        try:
            coat_weight = cmds.getAttr(f"{mat}.coat")
            if coat_weight > 0.001:
                clearcoat_ext = {}
                
                clearcoat_ext["clearcoatFactor"] = float(coat_weight)
                
                # Coat roughness
                try:
                    coat_roughness = cmds.getAttr(f"{mat}.coatRoughness")
                    clearcoat_ext["clearcoatRoughnessFactor"] = float(coat_roughness)
                except:
                    pass
                
                # Coat color (affects the tint)
                try:
                    coat_color = cmds.getAttr(f"{mat}.coatColor")[0]
                    # Note: GLTF clearcoat doesn't have a color, but we could encode it differently
                    # For now, just noting it exists
                except:
                    pass
                
                # Coat normal
                coat_norm_tex = self.get_texture(mat, "coatNormal")
                if coat_norm_tex is not None:
                    clearcoat_ext["clearcoatNormalTexture"] = {"index": coat_norm_tex}
                
                if clearcoat_ext:
                    if "extensions" not in pbr:
                        pbr["extensions"] = {}
                    pbr["extensions"]["KHR_materials_clearcoat"] = clearcoat_ext
                    self.extensions_used.add("KHR_materials_clearcoat")
        except:
            pass
        
        # Sheen (KHR_materials_sheen extension)
        try:
            sheen_weight = cmds.getAttr(f"{mat}.sheen")
            if sheen_weight > 0.001:
                sheen_ext = {}
                
                # Sheen color
                try:
                    sheen_color = cmds.getAttr(f"{mat}.sheenColor")[0]
                    sheen_ext["sheenColorFactor"] = [sheen_color[0] * sheen_weight, sheen_color[1] * sheen_weight, sheen_color[2] * sheen_weight]
                except:
                    sheen_ext["sheenColorFactor"] = [sheen_weight, sheen_weight, sheen_weight]
                
                # Sheen roughness
                try:
                    sheen_roughness = cmds.getAttr(f"{mat}.sheenRoughness")
                    sheen_ext["sheenRoughnessFactor"] = float(sheen_roughness)
                except:
                    pass
                
                if sheen_ext:
                    if "extensions" not in pbr:
                        pbr["extensions"] = {}
                    pbr["extensions"]["KHR_materials_sheen"] = sheen_ext
                    self.extensions_used.add("KHR_materials_sheen")
        except:
            pass
        
        return pbr
    
    def convert_legacy(self, mat, mat_type):
        """Convert legacy materials"""
        pbr = {"name": mat, "pbrMetallicRoughness": {"metallicFactor": 0.0}}
        
        tex = self.get_texture(mat, "color")
        if tex:
            pbr["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex}
        else:
            try:
                color = cmds.getAttr(f"{mat}.color")[0]
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [color[0], color[1], color[2], 1.0]
            except:
                pbr["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]
        
        pbr["pbrMetallicRoughness"]["roughnessFactor"] = 1.0 if mat_type == 'lambert' else 0.5
        return pbr
    
    def create_default(self, mat):
        """Default material"""
        return {
            "name": mat,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.5
            }
        }
    
    def get_texture(self, node, attr):
        """Get and embed texture"""
        try:
            full_attr = f"{node}.{attr}"
            if not cmds.objExists(full_attr):
                return None
            
            connections = cmds.listConnections(full_attr, source=True, destination=False)
            if not connections:
                return None
            
            file_node = None
            for conn in connections:
                if cmds.nodeType(conn) == 'file':
                    file_node = conn
                    break
                subs = cmds.listConnections(conn, source=True, destination=False, type='file')
                if subs:
                    file_node = subs[0]
                    break
            
            if not file_node:
                return None
            
            if file_node in self.texture_index_map:
                return self.texture_index_map[file_node]
            
            file_path = cmds.getAttr(f"{file_node}.fileTextureName")
            if not file_path or not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                tex_data = f.read()
            
            ext = os.path.splitext(file_path)[1].lower()
            mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'application/octet-stream')
            
            img_idx = len(self.gltf_data["images"])
            
            if self.is_glb:
                while len(self.binary_data) % 4 != 0:
                    self.binary_data.append(0)
                
                tex_offset = len(self.binary_data)
                self.binary_data.extend(tex_data)
                
                bv_idx = len(self.gltf_data["bufferViews"])
                self.gltf_data["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": tex_offset,
                    "byteLength": len(tex_data)
                })
                
                self.gltf_data["images"].append({
                    "mimeType": mime,
                    "bufferView": bv_idx
                })
            else:
                self.gltf_data["images"].append({
                    "uri": os.path.basename(file_path)
                })
            
            tex_idx = len(self.gltf_data["textures"])
            self.gltf_data["textures"].append({
                "sampler": 0,
                "source": img_idx
            })
            
            self.texture_index_map[file_node] = tex_idx
            return tex_idx
        except:
            return None
    
    
    def cleanup_empty_arrays(self):
        """Remove empty arrays from GLTF data (required for validation)"""
        # Remove empty animations array
        if "animations" in self.gltf_data and len(self.gltf_data["animations"]) == 0:
            del self.gltf_data["animations"]
        
        # Remove empty skins array
        if "skins" in self.gltf_data and len(self.gltf_data["skins"]) == 0:
            del self.gltf_data["skins"]
        
        # Remove empty extensions if no extensions were used
        if "extensionsUsed" in self.gltf_data and len(self.gltf_data["extensionsUsed"]) == 0:
            del self.gltf_data["extensionsUsed"]
    
    def write_glb(self, filepath):
        """Write GLB"""
        # Clean up empty arrays (GLTF validation requirement)
        self.cleanup_empty_arrays()
        
        while len(self.binary_data) % 4 != 0:
            self.binary_data.append(0)
        
        self.gltf_data["buffers"].append({"byteLength": len(self.binary_data)})
        
        json_data = json.dumps(self.gltf_data, separators=(',', ':'))
        json_bytes = json_data.encode('utf-8')
        
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '
        
        total = 12 + 8 + len(json_bytes) + 8 + len(self.binary_data)
        
        header = struct.pack('<III', 0x46546C67, 2, total)
        json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A)
        bin_chunk = struct.pack('<II', len(self.binary_data), 0x004E4942)
        
        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(json_chunk)
            f.write(json_bytes)
            f.write(bin_chunk)
            f.write(self.binary_data)
        
        return os.path.exists(filepath)
    
    def write_gltf(self, filepath):
        """Write GLTF"""
        # Clean up empty arrays (GLTF validation requirement)
        self.cleanup_empty_arrays()
        bin_path = os.path.splitext(filepath)[0] + '.bin'
        bin_name = os.path.basename(bin_path)
        
        with open(bin_path, 'wb') as f:
            f.write(self.binary_data)
        
        self.gltf_data["buffers"].append({
            "uri": bin_name,
            "byteLength": len(self.binary_data)
        })
        
        with open(filepath, 'w') as f:
            json.dump(self.gltf_data, f, indent=2)
        
        return os.path.exists(filepath)


# ============================================================================
# UI WITH ANIMATION CONTROLS
# ============================================================================

class GLTFExporterUI:
    def __init__(self):
        self.window_name = "mayaGltfExporter"
        
    def show(self):
        if cmds.window(self.window_name, exists=True):
            cmds.deleteUI(self.window_name)
        
        window = cmds.window(self.window_name, title=f"GLTF Exporter v{VERSION}", 
                            widthHeight=(420, 440))
        
        cmds.columnLayout(adjustableColumn=True, rowSpacing=8)
        cmds.text(label=f"GLTF/GLB Exporter v{VERSION}", font="boldLabelFont", height=30, 
                 backgroundColor=[0.2, 0.2, 0.2])
        cmds.text(label=VERSION_DATE, font="smallPlainLabelFont", height=15)
        cmds.separator(height=10, style='none')
        
        # Format
        cmds.frameLayout(label="Format", collapsable=False, marginHeight=10)
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(200, 200))
        self.format_radio = cmds.radioCollection()
        cmds.radioButton('rb_glb', label='GLB (Binary)', select=True)
        cmds.radioButton('rb_gltf', label='GLTF (JSON)')
        cmds.setParent('..')
        cmds.setParent('..')
        
        # Export Options
        cmds.frameLayout(label="Export Options", collapsable=False, marginHeight=10)
        cmds.columnLayout(adjustableColumn=True, rowSpacing=5)
        self.selection_check = cmds.checkBox(label="Export Selected Only", value=False)
        self.animation_check = cmds.checkBox(label="Export Animation", value=False, 
                                            changeCommand=self.toggle_animation_options)
        cmds.setParent('..')
        cmds.setParent('..')
        
        # Animation Options
        self.anim_frame = cmds.frameLayout(label="Animation Options", collapsable=False, 
                                          marginHeight=10, enable=False)
        cmds.columnLayout(adjustableColumn=True, rowSpacing=5)
        
        self.bake_check = cmds.checkBox(label="Bake Animation", value=False)
        self.force_bake_check = cmds.checkBox(
            label="Force Bake All (export even without keyframes)", 
            value=False,
            annotation="Bake all transforms even if they don't have keyframes"
        )
        
        cmds.separator(height=5, style='in')
        
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(200, 200))
        self.timeline_radio = cmds.radioCollection()
        cmds.radioButton('rb_timeline', label='Use Timeline Range', select=True,
                        onCommand=self.toggle_custom_range)
        cmds.radioButton('rb_custom', label='Custom Range',
                        onCommand=self.toggle_custom_range)
        cmds.setParent('..')
        
        cmds.rowLayout(numberOfColumns=4, columnWidth4=(80, 80, 80, 80))
        cmds.text(label="Start Frame:")
        self.start_field = cmds.intField(value=1, enable=False)
        cmds.text(label="End Frame:")
        self.end_field = cmds.intField(value=100, enable=False)
        cmds.setParent('..')
        
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(150, 100))
        cmds.text(label="Sample Every N Frames:")
        self.sample_field = cmds.intField(value=1, minValue=1)
        cmds.setParent('..')
        
        cmds.setParent('..')
        cmds.setParent('..')
        
        # Output
        cmds.frameLayout(label="Output", collapsable=False, marginHeight=10)
        cmds.rowLayout(numberOfColumns=2, columnWidth2=(340, 60), adjustableColumn=1)
        self.path_field = cmds.textField(placeholderText="Choose output file...")
        cmds.button(label="Browse", command=self.browse)
        cmds.setParent('..')
        cmds.setParent('..')
        
        cmds.separator(height=10, style='none')
        cmds.button(label="EXPORT", height=40, backgroundColor=[0.3, 0.5, 0.3], 
                   command=self.do_export)
        cmds.text(label="Supports: openPBRSurface • standardSurface • Lambert/Blinn/Phong", 
                 font="smallPlainLabelFont", height=15)
        cmds.text(label="Animation • Skinning • Textures", 
                 font="smallPlainLabelFont", height=15)
        
        cmds.showWindow(window)
        
        print(f"Maya GLTF Exporter v{VERSION}")
        print("="*60)
        print("Features:")
        print("  • openPBRSurface & standardSurface materials")
        print("  • Transform & skeletal animation")
        print("  • Frame baking with timeline controls")
        print("  • Texture embedding (GLB)")
        print("  • Fixed axis and pivot point handling")
        print("="*60 + "\n")
    
    def toggle_animation_options(self, value):
        """Enable/disable animation options"""
        cmds.frameLayout(self.anim_frame, edit=True, enable=value)
    
    def toggle_custom_range(self, *args):
        """Enable/disable custom range fields"""
        use_timeline = cmds.radioCollection(self.timeline_radio, query=True, 
                                           select=True) == 'rb_timeline'
        cmds.intField(self.start_field, edit=True, enable=not use_timeline)
        cmds.intField(self.end_field, edit=True, enable=not use_timeline)
    
    def browse(self, *args):
        fmt = cmds.radioCollection(self.format_radio, query=True, select=True)
        filter_str = "*.glb" if fmt == 'rb_glb' else "*.gltf"
        result = cmds.fileDialog2(fileFilter=filter_str, dialogStyle=2, fileMode=0, 
                                 caption="Save GLTF/GLB")
        if result:
            cmds.textField(self.path_field, edit=True, text=result[0])
    
    def do_export(self, *args):
        filepath = cmds.textField(self.path_field, query=True, text=True)
        if not filepath:
            cmds.confirmDialog(title='Error', message='Specify output file', button=['OK'])
            return
        
        fmt = cmds.radioCollection(self.format_radio, query=True, select=True)
        export_format = 'glb' if fmt == 'rb_glb' else 'gltf'
        base = os.path.splitext(filepath)[0]
        filepath = f"{base}.{export_format}"
        
        selection_only = cmds.checkBox(self.selection_check, query=True, value=True)
        export_anim = cmds.checkBox(self.animation_check, query=True, value=True)
        bake_anim = cmds.checkBox(self.bake_check, query=True, value=True)
        force_bake_all = cmds.checkBox(self.force_bake_check, query=True, value=True)
        
        use_timeline = cmds.radioCollection(self.timeline_radio, query=True, 
                                           select=True) == 'rb_timeline'
        start_frame = cmds.intField(self.start_field, query=True, value=True) if not use_timeline else None
        end_frame = cmds.intField(self.end_field, query=True, value=True) if not use_timeline else None
        sample_rate = cmds.intField(self.sample_field, query=True, value=True)
        
        try:
            exporter = GLTFExporter()
            success = exporter.export(filepath, export_format, selection_only,
                                    export_anim, bake_anim, force_bake_all, use_timeline,
                                    start_frame, end_frame, sample_rate)
            
            if success:
                msg = f'✓ Export complete!\n\n{filepath}'
                if export_anim:
                    msg += f'\n\nAnimation: {exporter.start_frame} - {exporter.end_frame} frames'
                cmds.confirmDialog(title='Success', message=msg, button=['OK'])
            else:
                cmds.confirmDialog(title='Failed', message='Export failed. Check Script Editor.', 
                                 button=['OK'], icon='warning')
        except Exception as e:
            cmds.confirmDialog(title='Error', message=f'{str(e)}', button=['OK'], icon='critical')
            traceback.print_exc()

# Launch
ui = GLTFExporterUI()
ui.show()