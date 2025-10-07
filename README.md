# Laser Cross Calibration

A precision laser cross-calibration system for volumetric PIV (Particle Image Velocimetry) camera calibration using ray tracing through complex optical geometries.

## Goal



## Surface Normal and Material Logic

This system uses a pure interface-based approach for optical ray tracing, where materials are defined at interfaces rather than as ambient media.

### Interface Material Definition

Each `OpticalInterface` defines two materials:
- **`material_pre`**: Material on the "from" side of the interface
- **`material_post`**: Material on the "to" side of the interface

The **surface normal direction** is crucial and points **toward `material_pre` (from `material_post`)**:

```
material_pre  =>  [Interface]  =>  material_post
     air      =>  (surface)   =>      glass
                     <-
               normal points this way
```

### Checking Surface Normals in Blender

When working with STL files in Blender, you can visualize surface normals to ensure correct material assignment:

![Surface normal visualization in Blender](readme-images/screenshot-blender-normals.png)

**Steps to check normals in Blender:**

1. **Import your STL file**: `File � Import � STL`
2. **Enter Edit Mode**: Select object, press `Tab` or click "Edit Mode"
3. **Enable face normal display**:
   - Open overlay options (overlapping circles icon in top-right)
   - Check "Face Orientation" to see red/blue faces
   - Check "Normals" and set to "Face" to see normal vectors
4. **Verify normal direction**:
   - Blue faces = normals pointing toward camera (outward)
   - Red faces = normals pointing away from camera (inward)
   - Normal arrows show exact direction

**Correct normal orientation for optical interfaces:**
- Normals should point from the **incoming medium** to the **outgoing medium**
- For air-to-glass interfaces: normal points from air side into glass
- For glass-to-air interfaces: normal points from glass side into air

### Ray Tracing Logic

When a ray hits an interface:

1. **Determine current medium**: Based on ray's travel history
2. **Check approach direction**: Dot product of ray direction with surface normal
3. **Apply refraction**: Using Snell's law with `material_pre` and `material_post` refractive indices
4. **Update ray medium**: Ray continues in the appropriate material based on interface definition

This approach eliminates ambiguity about "ambient" materials and provides physically accurate ray propagation through complex optical systems.