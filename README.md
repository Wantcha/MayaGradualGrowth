# Maya Gradual Growth Python Script
Animated mesh population in Maya is now at the tip of your fingers!

![thumbnail](https://user-images.githubusercontent.com/64153611/110856040-ccb96a80-82bf-11eb-8b77-29abb6124823.png)


## How to run and use the script
Make sure to set the workspace of your project somewhere. Drag the contents of the "script" folder from this repository into the "scripts" folder of your project. Then, you can open it in Maya.

Select the mesh you wish to grow things on and run this script.

Go into the **Modify** tab and click on the options box next to the **Paint Attributes Tool**.

Use the Weight Paint tools to paint the influence you want the growth to have over your mesh. If you just want to use the whole mesh, then Flood it with the value 1.0.

Values higher than 1.0 will result in meshes bigger than the ones provided. Treat the weight as a scale multiplier!

You can move, scale and animate the _ControlSphere to influence how the growth animation will play out, based on the Growth Animation Type.

In the interface, select the names of the submeshes you wish to grow, as well as the distribution, animation type and frame range and click the big button at the bottom. Then just wait for the magic to happen behind the scenes.

## How to control the parameters in the interface

![Menu](https://user-images.githubusercontent.com/64153611/110856440-4cdfd000-82c0-11eb-9ce9-bb4cbc263241.png)

### Submeshes names

Type in the names of the submeshes you want to populate with. You can use both spaces and endlines to separate them.

### Distribution type

The way the submeshes will be distributed on the base mesh.

1. Random: submeshes will be scattered randomly on the surface.
2. Aligned: Submeshes will be generated in 3 grid-like structures, each of them parallel to 2 of the main world axis, and then projected onto the base mesh. This method is similar to Tri-Planar projections used in texturing. This option is very good if you wish to cover the entire base mesh uniformly
3. Evenly Spaced: The submeshes will attempt to distribute themselves as evenly as possible, based on the distance between them on the surface. WARNING: this method is very slow, even for small amounts of submeshes, use at your own expense.

### Random Yaw Rotation

Tick this if you wish for the submeshes to recieve a random rotation along the local Y axis after they are placed.

### Number of Instances

If the distribution type is either Random, or Evenly Spaced, you can specify the exact number of submeshes you wish to instantiate.

### Instances along XYZ-axis

If the distribution type Aligned, you can specify how many instances are placed along the grids in each axis. For example, if we want to modify the grid that will get projected along the Z axis, we will increase or decrease the instances along the X and Y axis, which will determine how many submeshes will attempt to be projected on the width and the height of the grid plane.

### Animated Base Mesh

Tick this if your base mesh is to be deformed.

### Growth Animation Type

1. Uniform: All submeshes wil grow at the same rate starting at the same time
2. Sphere Mask: The _ControlSphere object will act as mask for the growth of the objects. Choosing **Inside Sphere** below will make the submeshes start growing only when they are inside the radius and modify their scale based on the distance to the sphere's center. Choosing **Outside Sphere** will have the opposite effect.
3. Along surface: The _ControlSphere object will detect the closest point to the base mesh and set it a "Starting Point". The submeshes will start to grow based on the surface distance to the "Starting Point". If the **Inwards Towards Sphere** option is selected below, the submeshes will start growing from the furthest distances on the mesh and close in on the "Starting Point". Choosing **Outwards From Sphere** will make the submeshes begin at the "Starting Point" and continue to grow towards the ends of the mesh. The **Submesh Growth Speed** represents the speed of the growth of one submesh. The amount of frames it would take one submesh to grow is described by the formula frameRange/growthSpeed.

### Frame Range

Define when should the script start growing the submeshes and when to end.

## (IMPORTANT) Script Limitations

If you wish to choose to tick the **Animated Base Mesh** option, you must make sure the mesh is UV unwrapped. A simple automatic unwrap should work fine, but the UV's aren't layed decently, the submeshes will not be placed properly.

If you choose the **Along Surface** animation type, make sure the base mesh has no loose parts, meaning all of the vertices are connected. Unexpected behaviour will occur if there are multiple separate pieces combined into one object.

