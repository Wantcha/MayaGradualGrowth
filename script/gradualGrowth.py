for name in dir():
    if not name.startswith('_'):
        try:
            del globals()[name]
        except:
            pass

import maya.cmds as mc
import random as rand
import math as m
import functools as ft
from maya.OpenMaya import MVector
import maya.OpenMaya as om
import re
import heapq
import os

'''
This is a tool meant to produce a gradual animated growth effect of smaller submeshes onto a bigger base mesh.
The interface and script provide the user with various ways to customize the look and placement of the growth effect, including
the ability to paint the influence of the growth, adjusting the parameters of the animation, or using a Control Sphere object to
visually specify how the effect should act.
'''

def calculate_distances(graph, starting_vertex):
    ''' Dijkstra pathfinding algorithm applied to mesh vertices.
    
        - graph: the graph constructed from the base mesh's vertices
        - starting_vertex: ID of the vertex we calculate the distances from '''
    
    
    distances = [999999] * len( graph )
    distances[starting_vertex] = 0

    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
    

class Graph():
    '''
    Graph data structure meant to retain information about a mesh.
    '''

    def __init__(self, size):
        self.vertexArray = [None] * size
        self.numVertices = 0
        
    def addVertex(self, node):
        if self.vertexArray[node] == None:
            self.numVertices += 1
            self.vertexArray[node] = []
        
    def addEdge(self, start, end, length):
        self.vertexArray[start].append( (end, length) )
        self.vertexArray[end].append( (start, length) )
        
    def printGraph(self):
        for i in range(self.numVertices):
            print (i, self.vertexArray[i])
        
    def getVertices(self):
        return self.vertexArray        
    

def distance( point1, point2 ):
    return m.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
    
def squareDistance( point1, point2 ):
    return (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2

def float_range(start, end, step):
    '''
    Custom Range function so we can iterate through "FOR" loops with a float iterator.
    '''
    while start <= end:
        yield start
        start += step

def returnCollisionPoints(point, dir, fnMesh):
    ''' Get collision information from a raycast onto a mesh

    - point: origin point in space of the ray
    - dir: direction of the ray
    - fnMesh: mesh used for the collision testing '''
    point = om.MFloatPoint(*point)
    
    hitPoints = om.MFloatPointArray()
    hitFaces = om.MIntArray()
    
    fnMesh.allIntersections(
        point, dir,
    		None, None,
    		False, om.MSpace.kWorld,
    		10000, False,
    		fnMesh.autoUniformGridParams(), False,
    		hitPoints,
    		None, hitFaces,
    		None, None,
    		None
       )       
    return hitPoints, hitFaces

def getNormal(face_):
    ''' Extract the normal vector in a MVector from Maya Python's string format

    - face_: ID of the face we are getting the normal of '''
    normal = mc.polyInfo(face_, faceNormals = True)
    
    # convert the string to array with regular expression
    normal = re.findall(r"[\w.-]+", normal[0])
    
    normal = MVector( float(normal[2]),\
                      float(normal[3]),\
                      float(normal[4]) )
    return normal

def getClosestVertex( objectPos, baseMesh ):
    ''' Retrieve the vertex of a mesh that is closest to a specified position

    - objectPos: position in world space
    - baseMesh: mesh we are getting the vertex from '''
    object = om.MPoint( objectPos[0], objectPos[1], objectPos[2] )

    startingPoint = om.MPoint()
    faceIdxUtil = om.MScriptUtil()
    faceIdxUtil.createFromInt(-1)
    faceIntPtr = faceIdxUtil.asIntPtr()

    # Get closest face ID
    baseMesh.getClosestPoint(object, startingPoint, om.MSpace.kWorld, faceIntPtr)
    faceIdx = faceIdxUtil.getInt(faceIntPtr)
    
    faceVertices = om.MIntArray()
    
    baseMesh.getPolygonVertices( faceIdx, faceVertices )
                    
    # setting vertex [0] as the closest one
    vert = om.MPoint()
    baseMesh.getPoint(faceVertices[0], vert)
    smallestDist = vert.distanceTo(object)
    closestID = faceVertices[0]
    
    v = om.MPoint()
    
    for i in range (1, len(faceVertices)):
        baseMesh.getPoint(faceVertices[0], v)
        dist = v.distanceTo(object)
        if(dist < smallestDist):
            smallestDist = dist
            closestID = faceVertices[i]
    return closestID

def animateAlongSurface( objects, inwards, startFrame, frameRange, growthSpeed):
    ''' Animate the scale of objects based on their distance from a point

    - objects: list of objects we want to animate
    - inwards: whether the scale should increase more rapidly the farther they are from the starting point or closer
    - startFrame: frame to start the animation on
    - frameRange: duration of the animation in frames
    - growthSpeed: how fast should an object scale '''
    objSizes = []
    
    for obj in objects:
        objSizes.append( mc.xform(obj, q = True, ws = True, s = True) )
    
    graph = g.getVertices()
    spherePos = mc.xform(controlSphereName, q = True, ws = True, rp = 1)
    
    
    sel = om.MSelectionList()
    dag = om.MDagPath()
    sel.add(baseMeshName)
    sel.getDagPath(0, dag)
    fnMesh = om.MFnMesh(dag)
    
    # Use the closest vertex on the base mesh from the control sphere as the starting point
    startPoint = getClosestVertex( spherePos, fnMesh )

    graphDistances = calculate_distances( graph, startPoint )
    objectDistances = []
    maxDist = -1
    for i in range ( len(graphDistances) ):
        if graphDistances[i] > maxDist:
            maxDist = graphDistances[i]
    
    # Get the closest vertex on the base mesh from each submesh and use them to get the geodesic distance from the
    # starting point to them
    for i in range(len(objects)):
        objPos = mc.xform(objects[i], q = True, ws = True, rp = 1) 
        objectDistances.append( graphDistances[ getClosestVertex( objPos, fnMesh ) ] / maxDist )
    

    growthFrames = frameRange / growthSpeed
    
    mc.progressWindow(title = 'Animation Progress', progress = 0, status = 'Animating...', isInterruptable = True, max = len(objects))
    if inwards == False:
        for j in range ( len( objects ) ):
            mc.select(objects[j])
            delay = (frameRange - growthFrames + 1) * objectDistances[j]
            mc.currentTime( delay, edit = True )
            mc.setKeyframe( v = 0, at = 'scaleX')
            mc.setKeyframe( v = 0, at = 'scaleY')
            mc.setKeyframe( v = 0, at = 'scaleZ')
            
            mc.currentTime( delay + growthFrames, edit = True )
            mc.setKeyframe( v = objSizes[j][0], at = 'scaleX')
            mc.setKeyframe( v = objSizes[j][1], at = 'scaleY')
            mc.setKeyframe( v = objSizes[j][2], at = 'scaleZ')
            if mc.progressWindow (q = True, isCancelled = True):
                break
            mc.progressWindow(edit = True, step = 1)
    else:
        for j in range ( len( objects ) ):
            mc.select(objects[j])
            delay = (frameRange - growthFrames + 1) * (1 - objectDistances[j])
   
            mc.currentTime( delay, edit = True )
            mc.setKeyframe( v = 0, at = 'scaleX')
            mc.setKeyframe( v = 0, at = 'scaleY')
            mc.setKeyframe( v = 0, at = 'scaleZ')
            
            mc.currentTime( delay + growthFrames, edit = True )
            mc.setKeyframe( v = objSizes[j][0], at = 'scaleX')
            mc.setKeyframe( v = objSizes[j][1], at = 'scaleY')
            mc.setKeyframe( v = objSizes[j][2], at = 'scaleZ')
            if mc.progressWindow (q = True, isCancelled = True):
                break
            mc.progressWindow(edit = True, step = 1)
    mc.progressWindow(endProgress = 1)

def animateWithSphere(startFrame, frameRange, objects, insideSphere):
    ''' Animate the scale of objects based on their distance to the _ControlSphere

    - objects: list of objects we want to animate
    - startFrame: frame to start the animation on
    - frameRange: duration of the animation in frames
    - insideSphere: whether objects should scale up the more they are inside the sphere, or the opposite '''
    
    objPositions = []
    objSizes = []
    
    for obj in objects:
        objPositions.append( mc.xform(obj, q = True, ws = True, rp = 1) )
        objSizes.append( mc.xform(obj, q = True, ws = True, s = True) )
    
    mc.progressWindow(title = 'Animation Progress', progress = 0, status = 'Animating...', isInterruptable = True, max = frameRange)
    for j in range(frameRange + 1):
        mc.currentTime( startFrame + j, edit = True )
        spherePos = mc.xform(controlSphereName, q = True, ws = True, rp = 1)
        sphereSize = mc.xform(controlSphereName, q = True, ws = True, s = True)
        sphereRadius = sphereSize[0]
        
        for z in range(len(objects)):
            
            pointDistanceToSphere = distance(objPositions[z], spherePos)
            
            # Whether objects should grow bigger the closer within the sphere they are, or the
            # farther away from it they are
            if insideSphere == False:
                sphereInfluence = pointDistanceToSphere / sphereRadius - 1
                if sphereInfluence > 1:
                    sphereInfluence = 1
                if sphereInfluence < 0:
                    sphereInfluence = 0
            
            else:
                sphereInfluence = 1 - pointDistanceToSphere / sphereRadius
                if sphereInfluence < 0:
                    sphereInfluence = 0
                    
            # Testing has proven that a square root dropoff from the center of the sphere generates better results
            # than a linear one, which is why apply this operation
            sphereInfluence = m.sqrt(sphereInfluence)
            mc.select(objects[z])
            mc.setKeyframe( v = objSizes[z][0] * sphereInfluence, at = 'scaleX')
            mc.setKeyframe( v = objSizes[z][1] * sphereInfluence , at = 'scaleY')
            mc.setKeyframe( v = objSizes[z][2] * sphereInfluence , at = 'scaleZ')
            if mc.progressWindow (q = True, isCancelled = True):
                break
        mc.progressWindow(edit = True, step = 1)
    mc.progressWindow(endProgress = 1)

def animateNoSphere(startFrame, endFrame, objects):
    ''' Animate the scale of objects uniformly

    - objects: list of objects we want to animate
    - startFrame: frame to start the animation on
    - frameRange: duration of the animation in frames '''
    
    mc.currentTime( endFrame, edit = True )
    for obj in objects:
        objSize = mc.xform(obj, q = True, ws = True, s = True)
        mc.select(obj)
        
        mc.setKeyframe( v = objSize[0] , at = 'scaleX')
        mc.setKeyframe( v = objSize[1] , at = 'scaleY')
        mc.setKeyframe( v = objSize[2] , at = 'scaleZ')
        
    mc.currentTime( startFrame, edit = True )
    for obj in objects:
        mc.select(obj)
        mc.setKeyframe( v = 0 , at = 'scaleX')
        mc.setKeyframe( v = 0 , at = 'scaleY')
        mc.setKeyframe( v = 0 , at = 'scaleZ')

def getAverageWeight(vertices):
    ''' Get the average weight of 3 vertices

    - vertices: the vertices whose weights are to be used to get the average '''
    
    index1 = int(re.findall(r"[\w]+", vertices[0])[2])
    index2 = int(re.findall(r"[\w]+", vertices[1])[2])
    index3 = int(re.findall(r"[\w]+", vertices[2])[2])
    
    weights = mc.getAttr(baseMeshShape + '.growthWeights')
    return (weights[index1] + weights[index2] + weights[index3]) / 3.0

def placeInstance(instanceName, position, size, face, randRotation, animatedParentGroup = None):
    ''' Instantiate an object and place, scale and rotate it at the desired position.

    - instanceName: name of the object to be instantiated
    - position: where to place the instance
    - size: how big should the instance be
    - face: polygon the instance is to be placed on
    - randRotation: if the instance should receive a random rotation on the local Y axis
    - animatedParentGroup: if the instance should be parented to something (required in the case of an animated base mesh) '''
    faceNormal = getNormal(face)
    
    psi = mc.angleBetween(euler = True, v1 = (0.0, 1.0, 0.0),
                            v2 = (faceNormal.x, faceNormal.y, faceNormal.z))
        
    objectInstance = mc.instance(instanceName)
    
    mc.move( position[0], position[1], position[2], objectInstance )
    mc.rotate( psi[0], psi[1], psi[2], objectInstance )
    if randRotation:
        mc.rotate( 0, rand.uniform(-180, 180), 0, objectInstance, r = True, os = True )
    
    mc.scale( size, size, size, objectInstance )
    
    if animatedParentGroup != None:
        mc.select(face, r = True)
        mc.pointOnPolyConstraint( face, animatedParentGroup, mo = False )
        mc.parent( objectInstance, animatedParentGroup )
        mc.parent( animatedParentGroup, groupName )
    else:
        mc.parent ( objectInstance, groupName )

def getTriangleFromHitpoint(hitPoint, faceID):
    ''' Given a polygon, return the vertices of a triangle within it based on their distance to the specified position.
    
    - hitPoint: position in space to calculate distance from
    - faceID: ID of the polygon '''
    
    mc.select(baseMeshName + '.f[' + str(faceID) + ']')
    mc.select( mc.polyListComponentConversion(tv = True) )
    selectedVerts = mc.ls( sl = True, fl = True )
    
    # If the polygon is a quad, remove the vertex that is furthest away from the hitPoint, so we can generate
    # a triangle out of the 3 closest vertices
    if len(selectedVerts) == 4:       
        maxDist = -1
        distID = -1
        for z in range(len(selectedVerts)):
            dist = squareDistance( mc.pointPosition(selectedVerts[z]), hitPoint )
            if dist > maxDist:
                maxDist = dist
                distID = z
        del selectedVerts[distID]
    return selectedVerts

def calculateAndInstantiate(subMeshName, hitPoint, hitFace, direction, randRotation):
    ''' In the case of Aligned Distribution Type, check whether a submesh should be placed after a raycast has hit the base mesh.
    
    - subMeshName: name of the submesh to be instantiated
    - hitPoint: position of the raycast intersection
    - hitFace: ID of the face where the raycats intersection occured
    - direction: direction of the raycast
    - randRotation: whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated
    - animatedParentGroup: if the submesh should be parented to something after being instantiated (required in the case of an animated base mesh) '''
    
    # If the base mesh is animated, for the pointOnPoly contraint to work properly, submeshes need to be parented to
    # a dummy object, in this case, an empty group
    animatedGroupName = None
    if isBaseAnimated:
        animatedGroupName = mc.group(em = True)
    
    selectedVerts = getTriangleFromHitpoint(hitPoint, hitFace)
                        
    averageWeight = getAverageWeight(selectedVerts)
    face = baseMeshName + '.f[' + str(hitFace) + ']'
    faceNormal = getNormal(face)
                        
    dotProduct = faceNormal[0] * direction[0] + faceNormal[1] * direction[1] + faceNormal[2] * direction[2]
                        
    # Only place an instance if the average weight of the triangle is above a certain threshold and the angle between the
    # normal of the face and the direction of the ray is small enough (otherwise we sometimes get odd collisions with faces
    # that are facing far away from the ray and we get overlaps between the "Imaginary Plane" projections)
    if averageWeight > 0.05 and abs(dotProduct) > 0.6:
        placeInstance(subMeshName, hitPoint, averageWeight, face, randRotation, animatedGroupName)
    else:
        mc.delete(animatedGroupName)

        
def calculateFaceAreaArray( sampleArray ):
    ''' Generate a list of polygon IDs, where the number of occurence of a poly is proportional to its area (polygons with bigger areas will be added multiple times).
    
    - sampleArray: given list to add elements into  
    
    Idea for this approach taken from: http://www.joesfer.com/?p=84'''
    
    faces = mc.polyEvaluate( baseMeshName, f = True )

    sel = om.MSelectionList()
    dag = om.MDagPath()
    sel.add(baseMeshName)
    sel.getDagPath(0, dag)
    meshFn = om.MFnMesh(dag)
    areas = []
    minArea = 999999

    # Get positions of all vertices on the mesh
    meshFn = om.MFnMesh(dag)
    positions = om.MPointArray()
    meshFn.getPoints(positions, om.MSpace.kWorld)
    
    # Calculate the areas of every polygon on the mesh
    for i in range( faces ):
        indices = om.MIntArray()
        meshFn.getPolygonVertices(i, indices)
        area = 0
        if len(indices) == 3:
            AB = distance( positions[ indices[0] ], positions[ indices[1] ] )
            BC = distance( positions[ indices[1] ], positions[ indices[2] ] )
            AC = distance( positions[ indices[2] ], positions[ indices[0] ] )
            s = (AB + BC + AC ) / 2.0
            area += m.sqrt( s * (s - AB) * (s - BC) * (s - AC) )
        else:
            AB = distance( positions[ indices[0] ], positions[ indices[1] ] )
            BC = distance( positions[ indices[1] ], positions[ indices[2] ] )
            CD = distance( positions[ indices[2] ], positions[ indices[3] ] )
            AD = distance( positions[ indices[3] ], positions[ indices[0] ] )
            AC = distance( positions[ indices[0] ], positions[ indices[2] ] )
            s = (AB + BC + AC ) / 2.0
            area += m.sqrt( s * (s - AB) * (s - BC) * (s - AC) )
            s = (AD + CD + AC ) / 2.0
            area += m.sqrt( s * (s - AD) * (s - CD) * (s - AC) )
        
        areas.append(area)
        if area < minArea:
            minArea = area

    for i in range( len(areas) ):
        areas[i] = int(m.ceil( areas[i] / minArea * 0.6 ))
        if areas[i] > 100:
            areas[i] = 100
        for j in range(areas[i]):
            sampleArray.append(i)
    

def randomPopulation(subMeshNames, numberOfInstances, randomRotation):  
    ''' Populate the base mesh with submeshes randomly
    
    - subMeshNames: list of names of the submeshes
    - numberOfInstances: how many submeshes should be instantiated
    - randRotation: whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated '''
     
    sampleArray = []
    calculateFaceAreaArray( sampleArray )
    
    mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating...', isInterruptable = True, max = numberOfInstances)
    for i in range(numberOfInstances):
        
        # If the base mesh is animated, for the pointOnPoly contraint to work properly, submeshes need to be parented to
        # a dummy object, in this case, an empty group
        
        randomFace = rand.choice( sampleArray )
        randomFace = baseMeshName + '.f[' + str(randomFace) + ']'
        mc.select(randomFace, r = True)
        
        mc.select( mc.polyListComponentConversion(tv = True) )
        selectedVerts = mc.ls( sl = True, fl = True )
        
        # Pick a random triangle within the selected polygon
        currentTriangle = rand.sample(selectedVerts, 3)
        
        averageWeight = getAverageWeight(currentTriangle)
        while averageWeight < 0.05:
            randomFace = rand.choice( sampleArray )
            randomFace = baseMeshName + '.f[' + str(randomFace) + ']'
            mc.select(randomFace, r = True)
        
            mc.select( mc.polyListComponentConversion(tv = True) )
            selectedVerts = mc.ls( sl = True, fl = True )
        
            # Pick another random triangle within the selected polygon
            currentTriangle = rand.sample(selectedVerts, 3)
        
            averageWeight = getAverageWeight(currentTriangle)
        # Get random point within this face
        # https://math.stackexchange.com/questions/538458/triangle-point-picking-in-3d
        v1 = MVector(*mc.pointPosition(currentTriangle[0], w = True))
        v2 = MVector(*mc.pointPosition(currentTriangle[1], w = True))
        v3 = MVector(*mc.pointPosition(currentTriangle[2], w = True))
        
        a = rand.random()
        b = rand.random()
        
        if a + b >= 1:
            a = 1-a
            b = 1-b
            
        randomPoint = v1 + ( v2 - v1) * a + ( v3 - v1 ) * b
        
        animatedGroupName = None
        if isBaseAnimated:
            animatedGroupName = mc.group(em = True)
        
        placeInstance(rand.choice(subMeshNames), randomPoint, averageWeight, randomFace, randomRotation, animatedGroupName)
        
        if mc.progressWindow (q = True, isCancelled = True):
            break
        mc.progressWindow(edit = True, step = 1)

    mc.progressWindow(endProgress = 1)

def projectPlane(originX, originY, originDepth, width, height, dir, subMeshNames, cellWidth, cellHeight, pWindow, randRotation):
    ''' Given information about an "Imaginary Plane", perpendicular to one of the 3 global axis, we perform raycasts from different points of his to the base mesh.
    
    - originX: first coordinate of the imaginary plane
    - originY: second coordinate of the imaginary plane
    - originDepth: third coordinate of the imaginary plane
    - width: width of the plane in world space units
    - height: height of the plane in world space units
    - dir: direction the plane is facing
    - subMeshNames: list of names of the submeshes
    - cellWidth: width of a plane's subdivision in world space units
    - cellHeight: height of a plane's subdivision in world space units
    - pWindow: progress window
    - randRotation: whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated '''
    
    point = [0, 0, 0]
    dirVector = om.MFloatVector(*dir)
    
    # Check in which of the 3 global axis the plane is facing. We will iterate through the imaginary plane's vertices in a 2D
    # fashion, only using the cellWidth and cellHeight. However, we have to turn these coordinates in world space, based on 
    # the global axis we've just identified, so we use these Index variables to figure out which axis does each of the 
    # provided parameters actually represent.
    if abs(dir[2]) == 1:
        iIndex = 0
        jIndex = 1
        originDepthIndex = 2
    elif abs(dir[1]) == 1:
        iIndex = 0
        jIndex = 2
        originDepthIndex = 1
    else:
        iIndex = 2
        jIndex = 1
        originDepthIndex = 0
    
    sel = om.MSelectionList()
    dag = om.MDagPath()
    sel.add(baseMeshName)
    sel.getDagPath(0, dag)
    
    mesh = om.MFnMesh(dag)
    for i in float_range(originX, width, cellWidth):
        for j in float_range(originY, height, cellHeight):
            point[iIndex] = i
            point[jIndex] = j
            point[originDepthIndex] = originDepth

            hitPoints, hitFaces = returnCollisionPoints(point, dirVector, mesh)
                
            for z in range(hitPoints.length()):
                calculateAndInstantiate(rand.choice(subMeshNames), hitPoints[z], hitFaces[z], dir, randRotation)
            if mc.progressWindow (pWindow, q = True, isCancelled = True):
                break

            mc.progressWindow(pWindow, edit = True, status= ('Populating mesh...'), step = 1)

def alignedPopulation(subMeshNames, width, height, depth, randRotation):
    ''' Populate the base mesh by projecting rays from uniformly subdivided imaginary planes perpendicular to the 3 global axis.
    
    - subMeshNames: list of names of the submeshes
    - width: number of subdivisions along the X global axis
    - height: number of subdivisions along the Y global axis
    - depth: number of subdivisions along the X global axis
    - randRotation: whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated '''
    
    bbox = mc.exactWorldBoundingBox(baseMeshName)
    
    planeOriginX = bbox[0]
    planeOriginY = bbox[1]
    planeOriginZ = bbox[2]
    planeEndX = bbox[3]
    planeEndY = bbox[4]
    planeEndZ = bbox[5]
    
    planeWidth = abs( planeEndX - planeOriginX )
    planeHeight = abs( planeEndY - planeOriginY )
    planeDepth = abs( planeEndZ - planeOriginZ )
   
    # We take the mesh bounding box and treat each side of it as one of the 3 "Imaginary Planes". We then calculate how big
    # each subdivision would be on each axis.
    cellWidth = planeWidth / width
    cellHeight = planeHeight / height
    cellDepth = planeDepth / depth
    
    totalRayCasts = (width + 1) * (height + 1) + (width + 1) * (depth + 1) + (height + 1) * (depth + 1)
    
    progrWindow = mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating mesh...', isInterruptable = True, max = totalRayCasts)
    
    # Take each of the 3 imaginary planes and project rays from their "vertices" onto the base mesh
    projectPlane(planeOriginX, planeOriginY, planeEndZ,
                planeEndX, planeEndY, [0, 0, -1], subMeshNames, cellWidth,
                cellHeight, progrWindow, randRotation)
    
    projectPlane(planeOriginZ, planeOriginY, planeEndX,
                planeEndZ, planeEndY, [-1, 0, 0], subMeshNames, cellDepth,
                cellHeight, progrWindow, randRotation)
                
    projectPlane(planeOriginX, planeOriginZ, planeEndY,
                planeEndX, planeEndZ, [0, -1, 0], subMeshNames, cellWidth,
                cellDepth, progrWindow, randRotation)
    mc.progressWindow(endProgress = 1)
   
def evenPopulation(subMeshNames, numberOfInstances, randRotation):
    ''' Populate the base mesh randomly, while trying to keep the instantiated submeshes as far away from each other.
    
    - subMeshNames: list of names of the submeshes
    - numberOfInstances: how many submeshes should be instantiated
    - randRotation: whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated
    
    Best Candidate Selection algorithm idea taken from: https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/'''
    
    graph = g.getVertices()
    sampleList = []
    
    # Generate the Samples list of polygon indices
    calculateFaceAreaArray( sampleList )
    
    numberOfVerts = mc.polyEvaluate( baseMeshName, v = True )
    startingVertex = rand.randrange(0, numberOfVerts)
    
    #Pick a random starting vertex
    while mc.getAttr(baseMeshShape + '.growthWeights')[startingVertex] < 0.05:
        startingVertex = rand.randrange(0, numberOfVerts)
        
    usedVertices = []
    usedVertices.append( startingVertex )
    
    sel = om.MSelectionList()
    dag = om.MDagPath()
    sel.add(baseMeshName)
    sel.getDagPath(0, dag)
    mesh = om.MFnMesh(dag)
    
    sampleMultiplier = 0.7
    dir = om.MVector()
    
    progrWindow = mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating mesh...', isInterruptable = True, max = numberOfInstances)
    for i in range(numberOfInstances):
        
        animatedGroupName = None
        if isBaseAnimated:
            animatedGroupName = mc.group(em = True)
        
        
        # We establish a number of candidates to generate and check, which increases linearly with the amount of submeshes
        # we have placed so far
        numberOfCandidates = len(usedVertices) * sampleMultiplier + 1
        
        bestDist = 0
        bestID = -1
        bestWeight = 0
        bestPos = []
        bestFace = baseMeshName + '.f[0]'
            
        for k in range(int(numberOfCandidates)):
            randomFace = rand.choice(sampleList)
            randomFace = baseMeshName + '.f[' + str(randomFace) + ']'
            mc.select(randomFace, r = True)
            
            mc.select( mc.polyListComponentConversion(tv = True) )
            selectedVerts = mc.ls( sl = True, fl = True )
            
            # Pick a random triangle within the selected polygon
            currentTriangle = rand.sample(selectedVerts, 3)
            
            averageWeight = getAverageWeight(currentTriangle)
            while averageWeight < 0.05:
                randomFace = rand.choice(sampleList)
                randomFace = baseMeshName + '.f[' + str(randomFace) + ']'
                mc.select(randomFace, r = True)
            
                mc.select( mc.polyListComponentConversion(tv = True) )
                selectedVerts = mc.ls( sl = True, fl = True )
            
                # Pick another random triangle within the selected polygon
                currentTriangle = rand.sample(selectedVerts, 3)
            
                averageWeight = getAverageWeight(currentTriangle)
                
            #mc.select(currentTriangle, r = True)
            
            # Get random point within this face
            v1 = MVector(*mc.pointPosition(currentTriangle[0], w = True))
            v2 = MVector(*mc.pointPosition(currentTriangle[1], w = True))
            v3 = MVector(*mc.pointPosition(currentTriangle[2], w = True))
            
            a = rand.random()
            b = rand.random()
            
            if a + b >= 1:
                a = 1-a
                b = 1-b
            randomPoint = v1 + ( v2 - v1 ) * a + ( v3 - v1 ) * b
            
            #GET A RANDOM VERTEX FROM THE SELECTED TRIANGLE
            vertID = int(re.findall(r"[\w]+", currentTriangle[rand.randint(0, 2)])[2])
            
            # We compare the geodesic distance (using the Graph we generated before and using the Dijkstra pathfinding
            # algorithm) between our current vertex and all of the other vertices we have used to place submeshes
            distances = calculate_distances( graph, vertID )
            minDist = 999999.9
            for vertexID in usedVertices:
                dist = distances[vertexID]
                if dist < minDist:
                    minDist = dist
            
            # We take the smallest distance and compare it with our Best Distance (defaulted to 0 at the beginning of each
            # candidate search). If it's greater than it, then it means this is the vertex which is the furthest away from
            # any other vertex we've used, so we make it our new Best Candidate
            if minDist > bestDist:
                bestDist = minDist
                bestID = vertID
                bestWeight = averageWeight
                bestPos = randomPoint
                bestFace = randomFace
 
        # Once we've gone through all the candidates, we place and rotate the new submesh according to the
        # Best Candidate's information and add the Best Vertex to the "used vertices" list
        placeInstance(rand.choice(subMeshNames), bestPos, bestWeight, bestFace, randRotation, animatedGroupName)
        
        usedVertices.append(bestID)
        
        if mc.progressWindow (q = True, isCancelled = True):
            break
        mc.progressWindow(edit = True, step = 1)

    mc.progressWindow(endProgress = 1)
    
def prepareBaseMesh():
    ''' Initialize the base mesh with the proper Paintable Attribute on its Shape Node, as well as generate the Control Sphere and vertex graph. '''
    
    if mc.attributeQuery('growthWeights', node = baseMeshShape, exists = True) == False:
        mc.addAttr(baseMeshShape, ln = 'growthWeights', nn = 'GrowthWeights', dt = 'doubleArray')
        
    mc.makePaintable('mesh', 'growthWeights', at = 'doubleArray')
    
    # Create the control sphere and assign it a semi-transparent material
    if mc.objExists(controlSphereName) == False:
        mc.polySphere(n = controlSphereName, r = 1)
        semitransparentMaterial = mc.shadingNode('blinn', asShader = True)
        mc.setAttr(semitransparentMaterial + '.transparency', 0.3, 0.3, 0.3, type = 'double3')
        mc.select(controlSphereName)
        mc.hyperShade(a = semitransparentMaterial)
        
    mSel = om.MSelectionList()
    mSel.add(baseMeshName)
    om.MGlobal.getActiveSelectionList(mSel)
    dagPath = om.MDagPath()
    component = om.MObject()
    mSel.getDagPath(0, dagPath, component)

    # Get positions of all vertices on the mesh
    meshFn = om.MFnMesh(dagPath)
    positions = om.MPointArray()
    meshFn.getPoints(positions, om.MSpace.kWorld)

    # Iterate and calculate vectors based on connected vertices
    iter = om.MItMeshVertex(dagPath, component)
    connectedVertices = om.MIntArray()
    checkedNodes = [False] * positions.length()
    
    # Initialize the graph with the base mesh's vertices
    while not iter.isDone():
        curVertexID = iter.index()
        checkedNodes[curVertexID] = True
    
        g.addVertex(curVertexID)
        iter.getConnectedVertices(connectedVertices)
    
        for i in range(connectedVertices.length()):
            neighbourID = connectedVertices[i]
            if checkedNodes[neighbourID] == False:
                g.addVertex(neighbourID)
                g.addEdge(curVertexID, neighbourID, squareDistance( positions[curVertexID], positions[neighbourID] ))
        iter.next()
        
def populateGenerateAnimate( alignmentTypeCtrl, randomCtrl, alignedCtrl, evenCtrl, subMeshNamesCtrl, useSphereCtrl, alongSurfaceCtrl, insideSphereCtrl, startFrameCtrl,
                            endFrameCtrl, numberOfInstancesCtrl, widthCtrl, heightCtrl, depthCtrl, animatedCtrl, speedCtrl, randRotCtrl, *pArgs ):
    ''' The core function that runs the script based on the parameters received from the UI.
    
    - alignmentTypeCtrl: radio group for the distribution/population type
    - randomCtrl: radio button for random population
    - alignedCtrl: radio button for aligned population
    - evenCtrl: radio button for evenly spaced population
    - subMeshNamesCtrl: text field with the names of the submeshes
    - useSphereCtrl: radio button for Sphere Mask animation type
    - alongSurfaceCtrl: radio button for Along Surface animation type
    - insideSphereCtrl: radio button for whether the animation should be inside the sphere, in the case of Sphere Mask animation, or grow towards the sphere in the case of Along Surface animation
    - startFrameCtrl: number field for the starting frame of the animation
    - endFrameCtrl: number field for the ending frame of the animation
    - numberOfInstancesCtrl: number field for the desired number of instances, in the cases of Random or Evenly Spaced population
    - widthCtrl: number field for the desired number of subdivisions along the X axis, in the case of Aligned population
    - heightCtrl: number field for the desired number of subdivisions along the Y axis, in the case of Aligned population
    - depthCtrl: number field for the desired number of subdivisions along the Z axis, in the case of Aligned population
    - animatedCtrl: checkbox for animated base meshes
    - speedCtrl: submesh growth animation speed, in the case of Along Surface animation
    - randRotCtrl: checkbox for whether the submesh is to receive a random rotation on the local Y axis after it has been instantiated'''
    
                                
    alignmentType = mc.radioCollection(alignmentTypeCtrl, q = True, sl = True)
    
    randomCtrl = randomCtrl.split("|")
    randomCtrl = randomCtrl[ len(randomCtrl)-1 ]
    alignedCtrl = alignedCtrl.split("|")
    alignedCtrl = alignedCtrl[ len(alignedCtrl)-1 ]
    evenCtrl = evenCtrl.split("|")
    evenCtrl = evenCtrl[ len(evenCtrl)-1 ]
    speed = mc.floatField( speedCtrl, q = True, v = True )
    
    subMeshNames = re.split(' |\n', mc.scrollField(subMeshNamesCtrl, q = True, text = True))
    useSphere = mc.radioButton (useSphereCtrl, q = True, sl = True )
    alongSurface = mc.radioButton (alongSurfaceCtrl, q = True, sl = True )
    insideSphere =  mc.radioButton(insideSphereCtrl, q = True, sl = True)
    startFrame = mc.intField(startFrameCtrl, q = True, v = True)
    endFrame = mc.intField(endFrameCtrl, q = True, v = True)
    numberOfInstances = mc.intField(numberOfInstancesCtrl, q = True, v = True)
    width = mc.intField( widthCtrl, q = True, v = True )
    height = mc.intField( heightCtrl, q = True, v = True )
    depth = mc.intField( depthCtrl, q = True, v = True )
    randRotation = mc.checkBox( randRotCtrl, q = True, v = True )

    frameRange = endFrame - startFrame + 1
    
    global isBaseAnimated
    isBaseAnimated = mc.checkBox( animatedCtrl, q = True, v = True )
    if mc.objExists(groupName) == True:
        mc.delete(groupName)
    
    mc.group(em = True, name = groupName)

    # Decide which way to populate/distribute the submeshes along the base mesh
    if alignmentType == randomCtrl:
        randomPopulation(subMeshNames, numberOfInstances, randRotation)
    elif alignmentType == alignedCtrl:
        alignedPopulation(subMeshNames, width, height, depth, randRotation)
    elif alignmentType == evenCtrl:
        evenPopulation(subMeshNames, numberOfInstances, randRotation)
    
    # Decide which way to animate the submeshes along the base mesh
    submeshes = mc.listRelatives(groupName, c = True)      
    if useSphere == True:
        animateWithSphere(startFrame, frameRange, submeshes, insideSphere)
    elif alongSurface == True:
        animateAlongSurface( submeshes, insideSphere, startFrame, frameRange, speed )
    else:
        animateNoSphere(startFrame, endFrame, submeshes)

def switchRandomAlignmentSettings( numberText, numberCtrl, *pArgs ):
    '''
    Switch the visibility of the Random animation type parameters in the interface.
    '''
    visibility = mc.text( numberText, q = True, enable = True )
    
    mc.text( numberText, e = True, enable = not visibility )
    mc.intField( numberCtrl, e = True, enable = not visibility )
    
def switchAlignedAlignmentSettings( widthText, widthCtrl, heightText, heightCtrl, depthText, depthCtrl, *pArgs ):
    '''
    Switch the visibility of the Aligned animation type parameters in the interface.
    '''
    visibility = mc.text( widthText, q = True, enable = True )
    
    mc.text( widthText, e = True, enable = not visibility )
    mc.intField( widthCtrl, e = True, enable = not visibility )
    mc.text( heightText, e = True, enable = not visibility )
    mc.intField( heightCtrl, e = True, enable = not visibility )
    mc.text( depthText, e = True, enable = not visibility )
    mc.intField( depthCtrl, e = True, enable = not visibility )

def switchInfluenceTypeSettings ( textCtrl, insideCtrl, outsideCtrl, *pArgs ):
    '''
    Switch the visibility of the "Inside Sphere"/"Outside Sphere" or "Inwards Towards Sphere"/"Outwards From Sphere" radio buttons.
    '''
    visibility = mc.text( textCtrl, q = True, vis = True )
    
    mc.text( textCtrl, e = True, vis = not visibility )
    mc.radioButton( insideCtrl, e = True, vis = not visibility )
    mc.radioButton( outsideCtrl, e = True, vis = not visibility )
    
def changeInfluenceTypeLabels( insideCtrl, outsideCtrl, label1, label2, *pArgs ):
    mc.radioButton( insideCtrl, e = True, label = label1 )
    mc.radioButton( outsideCtrl, e = True, label = label2 )
    
def changeInfluenceTypeLabelsForAlongSurface( insideCtrl, outsideCtrl, label1, label2, speedText, speed, *pArgs ):
    mc.radioButton( insideCtrl, e = True, label = label1 )
    mc.radioButton( outsideCtrl, e = True, label = label2 )
    
    vis = mc.text( speedText, q = True, vis = True )
    mc.text( speedText, e = True, m = not vis )
    mc.floatField( speed, e = True, m = not vis )
      
#GUI
def createUI():
    ''' Function to generate the user interface. '''
    
    windowID = "GUI"
    if mc.window(windowID, exists = True):
        mc.deleteUI(windowID)
    
    winControl = mc.window(windowID, title = 'Growth Control', w = 100)
    
    
    mc.columnLayout(columnAttach=('both', 5), rowSpacing = 7, columnWidth = 10, adjustableColumn=True)
    
    mc.rowLayout(nc = 3, cat = [ (1, 'both', 1), (2, 'both', 10), (3, 'both', 1) ], adjustableColumn = 2)
    mc.separator()
    scriptPath = os.environ['MAYA_SCRIPT_PATH'].split(";")[0]
    mc.image( image = scriptPath + '/thumbnail.png', h = 150)
    mc.separator()
    mc.setParent('..')
    
    mc.separator()
    mc.text( label = "Submeshes Names" )
    submeshNamesCtrl = mc.scrollField(editable = True, wordWrap = True, text = "MeshName", h = 100)
    mc.separator(style = 'in')
    
    mc.text( label = "Distribution Type" )

    alignmentTypeCtrl = mc.radioCollection()
    mc.rowLayout(nc = 3, cat = [ (1, 'both', 10), (2, 'both', 10), (3, 'both', 10) ])
    randomCtrl = mc.radioButton(label = 'Random', align = 'center', sl = True, ann = 'random')
    alignedCtrl = mc.radioButton(label = 'Aligned', align = 'center')
    evenCtrl = mc.radioButton(label = 'Evenly Spaced', align = 'center')
    mc.setParent( '..' )
    mc.separator(style = 'in')
    
    mc.rowLayout(nc = 3, cat = [ (1, 'both', 1), (2, 'both', 10), (3, 'both', 1) ])
    mc.separator()
    randRotCtrl = mc.checkBox( l = 'Random Yaw Rotation', v = True)
    mc.separator()
    mc.setParent('..')
    mc.separator(style = 'in')
    
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 5), (2, 'both', 5) ], adjustableColumn = 2)
    numberText = mc.text( "Number of instances" )
    numberCtrl = mc.intField(v = 100, w = 1)
    mc.setParent( '..' )
    mc.separator(style = 'in')
    
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 5), (2, 'both', 5) ], adjustableColumn = 2)
    widthText = mc.text( "Instances along X-axis", enable = False )
    widthCtrl = mc.intField(v = 20, w = 1, min = 1, max = 100, step = 1, enable = False)
    mc.setParent( '..' )
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 5), (2, 'both', 5) ], adjustableColumn = 2)
    heightText = mc.text( "Instances along Y-axis", enable = False )
    heightCtrl = mc.intField(v = 20, w = 1, min = 1, max = 100, step = 1, enable = False)
    mc.setParent( '..' )
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 5), (2, 'both', 5) ], adjustableColumn = 2)
    depthText = mc.text( "Instances along Z-axis", enable = False )
    depthCtrl = mc.intField(v = 20, w = 1, min = 1, max = 100, step = 1, enable = False)
    mc.setParent( '..' )
    mc.separator(style = 'in')
    
    mc.rowLayout(nc = 3, cat = [ (1, 'both', 1), (2, 'both', 10), (3, 'both', 1) ], adjustableColumn = 2)
    mc.separator()
    animatedCtrl = mc.checkBox( l = 'Animated Base Mesh', v = False)
    mc.separator()
    mc.setParent('..')
    mc.separator(style = 'in')
    
    mc.text( label = "Growth Animation Type" )
    
    mc.radioCollection()
    mc.rowLayout(nc = 3, cat = [ (1, 'both', 10), (2, 'both', 10), (3, 'both', 10) ] ,adjustableColumn = 1)
    uniformCtrl = mc.radioButton(label = 'Uniform', align = 'center')
    useSphereCtrl = mc.radioButton(label = 'Sphere Mask', align = 'center', sl = True)
    alongSurfaceCtrl = mc.radioButton(label = 'Along surface', align = 'center')
    mc.setParent( '..' )
    
    influenceTextCtrl = mc.text( label = 'Growth Influence Type' )
    mc.radioCollection()
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 10), (2, 'left', 10) ])
    insideCtrl = mc.radioButton(label = 'Inside Sphere', align = 'center', sl = True)
    outsideCtrl = mc.radioButton(label = 'Outside Sphere', align = 'center')
    mc.setParent( '..' )
    
    mc.rowLayout(nc = 2, cat = [ (1, 'left', 5), (2, 'both', 5) ], adjustableColumn = 2)
    speedTextCtrl = mc.text( label = 'Submesh growth speed', vis = False )
    speedCtrl = mc.floatField(v = 4, vis = False)
    mc.setParent( '..' )
    mc.separator(style = 'in')
    
    mc.rowLayout(nc = 5, adjustableColumn5 = 3, cat = [ (1, 'left', 1), (2, 'left', 2), (3, 'both', 5), (4, 'right', 2), (5, 'right', 1) ])
    mc.text( label = "Start Frame" )
    startFrameCtrl = mc.intField(w = 60, v = 1)
    mc.separator(w = 5)
    mc.text ( label = "End Frame" )
    endFrameCtrl = mc.intField(w = 60, v = 50)
    mc.setParent( '..' )
    mc.separator(style = 'in')
    
    mc.radioButton( randomCtrl, e = True, cc = ft.partial( switchRandomAlignmentSettings, numberText, numberCtrl ) )
    mc.radioButton( alignedCtrl, e = True, cc = ft.partial( switchAlignedAlignmentSettings, widthText, widthCtrl, heightText, heightCtrl, depthText, depthCtrl ) )
    mc.radioButton( evenCtrl, e = True, cc = ft.partial( switchRandomAlignmentSettings, numberText, numberCtrl ) )
    mc.radioButton( uniformCtrl, e = True, cc = ft.partial( switchInfluenceTypeSettings, influenceTextCtrl, insideCtrl, outsideCtrl) )
    mc.radioButton( useSphereCtrl, e = True, cc = ft.partial( changeInfluenceTypeLabels, insideCtrl, outsideCtrl, 'Inside Sphere', 'Outside Sphere'))
    mc.radioButton( alongSurfaceCtrl, e = True, cc = ft.partial( changeInfluenceTypeLabelsForAlongSurface, insideCtrl, outsideCtrl, 'Inwards Towards Sphere', 'Outwards From Sphere', speedTextCtrl, speedCtrl ) )
    
    mc.button( label = "Generate, populate and animate!",
        command = ft.partial( populateGenerateAnimate, alignmentTypeCtrl, randomCtrl, alignedCtrl, evenCtrl, submeshNamesCtrl, useSphereCtrl, alongSurfaceCtrl, insideCtrl,
        startFrameCtrl, endFrameCtrl, numberCtrl, widthCtrl, heightCtrl, depthCtrl, animatedCtrl, speedCtrl, randRotCtrl))

    mc.separator(style = 'in')
    
    mc.showWindow(winControl)
    
    
if __name__ == "__main__":
    
    # Store the base mesh name, base mesh shape, the group under which the submeshes will be parented under, control sphere
    # name, vertices graph and whether the base mesh is animated as global variables
    baseMeshName = mc.ls(sl = True)
    baseMeshName = baseMeshName[0]
    baseMeshShape = mc.listRelatives( shapes = True )[0]
    
    groupName = '_PopulationGroup'
    
    controlSphereName = '_ControlSphere'
    
    isBaseAnimated = False
    
    g = Graph( mc.polyEvaluate( v = True ) )
    
    # Create the UI and initialize the base mesh
    createUI()
    prepareBaseMesh()
        


    
   
    
    
    
    
    
    
    
    
    
    