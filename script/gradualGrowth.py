import maya.cmds as mc
import random as rand
import math as m
import functools as ft
from maya.OpenMaya import MVector
import maya.OpenMaya as om
import re
import heapq
import os

def calculate_distances(graph, starting_vertex):
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
    def __init__(self, size):
        self.vertexArray = [None] * size
        self.numVertices = 0
        
    #def __iter__(self):
        #return iter(self.vertexArray)
        
    def addVertex(self, node):
        if self.vertexArray[node] == None:
            self.numVertices += 1
            self.vertexArray[node] = []
        
    def addEdge(self, start, end, length):
        self.vertexArray[start].append( (end, length) )
        self.vertexArray[end].append( (start, length) )
        
    def printGraph(self):
        for i in range(self.numVertices):
            print i, self.vertexArray[i]
        
    def getVertices(self):
        return self.vertexArray        
    

def distance( point1, point2 ):
    return m.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
    
def squareDistance( point1, point2 ):
    return (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2

def float_range(start, end, step):
    while start <= end:
        yield start
        start += step

def returnCollisionPoints(point, dir, fnMesh):
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
    normal = mc.polyInfo(face_, faceNormals = True)
    
    # convert the string to array with regular expression
    normal = re.findall(r"[\w.-]+", normal[0]) #normal[0].split(' ')
    
    normal = MVector( float(normal[2]),\
                      float(normal[3]),\
                      float(normal[4]) )
    return normal

def getIndexOfTuple(l, index, value):
    for pos,t in enumerate(l):
        if t[index] == value:
            return pos

    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")

def getClosestVertex( objectPos, baseMesh ):
    object = om.MPoint( objectPos[0], objectPos[1], objectPos[2] )

    startingPoint = om.MPoint()
    faceIdxUtil = om.MScriptUtil()
    faceIdxUtil.createFromInt(-1)
    faceIntPtr = faceIdxUtil.asIntPtr()

    baseMesh.getClosestPoint(object, startingPoint, om.MSpace.kWorld, faceIntPtr)
    faceIdx = faceIdxUtil.getInt(faceIntPtr)
    
    faceVertices = om.MIntArray()
    
    baseMesh.getPolygonVertices( faceIdx, faceVertices )
                    
    #setting vertex [0] as the closest one
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

def populateAlongSurface( objects, inwards, startFrame, frameRange, growthSpeed):
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
    
    startPoint = getClosestVertex( spherePos, fnMesh )

    graphDistances = calculate_distances( graph, startPoint )
    objectDistances = []
    maxDist = -1
    for i in range ( len(graphDistances) ):
        if graphDistances[i] > maxDist:
            maxDist = graphDistances[i]
    
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

def populateWithSphere(startFrame, frameRange, objects, insideSphere):
    
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
            sphereInfluence = m.sqrt(sphereInfluence)
            mc.select(objects[z])
            mc.setKeyframe( v = objSizes[z][0] * sphereInfluence, at = 'scaleX')
            mc.setKeyframe( v = objSizes[z][1] * sphereInfluence , at = 'scaleY')
            mc.setKeyframe( v = objSizes[z][2] * sphereInfluence , at = 'scaleZ')
            if mc.progressWindow (q = True, isCancelled = True):
                break
        mc.progressWindow(edit = True, step = 1)
    mc.progressWindow(endProgress = 1)

def populateNoSphere(startFrame, endFrame, objects):
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
    index1 = int(re.findall(r"[\w]+", vertices[0])[2])
    index2 = int(re.findall(r"[\w]+", vertices[1])[2])
    index3 = int(re.findall(r"[\w]+", vertices[2])[2])
    
    weights = mc.getAttr(baseMeshShape + '.growthWeights')
    return (weights[index1] + weights[index2] + weights[index3]) / 3.0

def placeInstance(instanceName, position, size, face, randRotation, animatedParentGroup = None):
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
        #print animatedParentGroup
        mc.pointOnPolyConstraint( face, animatedParentGroup, mo = False )
        mc.parent( objectInstance, animatedParentGroup )
        mc.parent( animatedParentGroup, groupName )
    else:
        mc.parent ( objectInstance, groupName )

def calculateAndInstantiate(subMeshName, hitPoint, hitFace, direction, randRotation, animatedParentGroup = None):
    selectedVerts = getTriangleFromHitpoint(hitPoint, hitFace)
                        
    averageWeight = getAverageWeight(selectedVerts)
    face = baseMeshName + '.f[' + str(hitFace) + ']'
    faceNormal = getNormal(face)
                        
    dotProduct = faceNormal[0] * direction[0] + faceNormal[1] * direction[1] + faceNormal[2] * direction[2]
                        
    if averageWeight > 0.05 and abs(dotProduct) > 0.6:
        placeInstance(subMeshName, hitPoint, averageWeight, face, randRotation, animatedParentGroup )
        
def calculateFaceAreaArray( sampleArray ):
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
    numberOfFaces = mc.polyEvaluate( baseMeshName, f = True )
    vertList = mc.ls(baseMeshName + '.vtx[*]', fl = True)
    sampleArray = []
    calculateFaceAreaArray( sampleArray )
    
    mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating...', isInterruptable = True, max = numberOfInstances)
    for i in range(numberOfInstances):
        groupy  = None
        if isBaseAnimated:
            groupy = mc.group(em = True)
        
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
        v1 = MVector(*mc.pointPosition(currentTriangle[0]))
        v2 = MVector(*mc.pointPosition(currentTriangle[1]))
        v3 = MVector(*mc.pointPosition(currentTriangle[2]))
        
        a = rand.random()
        b = rand.random()
        
        if a + b >= 1:
            a = 1-a
            b = 1-b
            
        randomPoint = v1 + ( v2 - v1) * a + ( v3 - v1 ) * b
        print randomPoint[0], randomPoint[1], randomPoint[2]
        
        #objectInstance = mc.instance(rand.choice(subMeshNames))
        placeInstance(rand.choice(subMeshNames), randomPoint, averageWeight, randomFace, randomRotation, groupy)
        
        if mc.progressWindow (q = True, isCancelled = True):
            break
        mc.progressWindow(edit = True, step = 1)

    mc.progressWindow(endProgress = 1)


def getTriangleFromHitpoint(hitPoint, faceID):
    mc.select(baseMeshName + '.f[' + str(faceID) + ']')
    mc.select( mc.polyListComponentConversion(tv = True) )
    selectedVerts = mc.ls( sl = True, fl = True )
    
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

def projectPlane(originX, originY, originDepth, width, height, dir, subMeshNames, cellWidth, cellHeight, onlyOuterProjection, pWindow, randRotation):
    point = [0, 0, 0]
    dirVector = om.MFloatVector(*dir)
    
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
                
            #if onlyOuterProjection == True:
                #if hitPoints[0]:
                    #calculateAndInstantiate(subMeshName, hitPoints[0], hitFaces[0], dir)
                            
                #if hitPoints.length() > 1:
                    #calculateAndInstantiate(subMeshName, hitPoints[hitPoints.length()-1],
                        #hitFaces[hitPoints.length()-1], dir)
            #else:
            for z in range(hitPoints.length()):
                if isBaseAnimated:
                    calculateAndInstantiate(rand.choice(subMeshNames), hitPoints[z], hitFaces[z], dir, randRotation, mc.group(em = True))
                else:
                    calculateAndInstantiate(rand.choice(subMeshNames), hitPoints[z], hitFaces[z], dir, randRotation, None, randRotation)
            if mc.progressWindow (pWindow, q = True, isCancelled = True):
                break

            mc.progressWindow(pWindow, edit = True, status= ('Populating mesh...'), step = 1)

def alignedPopulation(subMeshNames, width, height, depth, randRotation):
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
   
    cellWidth = planeWidth / width
    cellHeight = planeHeight / height
    cellDepth = planeDepth / depth
    
    totalRayCasts = (width + 1) * (height + 1) + (width + 1) * (depth + 1) + (height + 1) * (depth + 1)
    
    progrWindow = mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating mesh...', isInterruptable = True, max = totalRayCasts)
    
    projectPlane(planeOriginX, planeOriginY, planeEndZ,
                planeEndX, planeEndY, [0, 0, -1], subMeshNames, cellWidth,
                cellHeight, False, progrWindow, randRotation)
    
    projectPlane(planeOriginZ, planeOriginY, planeEndX,
                planeEndZ, planeEndY, [-1, 0, 0], subMeshNames, cellDepth,
                cellHeight, False, progrWindow, randRotation)
                
    projectPlane(planeOriginX, planeOriginZ, planeEndY,
                planeEndX, planeEndZ, [0, -1, 0], subMeshNames, cellWidth,
                cellDepth, False, progrWindow, randRotation)
    mc.progressWindow(endProgress = 1)
   
def evenPopulation(subMeshNames, numberOfInstances, randRotation):
    graph = g.getVertices()
    sampleList = []
    calculateFaceAreaArray( sampleList )
    
    numberOfVerts = mc.polyEvaluate( baseMeshName, v = True )
    startingVertex = rand.randrange(0, numberOfVerts)
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
    numberOfFaces = mc.polyEvaluate( baseMeshName, f = True )
    vertList = mc.ls(baseMeshName + '.vtx[*]', fl = True)
    
    progrWindow = mc.progressWindow(title = 'Progress', progress = 0, status = 'Populating mesh...', isInterruptable = True, max = numberOfInstances)
    for i in range(numberOfInstances - 1):
        numberOfCandidates = len(usedVertices) * sampleMultiplier + 1
        
        bestDist = 0
        bestID = -1
        bestWeight = 0
        bestPos = []
        bestFace = None
        
        animatedGroup = None
        if isBaseAnimated:
            animatedGroup = mc.group(em = True)
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
                
            mc.select(currentTriangle, r = True)
            
            # Get random point within this face
            # https://math.stackexchange.com/questions/538458/triangle-point-picking-in-3d
            v1 = MVector(*mc.pointPosition(currentTriangle[0]))
            v2 = MVector(*mc.pointPosition(currentTriangle[1]))
            v3 = MVector(*mc.pointPosition(currentTriangle[2]))
            
            a = rand.random()
            b = rand.random()
            
            if a + b >= 1:
                a = 1-a
                b = 1-b
            randomPoint = v1 + ( v2 - v1 ) * a + ( v3 - v1 ) * b
            
            distToClosestVertID = 9999999
            #closestVertID = -1
            
            # GET THE CLOSEST VERTEX IN THE TRIANGLE
            #for j in range(3):
                #d = squareDistance( mc.pointPosition(currentTriangle[j]), randomPoint )
                #if d < distToClosestVertID:
                    #distToClosestVertID = d
                    #closestVertID = currentTriangle[j]
             
            # GET A RANDOM VERTEX        
            #randIndex = rand.randrange(0, numberOfVerts)
            #weight = getWeightAtVertexIndex(randIndex)
            #while weight == 0:
                #randIndex = rand.randrange(0, numberOfVerts)
                #weight = getWeightAtVertexIndex(randIndex)
            
            #GET A RANDOM VERTEX FROM THE SELECTED TRIANGLE
            vertID = int(re.findall(r"[\w]+", currentTriangle[rand.randint(0, 2)])[2])
            distances = calculate_distances( graph, vertID )
            minDist = 999999.9
            for vertexID in usedVertices:
                dist = distances[vertexID]
                if dist < minDist:
                    minDist = dist
                
            if minDist > bestDist:
                bestDist = minDist
                bestID = vertID
                bestWeight = averageWeight
                bestPos = randomPoint
                bestFace = randomFace
        
        usedVertices.append(bestID)
            
        #mesh.getVertexNormal(bestID, False, dir, om.MSpace.kWorld)
        #dir = getNormal(bestFace)
        placeInstance(rand.choice(subMeshNames), bestPos, bestWeight, bestFace, randRotation, animatedGroup)
        if mc.progressWindow (q = True, isCancelled = True):
            break
        mc.progressWindow(edit = True, step = 1)

    mc.progressWindow(endProgress = 1)
    
def prepareBaseMesh():
    if mc.attributeQuery('growthWeights', node = baseMeshShape, exists = True) == False:
        mc.addAttr(baseMeshShape, ln = 'growthWeights', nn = 'GrowthWeights', dt = 'doubleArray')
        
        #mc.deleteAttr(baseMeshShape + '.growthWeights')
        mc.makePaintable('mesh', 'growthWeights', at = 'doubleArray')
    
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

    if alignmentType == randomCtrl:
        randomPopulation(subMeshNames, numberOfInstances, randRotation)#calculateFaceAreaArray()
    elif alignmentType == alignedCtrl:
        alignedPopulation(subMeshNames, width, height, depth, randRotation)
    elif alignmentType == evenCtrl:
        evenPopulation(subMeshNames, numberOfInstances, randRotation)
        
    submeshes = mc.listRelatives(groupName, c = True)      
    #if useSphere == True:
        #populateWithSphere(startFrame, frameRange, submeshes, insideSphere)
    #elif alongSurface == True:
        #populateAlongSurface( submeshes, insideSphere, startFrame, frameRange, speed )
    #else:
        #populateNoSphere(startFrame, endFrame, submeshes)

def switchRandomAlignmentSettings( numberText, numberCtrl, *pArgs ):
    visibility = mc.text( numberText, q = True, enable = True )
    
    mc.text( numberText, e = True, enable = not visibility )
    mc.intField( numberCtrl, e = True, enable = not visibility )
    
def switchAlignedAlignmentSettings( widthText, widthCtrl, heightText, heightCtrl, depthText, depthCtrl, *pArgs ):
    visibility = mc.text( widthText, q = True, enable = True )
    
    mc.text( widthText, e = True, enable = not visibility )
    mc.intField( widthCtrl, e = True, enable = not visibility )
    mc.text( heightText, e = True, enable = not visibility )
    mc.intField( heightCtrl, e = True, enable = not visibility )
    mc.text( depthText, e = True, enable = not visibility )
    mc.intField( depthCtrl, e = True, enable = not visibility )

def switchInfluenceTypeSettings ( textCtrl, insideCtrl, outsideCtrl, *pArgs ):
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
    baseMeshName = mc.ls(sl = True)
    baseMeshName = baseMeshName[0]
    baseMeshShape = mc.listRelatives( shapes = True )[0]
    #print baseMeshName
    #print baseMeshShape
    
    groupName = '_PopulationGroup'
    
    controlSphereName = '_ControlSphere'
    
    isBaseAnimated = False
    
    g = Graph( mc.polyEvaluate( v = True ) )
    createUI()
    prepareBaseMesh()
        


    
   
    
    
    
    
    
    
    
    
    
    