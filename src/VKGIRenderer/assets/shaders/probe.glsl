
// probe id
// probe.id / grid.x
// id = x + y * max_x + z * max_x*max_y
// texstart = (x * tile_size, y * tile + z * max_y*tile)
// pos = (x - max_x/2)
ivec3 probIDToGrid(int id, SceneInfo sceneInfo) {
    ivec3 gridNum = ivec3(sceneInfo.gridNum);
    int x = id % gridNum.x;
    int y = (id % (gridNum.x * gridNum.y)) / gridNum.x;
    int z = id / (gridNum.x * gridNum.y);
    return ivec3(x,y,z);
}

ivec2 probeIDToIRDTexOffset(int id) {
    ivec2 startOffset = ivec2((id % IRD_MAP_PROBE_COLS) * IRD_MAP_TEX_SIZE, (id / IRD_MAP_PROBE_COLS) * IRD_MAP_TEX_SIZE);
    return startOffset;
}

ivec2 getIRDTexOffset(int probeId, vec2 uv) {
    ivec2 startOffset = probeIDToIRDTexOffset(probeId);
    ivec2 texOffset = ivec2(startOffset.x + IRD_MAP_BORDER + uv.x * IRD_MAP_SIZE, startOffset.y + IRD_MAP_BORDER + uv.y * IRD_MAP_SIZE);
    return texOffset;
}

ivec2 probeIDToDepthTexOffset(int id) {
    ivec2 startOffset = ivec2((id % DEPTH_MAP_PROBE_COLS) * DEPTH_MAP_TEX_SIZE, (id / DEPTH_MAP_PROBE_COLS) * DEPTH_MAP_TEX_SIZE);
    return startOffset;
}

ivec2 getDepthTexOffset(int probeId, vec2 uv) {
    ivec2 startOffset = probeIDToDepthTexOffset(probeId);
    ivec2 texOffset = ivec2(startOffset.x + DEPTH_MAP_BORDER + uv.x * DEPTH_MAP_SIZE, startOffset.y + DEPTH_MAP_BORDER + uv.y * DEPTH_MAP_SIZE);
    return texOffset;
}

int gridToProbeID(ivec3 grid, SceneInfo sceneInfo) {
    return grid.x + grid.y * int(sceneInfo.gridNum.x) + grid.z * int(sceneInfo.gridNum.y * sceneInfo.gridNum.x);
}


vec3 gridToPos(ivec3 grid, SceneInfo sceneInfo) {
    vec3 gridSize = sceneInfo.sceneSize * 2.0 / vec3(sceneInfo.gridNum);
    vec3 ogd = vec3(grid) * gridSize;
    return ogd - sceneInfo.sceneSize + sceneInfo.sceneCenter;
}

vec3 probeIDToPos(int id, SceneInfo sceneInfo) {
    ivec3 grid = probIDToGrid(id, sceneInfo);
    return gridToPos(grid, sceneInfo);
}

ivec3 posToGrid(vec3 pos, SceneInfo sceneInfo) {
    vec3 gridSize = sceneInfo.sceneSize * 2.0 / vec3(sceneInfo.gridNum);
    ivec3 grid = ivec3((pos + sceneInfo.sceneSize - sceneInfo.sceneCenter) / gridSize);
    return grid;
}

vec3 octahedronReverseMap(vec2 uv) {
    vec3 position = vec3(2.0 * (uv - 0.5), 0);
    position = vec3(position.x, 0, position.y);               
    vec2 absolute = abs(position.xz);
    // this is "true" z because
    // when it's under the bottom and needs to be flipped to
    // the other triangle by (1-x) (1-y)
    // 1 - (1-x) - (1-y)
    // x + y - 1
    // = -(1-x-y)
    position.y = 1.0 - absolute.x - absolute.y;
    
    // when it's bottom flip to the other triangle
    if(position.y < 0) {
        position.xz = sign(position.xz) 
                    * vec2(1.0 - absolute.y, 1.0 - absolute.x);
    }

    return position;
}

vec2 octahedronMap(vec3 direction) {        
    vec3 octant = sign(direction);

    // Scale the vector so |x| + |y| + |z| = 1 (surface of octahedron).
    float sum = dot(direction, octant);        
    vec3 octahedron = direction / sum;    

    // "Untuck" the corners using the same reflection across the diagonal as before.
    // (A reflection is its own inverse transformation).
    if(octahedron.y < 0) {
        vec3 absolute = abs(octahedron);
        octahedron.xz = octant.xz
                      * vec2(1.0 - absolute.z, 1.0 - absolute.x);
    }

    return octahedron.xz * 0.5 + 0.5;
}
