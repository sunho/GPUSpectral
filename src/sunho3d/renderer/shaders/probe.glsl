// probe id
// probe.id / grid.x
// id = x + y * max_x + z * max_x*max_y
// texstart = (x * tile_size, y * tile + z * max_y*tile)
// pos = (x - max_x/2)

uvec3 probIDToGrid(uint id, uvec3 gridNum) {
    uint z = id / (gridNum.x * gridNum.y);
    uint y = (id % (gridNum.x * gridNum.y)) / gridNum.x;
    uint x = id % gridNum.x;
    return uvec3(x,y,z);
}

ivec3 posToGrid(vec3 pos, uvec3 gridNum, vec3 sceneSize) {
    vec3 gridSize = sceneSize * 2.0 / gridNum;
    ivec3 grid = ivec3((pos + sceneSize) / gridSize);
    return grid;
}

int gridToProbeID(ivec3 grid, uvec3 gridNum) {
    return grid.x + grid.y * int(gridNum.x) + grid.z * int(gridNum.y * gridNum.x);
}

vec3 probeIDToPos(uint id, uvec3 gridNum, vec3 sceneSize) {
    uvec3 grid = probIDToGrid(id, gridNum);
    vec3 gridSize = sceneSize * 2.0 / gridNum;
    vec3 ogd = vec3(grid) * gridSize;
    return ogd - sceneSize;
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