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

uvec3 posToGrid(vec3 pos, uvec3 gridNum, vec3 sceneSize) {
    vec3 gridSize = sceneSize * 2.0 / gridNum;
    uvec3 grid = uvec3((pos + sceneSize) / gridSize);
    return grid;
}

uint gridToProbeID(uvec3 grid, uvec3 gridNum) {
    return grid.x + grid.y * gridNum.x + grid.z * gridNum.y * gridNum.z;
}

vec3 probeIDToPos(uint id, uvec3 gridNum, vec3 sceneSize) {
    uvec3 grid = probIDToGrid(id, gridNum);
    vec3 gridSize = sceneSize * 2.0 / gridNum;
    vec3 ogd = vec3(grid) * gridSize;
    return ogd - sceneSize;
}

vec3 octahedronReverseMap(vec2 uv) {
    vec3 position = vec3(2.0f * (uv - 0.5f), 0);                
    vec2 absolute = abs(position.xy);
    // this is "true" z because
    // when it's under the bottom and needs to be flipped to
    // the other triangle by (1-x) (1-y)
    // 1 - (1-x) - (1-y)
    // x + y - 1
    // = -(1-x-y)
    position.z = 1.0f - absolute.x - absolute.y;
    
    // when it's bottom flip to the other triangle
    if(position.z < 0) {
        position.xy = sign(position.xy) 
                    * vec2(1.0 - absolute.y, 1.0 - absolute.x);
    }

    return position;
}