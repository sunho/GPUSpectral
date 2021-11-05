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

vec3 probeIDToPos(uint id, uvec3 gridNum, vec3 sceneSize) {
    uvec3 grid = probIDToGrid(id, gridNum);
    vec3 gridSize = sceneSize / gridNum;
    vec3 ogd = vec3(grid) - vec3(gridNum)/2.0;
    return ogd * gridSize;
}