#define SHARC_QUERY 1
#define SHARC_UPDATE 0
#include "pathtracer.slang"

// the Render/Query shader that "traces rays while querying the cache
// on hits to accelerate rendering through early termination"

// TOOD : binding stuff

// TODO : seperate raygen thing?? or just if checks in pathtrace.slang
// [shader("raygeneration")]
// void query() {

// }