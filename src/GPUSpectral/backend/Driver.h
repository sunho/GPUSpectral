#pragma once

#include "DriverBase.h"
#include "DriverTypes.h"
#include "Program.h"
#include "PipelineState.h"

class HwDriver {
public:
#define DECL_COMMAND(R, N, ARGS, PARAMS) virtual R N(ARGS) = 0;
#define DECL_VOIDCOMMAND(N, ARGS, PARAMS) virtual void N(ARGS) = 0;

#include "Command.inc"

#undef DECL_VOIDCOMMAND
#undef DECL_COMMAND
};
