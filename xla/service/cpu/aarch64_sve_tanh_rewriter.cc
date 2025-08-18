#include "xla/service/cpu/aarch64_sve_tanh_rewriter.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/shape_util.h"

namespace xla {
namespace cpu {

absl::StatusOr<bool> Aarch64SveTanhRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kTanh &&
          instruction->shape().element_type() == F32) {
        
        const Shape& shape = instruction->shape();
        int64_t element_count = ShapeUtil::ElementsIn(shape);
        
        // Replace tanh with custom call for arrays >= 16 elements
        if (element_count >= 16) {
          HloInstruction* custom_call = computation->AddInstruction(
              HloInstruction::CreateCustomCall(
                  shape,
                  {instruction->mutable_operand(0)},
                  cpu::runtime::kAarch64SveHyperbolicTangentSymbolName));
          
          TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(custom_call));
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
          changed = true;
        }
      }
    }
  }
  
  return changed;
}

}  // namespace cpu
}  // namespace xla
