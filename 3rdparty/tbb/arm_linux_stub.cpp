#include "tbb/tbb_misc.h"

namespace tbb {
namespace internal {

void affinity_helper::protect_affinity_mask(bool) {}
affinity_helper::~affinity_helper() {}
void destroy_process_mask() {}

}
}
