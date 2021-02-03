macro(ocv_add_core_parallel_backend backend_id cond_var)
  if(${cond_var})
    include("${CMAKE_CURRENT_LIST_DIR}/detect_${backend_id}.cmake")
  endif()
endmacro()

ocv_add_core_parallel_backend("tbb" WITH_TBB)
ocv_add_core_parallel_backend("openmp" WITH_OPENMP)
