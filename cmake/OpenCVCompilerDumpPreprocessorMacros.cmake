function(ocv_dump_gcc_preprocessor fname exe extra_flags)
  set(src "${CMAKE_CURRENT_BINARY_DIR}/preprocessor/empty")
  file(WRITE "${src}" "")
  string(REPLACE " " ";" opt "${extra_flags}")
  execute_process(COMMAND "${exe}" ${opt} -dM -E -
    RESULT_VARIABLE res
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    INPUT_FILE "${src}"
  )
  set(dump_file "${CMAKE_CURRENT_BINARY_DIR}/preprocessor/${fname}")
  string(REPLACE "\n" ";" out "${out}")
  list(SORT out)
  string(REPLACE ";" "\n" out "${out}")
  set(out "CMD => ${res}\n${exe} ${extra_flags}\n\nSTDERR\n${err}\n===\n${out}")
  ocv_update_file("${dump_file}" "${out}")
  if(res)
    message(WARNING "ocv_dump_gcc_preprocessor has failed, check output file for details: ${dump_file}")
  endif()
endfunction()

# --- Entry here ---
if(CV_GCC OR CV_CLANG)
  foreach(LANG CXX)
    foreach(CFG ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${CFG}" CFG)
      ocv_dump_gcc_preprocessor("dump_${LANG}_${CFG}_baseline.txt"
        "${CMAKE_${LANG}_COMPILER}"
        "${CMAKE_${LANG}_FLAGS} ${CMAKE_${LANG}_FLAGS_${CFG}}")
      foreach(OPT ${CPU_DISPATCH_FINAL})
        ocv_dump_gcc_preprocessor("dump_${LANG}_${CFG}_${OPT}.txt"
          "${CMAKE_${LANG}_COMPILER}"
          "${CMAKE_${LANG}_FLAGS} ${CMAKE_${LANG}_FLAGS_${CFG}} ${CPU_DISPATCH_FLAGS_${OPT}}")
      endforeach() # OPT
    endforeach() # CFG
  endforeach() # LANG
endif()
