FILE(REMOVE_RECURSE
  "CMakeFiles/package_source"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/package_source.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
