FILE(REMOVE_RECURSE
  "CMakeFiles/dep"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/dep.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
