ifelse(TARGET_LANGUAGE, py, `define(ONLY_PYTHON, $1)', `define(ONLY_PYTHON, )')
ifelse(TARGET_LANGUAGE, c, `define(ONLY_C, $1)', `define(ONLY_C, )')
define(`RQ',`changequote(<,>)dnl`
'changequote`'')
