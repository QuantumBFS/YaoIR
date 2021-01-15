configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(YAO_OPT_TEST_DEPENDS
        FileCheck count not
        yao-opt
        )

add_lit_testsuite(check-yao-opt "Running the yao-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${YAO_OPT_TEST_DEPENDS}
        )
set_target_properties(check-yao-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(YAO_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${YAO_OPT_TEST_DEPENDS})
