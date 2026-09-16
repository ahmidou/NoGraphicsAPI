add_executable(example_clear_screen clear_screen.mm)
target_link_libraries(example_clear_screen PRIVATE
    NoGraphicsAPI::NoGraphicsAPI
    "-framework Cocoa"
    "-framework QuartzCore"
)
set_source_files_properties(clear_screen.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
set_target_properties(example_clear_screen PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.nogfx.clearscreen"
)
