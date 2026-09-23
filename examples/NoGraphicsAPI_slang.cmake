find_program(NOGRAPHICSAPI_SLANGC NAMES slangc REQUIRED)
if(NOT APPLE)
    find_program(NOGRAPHICSAPI_SPIRV_VAL NAMES spirv-val REQUIRED)
endif()

function(NoGraphicsAPI_require_tool_version program argument name minimum pattern)
    execute_process(
        COMMAND ${program} ${argument}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(STRIP "${output}${error}" version_text)
    string(REGEX MATCH "${pattern}" match "${version_text}")
    set(version "${CMAKE_MATCH_1}")
    if(NOT result EQUAL 0 OR
       NOT match OR
       version VERSION_LESS minimum)
        message(FATAL_ERROR
            "NoGraphicsAPI examples require ${name} ${minimum} or newer; "
            "found '${version_text}' at ${program}")
    endif()
endfunction()

NoGraphicsAPI_require_tool_version(
    "${NOGRAPHICSAPI_SLANGC}" -version Slang 2026.13.1
    "([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
if(NOT APPLE)
    NoGraphicsAPI_require_tool_version(
        "${NOGRAPHICSAPI_SPIRV_VAL}" --version SPIRV-Tools 2026.3
        "SPIRV-Tools v([0-9]+\\.[0-9]+)")
endif()

function(NoGraphicsAPI_compile_slang output source entry stage)
    cmake_parse_arguments(SLANG "" "" "DEPENDS" ${ARGN})
    set(options)
    if(stage STREQUAL "mesh" OR stage STREQUAL "amplification")
        list(APPEND options -capability spvMeshShadingEXT)
    endif()

    get_filename_component(output_dir "${output}" DIRECTORY)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND ${NOGRAPHICSAPI_SLANGC}
            ${source}
            -target spirv
            -profile spirv_1_5
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            -fvk-use-c-layout
            -matrix-layout-row-major
            -capability spvDescriptorHeapEXT
            -I ${CMAKE_CURRENT_SOURCE_DIR}
            -I ${PROJECT_SOURCE_DIR}/include
            -I ${PROJECT_SOURCE_DIR}/utility/include
            ${options}
            -entry ${entry}
            -stage ${stage}
            -o ${output}
        COMMAND ${NOGRAPHICSAPI_SPIRV_VAL}
            --target-env vulkan1.4 --scalar-block-layout ${output}
        DEPENDS ${source} ${SLANG_DEPENDS}
            ${PROJECT_SOURCE_DIR}/include/NoGraphicsAPI/types.h
            ${PROJECT_SOURCE_DIR}/utility/include/NoGraphicsAPIUtility/shader_types.h
        VERBATIM
        COMMENT "Compiling Slang ${stage} shader ${entry}"
    )
endfunction()

# Metal: one metallib per source holding all its entry points. metallib_4_0 keeps [numthreads] in the MSL
# ([[required_threads_per_threadgroup]]), which the Metal backend reads back from the compute pipeline.
function(NoGraphicsAPI_compile_slang_metallib output source)
    cmake_parse_arguments(SLANG "" "" "DEPENDS" ${ARGN})
    get_filename_component(output_dir "${output}" DIRECTORY)
    get_filename_component(output_name "${output}" NAME_WE)
    set(msl ${output_dir}/${output_name}.metal)
    set(air ${output_dir}/${output_name}.air)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND ${NOGRAPHICSAPI_SLANGC}
            ${source}
            -target metal
            -capability metallib_4_0
            -matrix-layout-row-major
            -I ${CMAKE_CURRENT_SOURCE_DIR}
            -I ${PROJECT_SOURCE_DIR}/include
            -I ${PROJECT_SOURCE_DIR}/utility/include
            -o ${msl}
        COMMAND xcrun metal -std=metal4.0 -c ${msl} -o ${air}
        COMMAND xcrun metallib ${air} -o ${output}
        DEPENDS ${source} ${SLANG_DEPENDS}
            ${PROJECT_SOURCE_DIR}/include/NoGraphicsAPI/types.h
            ${PROJECT_SOURCE_DIR}/utility/include/NoGraphicsAPIUtility/shader_types.h
        VERBATIM
        COMMENT "Compiling Slang ${output_name}.metallib"
    )
endfunction()

function(NoGraphicsAPI_add_example target)
    add_executable(${target} ${ARGN})
    foreach(source IN LISTS ARGN)
        if(source MATCHES "\\.slang$")
            set_source_files_properties(${source} PROPERTIES HEADER_FILE_ONLY TRUE)
            source_group("Shaders" FILES ${source})
        endif()
    endforeach()
    NoGraphicsAPI_disable_exceptions(${target})
    target_link_libraries(${target} PRIVATE NoGraphicsAPI_example_support)
endfunction()
