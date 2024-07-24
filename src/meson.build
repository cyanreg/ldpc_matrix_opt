build_opts = [
    '-D_ISOC11_SOURCE', '-D_XOPEN_SOURCE=700', '-U__STRICT_ANSI__',

    # Warnings
    '-Wundef', '-Wmissing-prototypes', '-Wshadow', '-Wparentheses',
    '-Wpointer-arith', '-Wno-pointer-sign',

    # Warnings to treat as errors
    '-Werror=implicit-function-declaration',
]

static_build = get_option('default_library') == 'static'

# Required dependencies
liblmo_dependencies = [
    dependency('vulkan'),
    dependency('libavutil'),
    dependency('threads'),

    cc.find_library('m', required : true),
]

# Base files
liblmo_sources = [
    'main.c',
    'vulkan_utils.c',

    # Version
    vcs_tag(command: ['git', 'rev-parse', '--short', 'HEAD'],
                      input: 'version.c.in', output: 'version.c',
                      fallback: 'release')
]

glslang_dep = dependency('', required: false)
shaderc_dep = dependency('', required: false)

if get_option('glsl') == 'shaderc'
    shaderc_dep = dependency('shaderc', required: false)
endif
if shaderc_dep.found() == false
    glslang_dep = dependency('glslang', required: true)
endif

if shaderc_dep.found() == true
    liblmo_dependencies += shaderc_dep
    liblmo_sources += 'vulkan_shaderc.c'
    conf.set('CONFIG_LIBSHADERC', 1)
endif

if glslang_dep.found() == true
    liblmo_dependencies += glslang_dep
    liblmo_sources += 'vulkan_glslang.c'
    conf.set('CONFIG_LIBGLSLANG', 1)
endif

shader_files = [
    'ec.comp',
    'bp.comp',
    'spa.comp',
]

file2string = find_program('file2string.py')
foreach file: shader_files
    shader_file = custom_target(file,
        input: file,
        output: file + '.inc',
        command: [file2string, '@INPUT@', '@OUTPUT@'],
    )
    liblmo_sources += shader_file
endforeach

# Generate local configure file
configure_file(
    output: 'config.h',
    configuration: conf,
)

liblmo_headers = [
]

# Check for wmain support (Windows/MinGW)
if cc.links('int wmain() { return 0; }', args: '-municode')
     conf.set('HAVE_WMAIN', 1)
     build_opts += '-municode'
endif

add_global_arguments(build_opts, language: 'c')

#libmocha_inc = include_directories('./include')
#liblmo_dep = library('lmo', liblmo_sources,
#    install: true,
#    dependencies: liblmo_dependencies,
#    soversion: ver_major,
#    include_directories: liblmo_inc,
#)

executable('lmo',
    install: true,
    sources: liblmo_sources,
    dependencies: liblmo_dependencies,
)
