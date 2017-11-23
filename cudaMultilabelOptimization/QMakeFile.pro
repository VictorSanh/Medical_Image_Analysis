TEMPLATE = app
TARGET = cudaMultilabelOptimization
QT += core
HEADERS += cudaDataterm.cuh \
    cudaOptimization.cuh \
    cutil.h
SOURCES += main.cpp \
    dataterm.cpp \
    imageSegmentation.cpp \
    dice.cpp \
    automaticValueExploration.cpp \
    segmentationManipulation.cpp
CUDA_SOURCES += cudaDataterm.cu \
    cudaOptimization.cu \
    cudaDataterm.cu

# Compiler flags tuned for my system
QMAKE_CXXFLAGS += -I../ \
    -O99 \
    -pipe \
#    -g \
    -Wall
LIBS += -L../ \
    -L/usr/lib/nvidia-current \
    -lcuda \
    -lcudart \
    -lgsl \
    -lgslcblas \
    -lm \
    -lX11

# #######################################################################
# CUDA
# #######################################################################


CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/samples
INCLUDEPATH += $$CUDA_DIR/include

QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
cuda.commands = $$CUDA_DIR/bin/nvcc \
    -c \
    $$NVFLAGS \
    -Xcompiler \
    $$join(QMAKE_CXXFLAGS,",") \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_NAME} \
    -o \
    ${QMAKE_FILE_OUT}
cuda.dependcy_type = TYPE_C
cuda.depend_command = nvcc \
    -M \
    -Xcompiler \
    $$join(QMAKE_CXXFLAGS,",") \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_NAME} \
    | \
    sed \
    "s,^.*: ,," \
    | \
    sed \
    "s,^ *,," \
    | \
    tr \
    -d \
    '\\\n'
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
