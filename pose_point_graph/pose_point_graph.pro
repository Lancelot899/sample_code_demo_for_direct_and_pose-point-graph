TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += ../sophus
INCLUDEPATH += /home/lancelot/opt/eigen

SOURCES += \
    pose_point_graph_example.cpp

##########################################################################
###################                g2o                  ##################
##########################################################################
INCLUDEPATH += /usr/local/include/
INCLUDEPATH += /usr/include/suitesparse

LIBS += /usr/lib/x86_64-linux-gnu/libcholmod.so.2.1.2
LIBS += -lg2o_core -lg2o_cli -lg2o_solver_cholmod -lg2o_parser -lg2o_stuff

