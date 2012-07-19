# Common stuff that all the subdirectories can include...


# How should be build everything
COLOR = yes
DEBUG = yes

FQTOP        := $(shell (cd $(TOP); pwd))	# Fully Qualified TOP directory
ROOT         := $(TOP)/root
LIB_DIR      := $(ROOT)/lib
BIN_DIR      := $(ROOT)/bin
SBIN_DIR     := $(ROOT)/sbin
INCLUDE_DIR  := $(ROOT)/include


# If you're using intel's compiler:
#CXX := icpc
#CC  := icc
#LD  := xild

CINCLUDES = -I. \
	$(shell pkg-config --cflags opencv) \
	-I$(INCLUDE_DIR) \
#	-I/usr/local/include/lapackpp \
	-I/usr/local/include/opencv

CXXINCLUDES  := $(CINCLUDES)

CAM_LIBS     :=  $(shell pkg-config --libs opencv) 
CAM_INCLUDES := $(shell pkg-config --cflags opencv)
#LAPACKPP_LIBS     := $(shell pkg-config --libs lapackpp) 
#LAPACKPP_INCLUDES := /usr/local/lib/lapackpp
LINKINCLUDES := -L. -L$(LIB_DIR)
EXEC_TEST_LIBS := # -lboost_test_exec_monitor
UNIT_TEST_LIBS := # -lboost_unit_test_framework

ifeq ($(DEBUG), yes)
DEBUG_LEVEL  := -ggdb
else
DEBUG_LEVEL  := -O2
endif

GCOVFLAGS    := $(DEBUG_LEVEL) -fprofile-arcs -ftest-coverage
OPTIMIZEFLAGS:= $(DEBUG_LEVEL) -march=nocona -fPIC

ifneq (,${TEST-COVERAGE})
CFLAGS       := $(CINCLUDES)   $(GCOVFLAGS) $(FLAGS)
CXXFLAGS     := $(CXXINCLUDES) $(GCOVFLAGS) $(FLAGS)
GCOVLIB      := -lgcov
else
CFLAGS       := $(CINCLUDES)   $(OPTIMIZEFLAGS) $(FLAGS)
CXXFLAGS     := $(CXXINCLUDES) $(OPTIMIZEFLAGS) $(FLAGS)
endif

COMPILEC     := $(CC)  $(CFLAGS)
COMPILECXX   := $(CXX) $(CXXFLAGS)
LINK         := $(CXX) $(LINKINCLUDES)

ifeq ($(COLOR), yes)
BUILD_MSG = "\033[33mBuilding $@\033[0m"
else
BUILD_MSG = "Building $@"
endif


ALL_LIBRARIES = $(LIBRARIES)
ifneq (,${STATIC})
ALL_LIBRARIES = $(LIBRARIES:%so=%a)
endif

.PHONY: all
	$(COMPILECXX) -c $< -MD -MT $@ -MF $(@:%o=%d) 
all:: $(ALL_LIBRARIES) $(BINARIES)
ifneq (,$(BINARIES))
	cp $(BINARIES) $(ROOT)/bin/
endif
ifneq (,$(ALL_LIBRARIES))
	cp $(ALL_LIBRARIES) $(ROOT)/lib/
endif
	@set -e; for subdir in $(SUBDIRS); do \
		echo "Making $@ in $$subdir"; \
		make -C $$subdir $@; \
	done

.PHONY: clean
clean::
	rm -f $(ALL_LIBRARIES) $(BINARIES) $(FLUFF) *.a *.o *.d  *.pyc *.gcno *.gcov *.gcda app.info
#	Clean up any test directories:
	@if [ -e "test" ]; then \
		make -C test clean; \
	fi


tags:: 
	ctags -R

.PHONY: test-coverage
# test-coverage needs to make clean at the top level
# make clean recurses all by itself, so we only want to do this
# at the top level, and not recursively:
ifeq (0, ${MAKELEVEL})	
test-coverage:: clean
	@echo "\033[31m*********************************"
	@echo "* REBUILDING WITH TEST-COVERAGE *"
	@echo "*********************************\033[0m"
	@echo ${MAKE}
	make TEST-COVERAGE=1
	make test TEST-COVERAGE=1
endif


.PHONY: test 
install indent clean test tags test-coverage::
	@set -e; for subdir in $(SUBDIRS); do \
		echo "Making $@ in $$subdir"; \
		make -C $$subdir $@; \
	done

test-coverage::
	@echo "Test coverage:"
	gcov *.cpp *.c

# for test, make the test directories after recursing

test::
	@if [ -e "test" ]; then \
		make -C test; \
	fi

indent::
	@set -e; for file in $(wildcard *.c *.h); do \
		echo "indenting $$file"; \
		indent -gnu -nut -cli2 -ss -v -i2 $$file; \
		rm -f $$file~; \
	done

#
# Default rules for building...
#
-include *.d

#   Compile C++ files, and create appropriate .d files for each of them.

%.o : %.cpp
	@echo $(BUILD_MSG)
	$(COMPILECXX) -c $< -MD -MT $@ -MF $(@:%o=%d) 


#   Compile C files, and create appropriate .d files for each of them.

%.o : %.c
	@echo $(BUILD_MSG)
	$(COMPILEC) -c $< -MD -MT $@ -MF $(@:%o=%d)


#$(BINARIES): $($(addsuffix _OBJS, $(BINARIES)))
#	$(LINK) $< -MD -MT $@ -MF $(@:%o=%d) -o $@
#	cp $@ $(BIN_DIR)


