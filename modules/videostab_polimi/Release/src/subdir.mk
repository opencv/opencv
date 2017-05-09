################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/deblurring.cpp \
../src/fast_marching.cpp \
../src/frame_source.cpp \
../src/global_motion.cpp \
../src/inpainting.cpp \
../src/log.cpp \
../src/motion_stabilizing.cpp \
../src/optical_flow.cpp \
../src/outlier_rejection.cpp \
../src/stabilizer.cpp \
../src/videostab.cpp \
../src/wobble_suppression.cpp 

OBJS += \
./src/deblurring.o \
./src/fast_marching.o \
./src/frame_source.o \
./src/global_motion.o \
./src/inpainting.o \
./src/log.o \
./src/motion_stabilizing.o \
./src/optical_flow.o \
./src/outlier_rejection.o \
./src/stabilizer.o \
./src/videostab.o \
./src/wobble_suppression.o 

CPP_DEPS += \
./src/deblurring.d \
./src/fast_marching.d \
./src/frame_source.d \
./src/global_motion.d \
./src/inpainting.d \
./src/log.d \
./src/motion_stabilizing.d \
./src/optical_flow.d \
./src/outlier_rejection.d \
./src/stabilizer.d \
./src/videostab.d \
./src/wobble_suppression.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/usr/local/include/opencv -I/usr/local/include -I"/home/rodrygojose/opencv/modules/videostab_polimi/include/opencv2/videostab" -O3 -p -pg -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


