################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Neural_Network_Lib/Src/model_parameters.c \
../Neural_Network_Lib/Src/model_utils.c 

OBJS += \
./Neural_Network_Lib/Src/model_parameters.o \
./Neural_Network_Lib/Src/model_utils.o 

C_DEPS += \
./Neural_Network_Lib/Src/model_parameters.d \
./Neural_Network_Lib/Src/model_utils.d 


# Each subdirectory must supply rules for building sources it contributes
Neural_Network_Lib/Src/%.o Neural_Network_Lib/Src/%.su Neural_Network_Lib/Src/%.cyclo: ../Neural_Network_Lib/Src/%.c Neural_Network_Lib/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F407xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Neural_Network_Lib-2f-Src

clean-Neural_Network_Lib-2f-Src:
	-$(RM) ./Neural_Network_Lib/Src/model_parameters.cyclo ./Neural_Network_Lib/Src/model_parameters.d ./Neural_Network_Lib/Src/model_parameters.o ./Neural_Network_Lib/Src/model_parameters.su ./Neural_Network_Lib/Src/model_utils.cyclo ./Neural_Network_Lib/Src/model_utils.d ./Neural_Network_Lib/Src/model_utils.o ./Neural_Network_Lib/Src/model_utils.su

.PHONY: clean-Neural_Network_Lib-2f-Src

