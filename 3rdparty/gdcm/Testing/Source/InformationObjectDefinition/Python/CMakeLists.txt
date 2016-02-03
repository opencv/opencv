# Define the tests for gdcm-python
# gdcm-python
set(GDCM_PYTHON_TEST_SRCS
  TestMRModule
  )

# Loop over files and create executables
foreach(name ${GDCM_PYTHON_TEST_SRCS})
  ADD_PYTHON_TEST(${name}Python ${name}.py)
endforeach()
