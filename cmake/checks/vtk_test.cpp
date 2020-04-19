#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkMath.h>

int main()
{
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  return 0;
}
