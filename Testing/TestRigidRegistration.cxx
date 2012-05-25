#include "vtkLinearTransform.h"
#include "vtkMNITransformReader.h"
#include "vtkMNITransformWriter.h"
#include "vtkCamera.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkImageFastBlur.h"
#include "vtkImageProperty.h"
#include "vtkImageRegistration.h"
#include "vtkImageReslice.h"
#include "vtkImageResliceMapper.h"
#include "vtkImageSincInterpolator.h"
#include "vtkImageSlice.h"
#include "vtkImageStack.h"
#include "vtkInteractorStyleImage.h"
#include "vtkMINCImageReader.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkPNGWriter.h"
#include "vtkPointData.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkTransform.h"
#include "vtkWindowToImageFilter.h"

const double kTHRESHOLD = 1.0;

namespace {

//---------------------Function Prototypes----------------------------------
int RegressionTestMatrix(vtkMatrix4x4 *matrix, const char *baselinePath);
bool FileExists(const char* filePath);
void ReadDICOMImage(vtkImageData *data,
                    vtkMatrix4x4 *matrix,
                    const char *directoryName);
void SetViewFromMatrix(vtkRenderer *renderer,
                       vtkInteractorStyleImage *istyle,
                       vtkMatrix4x4 *matrix);
void ReadMINCImage(vtkImageData *data,
                   vtkMatrix4x4 *matrix,
                   const char *fileName);
void GetMatrixFromFile(double matrix[3][3], const char *path);
void GetPrimitiveArrayFromObject(double target[3][3], vtkMatrix4x4 *source);
double RadianToDegree(double radian);
bool QuaternionsAreEqual(double baseline[4],
                         double target[4],
                         double threshold);

//-------------------------------------------------------------------------
void GetMatrixFromFile(double matrix[3][3], const char *path)
{
  vtkSmartPointer<vtkMNITransformReader> reader =
    vtkSmartPointer<vtkMNITransformReader>::New();
  reader->SetFileName(path);
  vtkLinearTransform *transform =
    vtkLinearTransform::SafeDownCast(reader->GetTransform());
  vtkMatrix4x4 *inputMatrix = transform->GetMatrix();

  for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
      {
      matrix[i][j] = inputMatrix->GetElement(i, j);
      }
    }
}

//-------------------------------------------------------------------------
void GetPrimitiveArrayFromObject(double target[3][3], vtkMatrix4x4 *source)
{
  for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
      {
        target[i][j] = source->GetElement(i, j);
      }
    }
}

//-------------------------------------------------------------------------
double RadianToDegree(double radian)
{
  return (radian * 180 / (atan(1)*4));
}

//-------------------------------------------------------------------------
bool QuaternionsAreEqual(double baseline[4],
                         double target[4],
                         double threshold)
{
  double baseTheta = 0.0;
  double targetTheta = 0.0;

  // Calculate the quaternions' angles
  baseTheta = atan2(sqrt(pow(baseline[1], 2) +
                         pow(baseline[2], 2) +
                         pow(baseline[3], 2)), baseline[0]);
  targetTheta = atan2(sqrt(pow(target[1], 2) +
                           pow(target[2], 2) +
                           pow(target[3], 2)), target[0]);

  // Convert angles from radian to degrees for threshold comparison
  double baseDegree = RadianToDegree(baseTheta);
  double targetDegree = RadianToDegree(targetTheta);

  // Return true if the difference of the angles is within the threshold
  if (abs(baseDegree - targetDegree) < threshold)
  {
    return true;
  }

  return false;
}

//-------------------------------------------------------------------------
int RegressionTestMatrix(vtkMatrix4x4 *matrix, const char *baselinePath)
{
  double baselineMatrix[3][3];
  double baselineQuaternion[4];
  GetMatrixFromFile(baselineMatrix, baselinePath);
  vtkMath::Matrix3x3ToQuaternion(baselineMatrix, baselineQuaternion);

  double targetMatrix[3][3];
  double targetQuaternion[4];
  GetPrimitiveArrayFromObject(targetMatrix, matrix);
  vtkMath::Matrix3x3ToQuaternion(targetMatrix, targetQuaternion);

  if (QuaternionsAreEqual(baselineQuaternion, targetQuaternion, kTHRESHOLD)
      == true)
    {
    return 0;
    }

  return 1;
}

//-------------------------------------------------------------------------
bool FileExists(const char* filePath)
{
  ifstream file(filePath);
  return file;
}

//-------------------------------------------------------------------------
void ReadDICOMImage(
  vtkImageData *data, vtkMatrix4x4 *matrix, const char *directoryName)
{
  // read the image
  vtkSmartPointer<vtkDICOMImageReader> reader =
    vtkSmartPointer<vtkDICOMImageReader>::New();

  reader->SetDirectoryName(directoryName);
  reader->Update();

  // the reader flips the image and reverses the ordering, so undo these
  vtkSmartPointer<vtkImageReslice> flip =
    vtkSmartPointer<vtkImageReslice>::New();

  flip->SetInputConnection(reader->GetOutputPort());
  flip->SetResliceAxesDirectionCosines(
    1,0,0, 0,-1,0, 0,0,-1);
  flip->Update();

  vtkImageData *image = flip->GetOutput();

  // get the data
  data->CopyStructure(image);
  data->GetPointData()->PassData(image->GetPointData());
  data->SetOrigin(0,0,0);

  // generate the matrix
  float *position = reader->GetImagePositionPatient();
  float *orientation = reader->GetImageOrientationPatient();
  float *xdir = &orientation[0];
  float *ydir = &orientation[3];
  float zdir[3];
  vtkMath::Cross(xdir, ydir, zdir);

  for (int i = 0; i < 3; i++)
    {
    matrix->Element[i][0] = xdir[i];
    matrix->Element[i][1] = ydir[i];
    matrix->Element[i][2] = zdir[i];
    matrix->Element[i][3] = position[i];
    }
  matrix->Element[3][0] = 0;
  matrix->Element[3][1] = 0;
  matrix->Element[3][2] = 0;
  matrix->Element[3][3] = 1;
  matrix->Modified();
}

//-------------------------------------------------------------------------
void ReadMINCImage(
  vtkImageData *data, vtkMatrix4x4 *matrix, const char *fileName)
{
  // read the image
  vtkSmartPointer<vtkMINCImageReader> reader =
    vtkSmartPointer<vtkMINCImageReader>::New();

  reader->SetFileName(fileName);
  reader->Update();

  double spacing[3];
  reader->GetOutput()->GetSpacing(spacing);
  spacing[0] = fabs(spacing[0]);
  spacing[1] = fabs(spacing[1]);
  spacing[2] = fabs(spacing[2]);

  // flip the image rows into a DICOM-style ordering
  vtkSmartPointer<vtkImageReslice> flip =
    vtkSmartPointer<vtkImageReslice>::New();

  flip->SetInputConnection(reader->GetOutputPort());
  flip->SetResliceAxesDirectionCosines(
    -1,0,0, 0,-1,0, 0,0,1);
  flip->SetOutputSpacing(spacing);
  flip->Update();

  vtkImageData *image = flip->GetOutput();

  // get the data
  data->CopyStructure(image);
  data->GetPointData()->PassData(image->GetPointData());

  // generate the matrix, but modify to use DICOM coords
  static double xyFlipMatrix[16] =
    { -1, 0, 0, 0,  0, -1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1 };
  // correct for the flip that was done earlier
  vtkMatrix4x4::Multiply4x4(*reader->GetDirectionCosines()->Element,
                            xyFlipMatrix, *matrix->Element);
  // do the left/right, up/down dicom-to-minc transformation
  vtkMatrix4x4::Multiply4x4(xyFlipMatrix, *matrix->Element, *matrix->Element);
  matrix->Modified();
}

//-------------------------------------------------------------------------
void SetViewFromMatrix(
  vtkRenderer *renderer,
  vtkInteractorStyleImage *istyle,
  vtkMatrix4x4 *matrix)
{
  istyle->SetCurrentRenderer(renderer);

  // This view assumes the data uses the DICOM Patient Coordinate System.
  // It provides a right-is-left view of axial and coronal images
  double viewRight[4] = { 1.0, 0.0, 0.0, 0.0 };
  double viewUp[4] = { 0.0, -1.0, 0.0, 0.0 };

  matrix->MultiplyPoint(viewRight, viewRight);
  matrix->MultiplyPoint(viewUp, viewUp);

  istyle->SetImageOrientation(viewRight, viewUp);
}

} /* end of namespace */

//-------------------------------------------------------------------------
int TestRigidRegistration(int argc, char *argv[])
{
  std::string outputImagePath(argv[6]);
  std::string outputMatrixPath(argv[9]);

  // -------------------------------------------------------
  // parameters for registration

  int interpolatorType = vtkImageRegistration::Linear;
  double transformTolerance = 0.1; // tolerance on transformation result
  int numberOfBins = 64; // for Mattes' mutual information
  double initialBlurFactor = 4.0;

  // -------------------------------------------------------
  // load the images

  int n = 0;

  // argv[7] = MPRAGE DICOM Directory passed in from CMakeList.txt
  // argv[8] = FLAIR DICOM Directory passed in from CMakeList.txt
  std::string sourceDir = argv[7];
  std::string targetDir = argv[8];

  // Read the source image
  vtkSmartPointer<vtkImageData> sourceImage =
    vtkSmartPointer<vtkImageData>::New();
  vtkSmartPointer<vtkMatrix4x4> sourceMatrix =
    vtkSmartPointer<vtkMatrix4x4>::New();

  ReadDICOMImage(sourceImage, sourceMatrix, sourceDir.c_str());

  // Read the target image
  vtkSmartPointer<vtkImageData> targetImage =
    vtkSmartPointer<vtkImageData>::New();
  vtkSmartPointer<vtkMatrix4x4> targetMatrix =
    vtkSmartPointer<vtkMatrix4x4>::New();

  ReadDICOMImage(targetImage, targetMatrix, targetDir.c_str());

  if (targetImage == 0 || sourceImage == 0)
    {
    // Failed to load images. Fail the test.
    return 1;
    }

  // -------------------------------------------------------
  // display the images

  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  vtkSmartPointer<vtkInteractorStyleImage> istyle =
    vtkSmartPointer<vtkInteractorStyleImage>::New();

  istyle->SetInteractionModeToImage3D();
  interactor->SetInteractorStyle(istyle);
  renderWindow->SetInteractor(interactor);
  renderWindow->AddRenderer(renderer);

  vtkSmartPointer<vtkImageSlice> sourceActor =
    vtkSmartPointer<vtkImageSlice>::New();
  vtkSmartPointer<vtkImageResliceMapper> sourceMapper =
    vtkSmartPointer<vtkImageResliceMapper>::New();
  vtkSmartPointer<vtkImageProperty> sourceProperty =
    vtkSmartPointer<vtkImageProperty>::New();

  sourceMapper->SetInput(sourceImage);
  sourceMapper->SliceAtFocalPointOn();
  sourceMapper->SliceFacesCameraOn();
  sourceMapper->ResampleToScreenPixelsOff();

  double sourceRange[2];
  sourceImage->GetScalarRange(sourceRange);
  sourceProperty->SetInterpolationTypeToLinear();
  sourceProperty->SetColorWindow((sourceRange[1]-sourceRange[0]));
  sourceProperty->SetColorLevel(0.5*(sourceRange[0]+sourceRange[1]));
  sourceProperty->CheckerboardOn();
  sourceProperty->SetCheckerboardSpacing(40,40);

  sourceActor->SetMapper(sourceMapper);
  sourceActor->SetProperty(sourceProperty);
  sourceActor->SetUserMatrix(sourceMatrix);

  vtkSmartPointer<vtkImageSlice> targetActor =
    vtkSmartPointer<vtkImageSlice>::New();
  vtkSmartPointer<vtkImageResliceMapper> targetMapper =
    vtkSmartPointer<vtkImageResliceMapper>::New();
  vtkSmartPointer<vtkImageProperty> targetProperty =
    vtkSmartPointer<vtkImageProperty>::New();

  targetMapper->SetInput(targetImage);
  targetMapper->SliceAtFocalPointOn();
  targetMapper->SliceFacesCameraOn();
  targetMapper->ResampleToScreenPixelsOff();

  double targetRange[2];
  targetImage->GetScalarRange(targetRange);
  targetProperty->SetInterpolationTypeToLinear();
  targetProperty->SetColorWindow((targetRange[1]-targetRange[0]));
  targetProperty->SetColorLevel(0.5*(targetRange[0]+targetRange[1]));

  targetActor->SetMapper(targetMapper);
  targetActor->SetProperty(targetProperty);
  targetActor->SetUserMatrix(targetMatrix);

  vtkSmartPointer<vtkImageStack> imageStack =
    vtkSmartPointer<vtkImageStack>::New();
  imageStack->AddImage(targetActor);
  imageStack->AddImage(sourceActor);

  renderer->AddViewProp(imageStack);
  renderer->SetBackground(0,0,0);

  renderWindow->SetSize(720,720);

  double bounds[6], center[4];
  targetImage->GetBounds(bounds);
  center[0] = 0.5*(bounds[0] + bounds[1]);
  center[1] = 0.5*(bounds[2] + bounds[3]);
  center[2] = 0.5*(bounds[4] + bounds[5]);
  center[3] = 1.0;
  targetMatrix->MultiplyPoint(center, center);

  vtkCamera *camera = renderer->GetActiveCamera();
  renderer->ResetCamera();
  camera->SetFocalPoint(center);
  camera->ParallelProjectionOn();
  camera->SetParallelScale(132);
  SetViewFromMatrix(renderer, istyle, targetMatrix);
  renderer->ResetCameraClippingRange();

  renderWindow->Render();

  // -------------------------------------------------------
  // prepare for registration

  // get information about the images
  double targetSpacing[3], sourceSpacing[3];
  targetImage->GetSpacing(targetSpacing);
  sourceImage->GetSpacing(sourceSpacing);

  for (int jj = 0; jj < 3; jj++)
    {
    targetSpacing[jj] = fabs(targetSpacing[jj]);
    sourceSpacing[jj] = fabs(sourceSpacing[jj]);
    }

  double minSpacing = sourceSpacing[0];
  if (minSpacing > sourceSpacing[1])
    {
    minSpacing = sourceSpacing[1];
    }
  if (minSpacing > sourceSpacing[2])
    {
    minSpacing = sourceSpacing[2];
    }

  // blur source image with Blackman-windowed sinc
  vtkSmartPointer<vtkImageSincInterpolator> sourceBlurKernel =
    vtkSmartPointer<vtkImageSincInterpolator>::New();
  sourceBlurKernel->SetWindowFunctionToBlackman();

  // reduce the source resolution
  vtkSmartPointer<vtkImageFastBlur> sourceBlur =
    vtkSmartPointer<vtkImageFastBlur>::New();
  sourceBlur->SetInput(sourceImage);
  sourceBlur->SetResizeMethodToOutputSpacing();
  sourceBlur->SetInterpolator(sourceBlurKernel);
  sourceBlur->InterpolateOn();

  // blur target with Blackman-windowed sinc
  vtkSmartPointer<vtkImageSincInterpolator> targetBlurKernel =
    vtkSmartPointer<vtkImageSincInterpolator>::New();
  targetBlurKernel->SetWindowFunctionToBlackman();

  // keep target at full resolution
  vtkSmartPointer<vtkImageFastBlur> targetBlur =
    vtkSmartPointer<vtkImageFastBlur>::New();
  targetBlur->SetInput(targetImage);
  targetBlur->SetResizeMethodToOutputSpacing();
  targetBlur->SetInterpolator(targetBlurKernel);
  targetBlur->InterpolateOn();

  // get the initial transformation
  vtkSmartPointer<vtkMatrix4x4> matrix =
    vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->DeepCopy(targetMatrix);
  matrix->Invert();
  vtkMatrix4x4::Multiply4x4(matrix, sourceMatrix, matrix);

  // set up the registration
  vtkSmartPointer<vtkImageRegistration> registration =
    vtkSmartPointer<vtkImageRegistration>::New();
  registration->SetTargetImageInputConnection(targetBlur->GetOutputPort());
  registration->SetSourceImageInputConnection(sourceBlur->GetOutputPort());
  registration->SetInitializerTypeToCentered();
  registration->SetTransformTypeToRigid();
  //registration->SetTransformTypeToScaleTargetAxes();
  //registration->SetTransformTypeToAffine();
  registration->SetMetricTypeToNormalizedMutualInformation();
  //registration->SetMetricTypeToNormalizedCrossCorrelation();
  registration->SetInterpolatorType(interpolatorType);
  registration->SetJointHistogramSize(numberOfBins,numberOfBins);
  registration->SetMetricTolerance(1e-4);
  registration->SetTransformTolerance(transformTolerance);
  registration->SetMaximumNumberOfIterations(500);

  // -------------------------------------------------------
  // do the registration

  // the registration starts at low-resolution
  double blurFactor = initialBlurFactor;
  // two stages for each resolution:
  // first without interpolation, and then with interpolation
  int stage = 0;
  // will be set to "true" when registration is initialized
  bool initialized = false;

  for (;;)
    {
    if (stage == 0)
      {
      registration->SetInterpolatorTypeToNearest();
      registration->SetTransformTolerance(minSpacing*blurFactor);
      }
    else
      {
      registration->SetInterpolatorType(interpolatorType);
      registration->SetTransformTolerance(transformTolerance*blurFactor);
      }
    if (blurFactor < 1.1)
      {
      // full resolution: no blurring or resampling
      sourceBlur->SetInterpolator(0);
      sourceBlur->InterpolateOff();
      sourceBlur->SetOutputSpacing(sourceSpacing);
      sourceBlur->Update();

      targetBlur->SetInterpolator(0);
      sourceBlur->InterpolateOff();
      targetBlur->SetOutputSpacing(targetSpacing);
      targetBlur->Update();
      }
    else
      {
      // reduced resolution: set the blurring
      double spacing[3];
      for (int j = 0; j < 3; j++)
        {
        spacing[j] = blurFactor*minSpacing;
        if (spacing[j] < sourceSpacing[j])
          {
          spacing[j] = sourceSpacing[j];
          }
        }

      sourceBlurKernel->SetBlurFactors(
        spacing[0]/sourceSpacing[0],
        spacing[1]/sourceSpacing[1],
        spacing[2]/sourceSpacing[2]);

      sourceBlur->SetOutputSpacing(spacing);
      sourceBlur->Update();

      targetBlurKernel->SetBlurFactors(
        blurFactor*minSpacing/targetSpacing[0],
        blurFactor*minSpacing/targetSpacing[1],
        blurFactor*minSpacing/targetSpacing[2]);

      targetBlur->Update();
      }

    if (initialized)
      {
      // re-initialize with the matrix from the previous step
      registration->SetInitializerTypeToNone();
      matrix->DeepCopy(registration->GetTransform()->GetMatrix());
      }

    registration->Initialize(matrix);

    initialized = true;

    while (registration->Iterate())
      {
      // registration->UpdateRegistration();
      // will iterate until convergence or failure
      vtkMatrix4x4::Multiply4x4(
        targetMatrix,registration->GetTransform()->GetMatrix(),sourceMatrix);
      sourceMatrix->Modified();
      interactor->Render();
      }

    // prepare for next iteration
    if (stage == 1)
      {
      blurFactor /= 2.0;
      if (blurFactor < 0.9)
        {
        break;
        }
      }
    stage = (stage + 1) % 2;
    }

  // Determine if it is necessary to build the baseline image.
  // Return test failure after creating baseline.
  bool builtBaselineFiles = false;
  if (!FileExists(outputImagePath.c_str()))
    {
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
      vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetMagnification(1);
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->Update();

    vtkSmartPointer<vtkPNGWriter> writer =
      vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(outputImagePath.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();
    builtBaselineFiles = true;
    }

  // Determine if it is necessary to build the baseline matrix.
  // Return test failure after creating baseline.
  if (!FileExists(outputMatrixPath.c_str()))
    {
    vtkSmartPointer<vtkMNITransformWriter> transformWriter =
      vtkSmartPointer<vtkMNITransformWriter>::New();
    transformWriter->SetFileName(outputMatrixPath.c_str());
    transformWriter->SetTransform(registration->GetTransform());
    registration->Update();
    transformWriter->Write();
    builtBaselineFiles = true;
    }

  if (builtBaselineFiles)
    {
    // First time building the baseline files. Fail the test.
    cerr << "First pass building baseline images and transformation matrices. Rerun the tests." << std::endl;
    return 1;
    }

  // If the baseline image and matrix already exist, perform
  // regression tests
  bool testHasFailed = false;

  renderWindow->Render();
  int retVal = vtkRegressionTestImage(renderWindow);
  if (retVal == vtkRegressionTester::DO_INTERACTOR)
    {
    interactor->Start();
    }

  if (retVal == 0)
    {
    cerr << "Image Regression Test Failed." << std::endl;
    testHasFailed = true;
    }

  if (RegressionTestMatrix(registration->GetTransform()->GetMatrix(),
                           outputMatrixPath.c_str()) == 1)
    {
    cerr << "Matrix Regression Test Failed." << std::endl;
    testHasFailed = true;
    }

  return testHasFailed ? 1 : 0;
}
