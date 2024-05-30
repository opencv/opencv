#include <iostream>
#include <map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX 1
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

// model data should be identical to the code from tests
enum class ModelType
{
    Empty = 0,
    File = 1,
    Clipping = 2,
    Color = 3,
    Centered = 4
};

static void generateNormals(const std::vector<Vec3f>& points, const std::vector<std::vector<int>>& indices,
                            std::vector<Vec3f>& normals)
{
    std::vector<std::vector<Vec3f>> preNormals(points.size(), std::vector<Vec3f>());

    for (const auto& tri : indices)
    {
        Vec3f p0 = points[tri[0]];
        Vec3f p1 = points[tri[1]];
        Vec3f p2 = points[tri[2]];

        Vec3f cross = cv::normalize((p1 - p0).cross(p2 - p0));
        for (int i = 0; i < 3; i++)
        {
            preNormals[tri[i]].push_back(cross);
        }
    }

    normals.reserve(points.size());
    for (const auto& pn : preNormals)
    {
        Vec3f sum { };
        for (const auto& n : pn)
        {
            sum += n;
        }
        normals.push_back(cv::normalize(sum));
    }
}

class ModelData
{
public:
    ModelData(ModelType type = ModelType::Empty, std::string objPath = { })
    {
        switch (type)
        {
        case ModelType::Empty:
        {
            position = Vec3d(0.0, 0.0, 0.0);
            lookat   = Vec3d(0.0, 0.0, 0.0);
            upVector = Vec3d(0.0, 1.0, 0.0);

            fovy = 45.0;
            zNear = 0.1;
            zFar = 50;
            scaleCoeff = 1000.0;

            vertices = std::vector<Vec3f>(4, {2.0f, 0, -2.0f});
            colors   = std::vector<Vec3f>(4, {0, 0, 1.0f});
            indices = { };
        }
        break;
        case ModelType::File:
        {
            position = Vec3d( 1.9, 0.4, 1.3);
            lookat   = Vec3d( 0.0, 0.0, 0.0);
            upVector = Vec3d( 0.0, 1.0, 0.0);

            fovy = 45.0;
            zNear = 0.1;
            zFar = 50;
            scaleCoeff = 1000.0;

            objectPath = objPath;
            std::vector<vector<int>> indvec;
            loadMesh(objectPath, vertices, indvec);
            // using per-vertex normals as colors
            generateNormals(vertices, indvec, colors);
            if (vertices.size() != colors.size())
            {
                std::runtime_error("Model should contain normals for each vertex");
            }
            for (const auto &vec : indvec)
            {
                indices.push_back({vec[0], vec[1], vec[2]});
            }

            for (auto &color : colors)
            {
                color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2]));
            }
        }
        break;
        case ModelType::Clipping:
        {
            position = Vec3d(0.0, 0.0, 5.0);
            lookat   = Vec3d(0.0, 0.0, 0.0);
            upVector = Vec3d(0.0, 1.0, 0.0);

            fovy = 45.0;
            zNear = 0.1;
            zFar = 50;
            scaleCoeff = 1000.0;

            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -6.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -2.5, -5.0}, {-1.0,  1.0, -5.0},
                {-6.5, -1.0, -3.0}, {-2.5, -2.0, -3.0}, { 1.0,  1.0, -5.0},
            };

            indices = { {0, 1, 2}, {3, 4, 5}, {6, 7, 8} };

            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);
            Vec3f col3(150.0,  10.0, 238.0);

            col1 *= (1.f / 255.f);
            col2 *= (1.f / 255.f);
            col3 *= (1.f / 255.f);

            colors =
            {
                col1, col2, col3,
                col2, col3, col1,
                col3, col1, col2,
            };
        }
        break;
        case ModelType::Centered:
        {
            position = Vec3d(0.0, 0.0, 5.0);
            lookat   = Vec3d(0.0, 0.0, 0.0);
            upVector = Vec3d(0.0, 1.0, 0.0);

            fovy = 45.0;
            zNear = 0.1;
            zFar = 50;
            scaleCoeff = 1000.0;

            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -2.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -1.5, -5.0}, {-1.0,  0.5, -5.0},
            };

            indices = { {0, 1, 2}, {3, 4, 5} };

            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);

            col1 *= (1.f / 255.f);
            col2 *= (1.f / 255.f);

            colors =
            {
                col1, col2, col1,
                col2, col1, col2,
            };
        }
        break;
        case ModelType::Color:
        {
            position = Vec3d(0.0, 0.0, 5.0);
            lookat   = Vec3d(0.0, 0.0, 0.0);
            upVector = Vec3d(0.0, 1.0, 0.0);

            fovy = 60.0;
            zNear = 0.1;
            zFar = 50;
            scaleCoeff = 1000.0;

            vertices =
            {
                { 2.0,  0.0, -2.0},
                { 0.0,  2.0, -3.0},
                {-2.0,  0.0, -2.0},
                { 0.0, -2.0,  1.0},
            };

            indices = { {0, 1, 2}, {0, 2, 3} };

            colors =
            {
                { 0.0f, 0.0f, 1.0f},
                { 0.0f, 1.0f, 0.0f},
                { 1.0f, 0.0f, 0.0f},
                { 0.0f, 1.0f, 0.0f},
            };
        }
        break;

        default:
            CV_Error(Error::StsBadArg, "Unknown model type");
            break;
        }
    }

    ModelData(std::string modelPath, double fov, double nearPlane, double farPlane, double scale, Vec3d pos, Vec3d center, Vec3d up)
    {
        objectPath = modelPath;
        position = pos;
        lookat   = center;
        upVector = up;
        fovy = fov;
        zNear = nearPlane;
        zFar = farPlane;
        scaleCoeff = scale;

        std::vector<vector<int>> indvec;

        loadMesh(objectPath, vertices, indvec, noArray(), colors);
        if (vertices.size() != colors.size())
        {
            std::runtime_error("Model should contain normals for each vertex");
        }
        for (const auto &vec : indvec)
        {
            indices.push_back({vec[0], vec[1], vec[2]});
        }
    }

    Vec3d position;
    Vec3d lookat;
    Vec3d upVector;

    double fovy, zNear, zFar, scaleCoeff;

    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors;

    string objectPath;
};


struct DrawData
{
    ogl::Arrays arr;
    ogl::Buffer indices;
};

void draw(void* userdata);

void draw(void* userdata)
{
    DrawData* data = static_cast<DrawData*>(userdata);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

static void generateImage(cv::Size imgSz, TriangleShadingType shadingType, TriangleCullingMode cullingMode,
                          const ModelData& modelData, cv::Mat& colorImage, cv::Mat& depthImage)
{
    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", imgSz.width, imgSz.height);

    DrawData data;

    std::vector<Vec3f> vertices;
    std::vector<Vec4f> colors4f;
    std::vector<int> idxLinear;

    if (shadingType == RASTERIZE_SHADING_FLAT)
    {
        // rearrange vertices and colors for flat shading
        int ctr = 0;
        for (const auto& idx : modelData.indices)
        {
            for (int i = 0; i < 3; i++)
            {
                vertices.push_back(modelData.vertices[idx[i]]);
                idxLinear.push_back(ctr++);
            }

            Vec3f ci = modelData.colors[idx[0]];
            for (int i = 0; i < 3; i++)
            {
                colors4f.emplace_back(ci[0], ci[1], ci[2], 1.f);
            }
        }
    }
    else
    {
        vertices = modelData.vertices;
        for (const auto& c : modelData.colors)
        {
            Vec3f ci = (shadingType == RASTERIZE_SHADING_SHADED) ? c: cv::Vec3f::all(1.f);
            colors4f.emplace_back(ci[0], ci[1], ci[2], 1.0);
        }

        for (const auto& idx : modelData.indices)
        {
            for (int i = 0; i < 3; i++)
            {
                idxLinear.push_back(idx[i]);
            }
        }
    }

    data.arr.setVertexArray(vertices);
    data.arr.setColorArray(colors4f);
    data.indices.copyFrom(idxLinear);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(modelData.fovy, (double)imgSz.width / imgSz.height, modelData.zNear, modelData.zFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
    gluLookAt(modelData.position[0], modelData.position[1], modelData.position[2],
              modelData.lookat  [0], modelData.lookat  [1], modelData.lookat  [2],
              modelData.upVector[0], modelData.upVector[1], modelData.upVector[2]);

    if (cullingMode == RASTERIZE_CULLING_NONE)
    {
        glDisable(GL_CULL_FACE);
    }
    else
    {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        if (cullingMode == RASTERIZE_CULLING_CW)
        {
            glFrontFace(GL_CW);
        }
        else
        {
            glFrontFace(GL_CCW);
        }
    }

    glEnable(GL_DEPTH_TEST);

    cv::setOpenGlDrawCallback("OpenGL", draw, &data);

    const int framesToSkip = 10;
    for (int f = 0; f < framesToSkip; f++)
    {
        updateWindow("OpenGL");

        colorImage = cv::Mat(imgSz.height, imgSz.width, CV_8UC3);
        glReadPixels(0, 0, imgSz.width, imgSz.height, GL_RGB, GL_UNSIGNED_BYTE, colorImage.data);
        cv::cvtColor(colorImage, colorImage, cv::COLOR_RGB2BGR);
        cv::flip(colorImage, colorImage, 0);

        depthImage = cv::Mat(imgSz.height, imgSz.width, CV_32F);
        glReadPixels(0, 0, imgSz.width, imgSz.height, GL_DEPTH_COMPONENT, GL_FLOAT, depthImage.data);
        // map from [0, 1] to [zNear, zFar]
        for (auto it = depthImage.begin<float>(); it != depthImage.end<float>(); ++it)
        {
            *it = (float)(modelData.zNear * modelData.zFar / (double(*it) * (modelData.zNear - modelData.zFar) + modelData.zFar));
        }
        cv::flip(depthImage, depthImage, 0);
        depthImage.convertTo(depthImage, CV_16U, modelData.scaleCoeff);

        char key = (char)waitKey(40);
        if (key == 27)
            break;
    }

    cv::setOpenGlDrawCallback("OpenGL", 0, 0);
    cv::destroyAllWindows();
}


int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
            "{ help h usage ? |      | show this message }"
            "{ outPath        |      | output path for generated images }"
            "{ modelPath      |      | path to 3d model to render }"
            "{ custom         |      | pass it to use custom camera parameters instead of iterating through test parameters }"
            "{ fov            | 45.0 | (if custom parameters are used) field of view }"
            "{ posx           | 1.0  | (if custom parameters are used) camera position x }"
            "{ posy           | 1.0  | (if custom parameters are used) camera position y }"
            "{ posz           | 1.0  | (if custom parameters are used) camera position z }"
            "{ lookatx        | 0.0  | (if custom parameters are used) lookup camera direction x }"
            "{ lookaty        | 0.0  | (if custom parameters are used) lookup camera direction y }"
            "{ lookatz        | 0.0  | (if custom parameters are used) lookup camera direction z }"
            "{ upx            | 0.0  | (if custom parameters are used) up camera direction x }"
            "{ upy            | 1.0  | (if custom parameters are used) up camera direction y }"
            "{ upz            | 0.0  | (if custom parameters are used) up camera direction z }"
            "{ resx           | 640  | (if custom parameters are used) camera resolution x }"
            "{ resy           | 480  | (if custom parameters are used) camera resolution y }"
            "{ zNear          | 0.1  | (if custom parameters are used) near z clipping plane }"
            "{ zFar           |  50  | (if custom parameters are used) far z clipping plane }"
            "{ scaleCoeff     | 1000 | (if custom parameters are used) scale coefficient for saving depth }"
            "{ shading        |      | (if custom parameters are used) shading type: white/flat/shaded }"
            "{ culling        |      | (if custom parameters are used) culling type: none/cw/ccw }"
            "{ colorPath      |      | (if custom parameters are used) output path for color image }"
            "{ depthPath      |      | (if custom parameters are used) output path for depth image }"
    );
    parser.about("This app is used to generate test data for triangleRasterize() function");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string modelPath = parser.get<std::string>("modelPath");
    if (modelPath.empty())
    {
        std::cout << "No model path given" << std::endl;
        return -1;
    }

    if (parser.has("custom"))
    {
        double fov = parser.get<double>("fov");
        Vec3d position, lookat, upVector;
        position[0] = parser.get<double>("posx");
        position[1] = parser.get<double>("posy");
        position[2] = parser.get<double>("posz");
        lookat[0]   = parser.get<double>("lookatx");
        lookat[1]   = parser.get<double>("lookaty");
        lookat[2]   = parser.get<double>("lookatz");
        upVector[0] = parser.get<double>("upx");
        upVector[1] = parser.get<double>("upy");
        upVector[2] = parser.get<double>("upz");
        Size res;
        res.width  = parser.get<int>("resx");
        res.height = parser.get<int>("resy");
        double zNear = parser.get<double>("zNear");
        double zFar  = parser.get<double>("zFar");
        double scaleCoeff = parser.get<double>("scaleCoeff");

        std::map<std::string, cv::TriangleShadingType> shadingTxt = {
            { "white",  RASTERIZE_SHADING_WHITE  },
            { "flat",   RASTERIZE_SHADING_FLAT   },
            { "shaded", RASTERIZE_SHADING_SHADED },
        };
        cv::TriangleShadingType shadingType = shadingTxt.at(parser.get<std::string>("shading"));

        std::map<std::string, cv::TriangleCullingMode> cullingTxt = {
            { "none", RASTERIZE_CULLING_NONE },
            { "cw",   RASTERIZE_CULLING_CW   },
            { "ccw",  RASTERIZE_CULLING_CCW  },
        };
        cv::TriangleCullingMode cullingMode = cullingTxt.at(parser.get<std::string>("culling"));

        std::string colorPath = parser.get<std::string>("colorPath");
        std::string depthPath = parser.get<std::string>("depthPath");

        Mat colorImage, depthImage;
        ModelData modelData(modelPath, fov, zNear, zFar, scaleCoeff, position, lookat, upVector);
        generateImage(res, shadingType, cullingMode, modelData, colorImage, depthImage);

        cv::imwrite(colorPath, colorImage);
        cv::imwrite(depthPath, depthImage);
    }
    else
    {
        std::string outPath = parser.get<std::string>("outPath");
        if (outPath.empty())
        {
            std::cout << "No output path given" << std::endl;
            return -1;
        }

        std::array<cv::Size, 4> resolutions = {cv::Size{700, 700}, cv::Size{640, 480}, cv::Size(256, 256), cv::Size(320, 240)};
        std::vector<std::pair<cv::TriangleShadingType, std::string>> shadingTxt = {
            {RASTERIZE_SHADING_WHITE,  "White"},
            {RASTERIZE_SHADING_FLAT,   "Flat"},
            {RASTERIZE_SHADING_SHADED, "Shaded"},
        };
        std::vector<std::pair<cv::TriangleCullingMode, std::string>> cullingTxt = {
            {RASTERIZE_CULLING_NONE, "None"},
            {RASTERIZE_CULLING_CW,   "CW"},
            {RASTERIZE_CULLING_CCW,  "CCW"},
        };
        std::vector<std::pair<ModelType, std::string>> modelTxt = {
            {ModelType::File,     "File"},
            {ModelType::Clipping, "Clipping"},
            {ModelType::Color,    "Color"},
            {ModelType::Centered, "Centered"},
        };

        for (const auto& res : resolutions)
        {
            for (const auto shadingPair : shadingTxt)
            {
                cv::TriangleShadingType shadingType = shadingPair.first;
                std::string shadingName = shadingPair.second;

                for (const auto cullingPair : cullingTxt)
                {
                    cv::TriangleCullingMode cullingMode = cullingPair.first;
                    std::string cullingName = cullingPair.second;

                    for (const auto modelPair : modelTxt)
                    {
                        ModelType modelType = modelPair.first;
                        std::string modelName = modelPair.second;

                        std::string suffix = cv::format("%s_%dx%d_Cull%s", modelName.c_str(), res.width, res.height, cullingName.c_str());

                        std::cout << suffix + "_" + shadingName << "..." << std::endl;

                        cv::Mat colorImage, depthImage;

                        ModelData modelData(modelType, modelPath);
                        generateImage(res, shadingType, cullingMode, modelData, colorImage, depthImage);

                        std::string gtPathColor = outPath + "/example_image_" + suffix + "_" + shadingName + ".png";
                        std::string gtPathDepth = outPath + "/depth_image_"   + suffix + ".png";

                        cv::imwrite(gtPathColor, colorImage);
                        cv::imwrite(gtPathDepth, depthImage);
                    }
                }
            }
        }
    }

    return 0;
}
