#include "precomp.hpp"

namespace cv {

    struct Triangle
    {
        Vec4f vertices[3];
        Vec3f color[3];

        void setVertexPosition(int index, Vec4f vertex);
        void setVertexColor(int index, Vec3f color);

        Vec3f getTriangleColor() const { return color[0]; }
    };

    void Triangle::setVertexPosition(int index, Vec4f vertex)
    {
        vertices[index] = vertex;
    }

    void Triangle::setVertexColor(int index, Vec3f colorPerVertex)
    {
        color[index] = colorPerVertex;
    }

    Vec3f normalize_vector(Vec3f a)
    {
        float length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        return Vec3f(a[0] / length, a[1] / length, a[2] / length);
    }

    Matx44f lookAtMatrixCal(const Vec3f& position, const Vec3f& lookat, const Vec3f& upVector)
    {
        Vec3f w, u;
        w = normalize_vector(position - lookat);
        u = normalize_vector(upVector.cross(w));

        Vec3f v = w.cross(u);
        Vec4f w_prime(w[0], w[1], w[2], 0.0f), u_prime(u[0], u[1], u[2], 0.0f), v_prime(v[0], v[1], v[2], 0.0f), identity(0.0f, 0.0f, 0.0f, 1.0f);
        Matx44f res(u_prime[0], u_prime[1], u_prime[2], u_prime[3],
                    v_prime[0], v_prime[1], v_prime[2], v_prime[3],
                    w_prime[0], w_prime[1], w_prime[2], w_prime[3],
                    identity[0], identity[1], identity[2], identity[3]);

        Matx44f translate(1.0f, 0.0f, 0.0f, -(float)position[0],
            0.0f, 1.0f, 0.0f, -(float)position[1],
            0.0f, 0.0f, 1.0f, -(float)position[2],
            0.0f, 0.0f, 0.0f, 1.0f);
        res = res * translate;

        return res;
    }

    Matx44f perspectMatrixCal(float aspect, float fovy, float zNear, float zFar)
    {
        float radian = (fovy / 360.0f) * M_PI;
        float d = 1.0f / std::tan(radian);
        Matx44f res(d / aspect, 0, 0, 0,
            0, d, 0, 0,
            0, 0, (zNear + zFar) / (zNear - zFar), 2 * zFar * zNear / (zNear - zFar),
            0, 0, -1, 0);
        return res;
    }

    bool insideTriangle(float x, float y, const Vec4f* vertices)
    {
        Vec3f A(vertices[0][0], vertices[0][1], 1.0);
        Vec3f B(vertices[1][0], vertices[1][1], 1.0);
        Vec3f C(vertices[2][0], vertices[2][1], 1.0);
        Vec3f P(x, y, 1.0);

        Vec3f ACcrossAB = (C - A).cross(B - A);
        Vec3f ACcrossAP = (C - A).cross(P - A);

        Vec3f ABcrossAC = -ACcrossAB;
        Vec3f ABcrossAP = (B - A).cross(P - A);

        if (ACcrossAB.dot(ACcrossAP) >= 0 && ABcrossAC.dot(ABcrossAP) >= 0)
        {
            float beta = norm(ACcrossAP) / norm(ACcrossAB);
            float gamma = norm(ABcrossAP) / norm(ABcrossAC);
            if (beta + gamma <= 1)
                return true;
        }

        return false;
    }

    Vec3f barycentricCal(float x, float y, const Vec4f* vertices, int width, int height)
    {
        Vec3f A(vertices[0][0] / width, vertices[0][1] / height, 1.0);
        Vec3f B(vertices[1][0] / width, vertices[1][1] / height, 1.0);
        Vec3f C(vertices[2][0] / width, vertices[2][1] / height, 1.0);
        Vec3f P(x / width, y / height, 1.0);

        Vec3f ACcrossAB = (C - A).cross(B - A);
        Vec3f ACcrossAP = (C - A).cross(P - A);

        Vec3f ABcrossAC = -ACcrossAB;
        Vec3f ABcrossAP = (B - A).cross(P - A);

        if (ACcrossAB.dot(ACcrossAP) >= 0 && ABcrossAC.dot(ABcrossAP) >= 0)
        {
            float beta = norm(ACcrossAP) / norm(ACcrossAB);
            float gamma = norm(ABcrossAP) / norm(ABcrossAC);
            if (beta + gamma <= 1)
                return Vec3f( 1.0 - beta - gamma, beta, gamma );
        }
    }

    void triangle_rendering(const Triangle& tri, int width, int height, bool shadingMode,
                                    Mat& depth_buf, Mat& color_buf)
    {
        float min_x = width, max_x = 0;
        float min_y = height, max_y = 0;

        for (int i = 0; i < 3; i++)
        {
            min_x = std::max(std::min(tri.vertices[i][0], min_x), 0.0f);
            max_x = std::min(std::max(tri.vertices[i][0], max_x), (float)width);
            min_y = std::max(std::min(tri.vertices[i][1], min_y), 0.0f);
            max_y = std::min(std::max(tri.vertices[i][1], max_y), (float)height);
        }

        for(int y = min_y; y < max_y; y++)
            for (int x = min_x; x < max_x; x++)
            {
                if (insideTriangle(x + 0.5, y + 0.5, tri.vertices))
                {
                    Vec3f barycentricCoord = barycentricCal(x + 0.5, y + 0.5, tri.vertices, width, height);
                    float alpha = barycentricCoord[0], beta = barycentricCoord[1], gamma = barycentricCoord[2];
                    float z_interpolated = 1.0f / (alpha / tri.vertices[0][3] + beta / tri.vertices[1][3] + gamma / tri.vertices[2][3]);

                    float z = depth_buf.at<float>(y, x);
                    if (z_interpolated < z)
                    {
                        if (shadingMode)
                            color_buf.at<Vec3f>(y, x) = tri.getTriangleColor();
                        else {
                            float r1 = alpha * z_interpolated / tri.vertices[0][3] * tri.color[0][0] + beta * z_interpolated / tri.vertices[1][3] * tri.color[1][0]
                                + gamma * z_interpolated / tri.vertices[2][3] * tri.color[2][0];
                            float g1 = alpha * z_interpolated / tri.vertices[0][3] * tri.color[0][1] + beta * z_interpolated / tri.vertices[1][3] * tri.color[1][1]
                                + gamma * z_interpolated / tri.vertices[2][3] * tri.color[2][1];
                            float b1 = alpha * z_interpolated / tri.vertices[0][3] * tri.color[0][2] + beta * z_interpolated / tri.vertices[1][3] * tri.color[1][2]
                                + gamma * z_interpolated / tri.vertices[2][3] * tri.color[2][2];

                            color_buf.at<Vec3f>(y, x) = Vec3f(r1, g1, b1);
                        }
                        depth_buf.at<float>(y, x) = z_interpolated;
                    }
                }
            }
    }

    void triangleRasterize(InputArray vertices, InputArray indices,
                           InputArray colors, InputArray cameraMatrix, int width, int height, bool shadingMode,
                           OutputArray depth_buf, OutputArray color_buf)
    {
        Mat camera = cameraMatrix.getMat();

        Vec3f position = camera.row(0);
        Vec3f lookat = camera.row(1);
        Vec3f upVector = camera.row(2);
        float fovy = camera.at<float>(3, 0), zNear = camera.at<float>(3, 1), zFar = camera.at<float>(3, 2);

        Matx44f lookAtMatrix = lookAtMatrixCal(position, lookat, upVector);
        Matx44f perspectMatrix = perspectMatrixCal((float)width / (float)height, fovy, zNear, zFar);
        Matx44f mvpMatrix = perspectMatrix * lookAtMatrix;

        Mat depth_buf_mat = depth_buf.getMat();
        Mat color_buf_mat = color_buf.getMat();

        if (vertices.empty() && indices.empty() && colors.empty())
            return;

        std::vector<Vec3f> verticesVector = vertices.getMat();
        std::vector<Vec3f> indicesVector = indices.getMat();
        std::vector<Vec3f> colorsVector = colors.getMat();

        for (int i = 0; i < indicesVector.size(); i++)
        {
            Triangle tri;
            Vec4f ver[3] = {
                mvpMatrix * Vec4f(verticesVector[indicesVector[i][0]][0], verticesVector[indicesVector[i][0]][1], verticesVector[indicesVector[i][0]][2], 1.0),
                mvpMatrix * Vec4f(verticesVector[indicesVector[i][1]][0], verticesVector[indicesVector[i][1]][1], verticesVector[indicesVector[i][1]][2], 1.0),
                mvpMatrix * Vec4f(verticesVector[indicesVector[i][2]][0], verticesVector[indicesVector[i][2]][1], verticesVector[indicesVector[i][2]][2], 1.0)
            };

            for (auto& vertex: ver)
            {
                vertex = Vec4f(vertex[0] / vertex[3], vertex[1] / vertex[3], vertex[2] / vertex[3], vertex[3]);
            }

            for (int j = 0; j < 3; j++)
            {
                auto vertex = ver[j];
                vertex[0] = 0.5 * width * (vertex[0] + 1.0);
                vertex[1] = 0.5 * height * (vertex[1] + 1.0);
                vertex[2] = vertex[2] * 0.5 + 0.5;

                tri.setVertexPosition(j, Vec4f(vertex[0], vertex[1], vertex[2], vertex[3]));
            }

            for (int j = 0; j < 3; j++)
            {
                tri.setVertexColor(j, colorsVector[indicesVector[i][j]]);
            }

            triangle_rendering(tri, width, height, shadingMode, depth_buf_mat, color_buf_mat);
        }
    }
}
