#include "precomp.hpp"

namespace cv {

    void Triangle::setVertexPosition(int index, Vec3f vertex)
    {
        vertices[index] = vertex;
    }

    void Triangle::setVertexColor(int index, Vec3f colorPerVertex)
    {
        color[index] = colorPerVertex;
    }

    Matx44f lookAtMatrixCal(const Vec3f& position, const Vec3f& lookat, const Vec3f& upVector)
    {
        Vec3f w, u;
        normalize(position - lookat, w, 1.0, 0.0, NORM_L1);
        normalize(upVector.cross(w), u, 1.0, 0.0, NORM_L1);

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

    bool insideTriangle(float x, float y, const Vec3f* vertices)
    {
        Vec3f A(vertices[0][0], vertices[0][1], 1.0), B(vertices[1][0], vertices[1][1], 1.0), C(vertices[2][0], vertices[2][1], 1.0);
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

    Vec3f barycentricCal(float x, float y, const Vec3f* vertices)
    {
       Vec3f A(vertices[0][0], vertices[0][1], 1.0), B(vertices[1][0], vertices[1][1], 1.0), C(vertices[2][0], vertices[2][1], 1.0);
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
                return Vec3f( 1.0 - beta - gamma, beta, gamma );
        }
    }

    void triangle_rendering(const Triangle& tri, int width, int height, bool isConstant,
                                    std::vector<float>& depth_buf, std::vector<Vec3f>& color_buf)
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
                    Vec3f barycentricCoord = barycentricCal(x + 0.5, y + 0.5, tri.vertices);
                    float alpha = barycentricCoord[0], beta = barycentricCoord[1], gamma = barycentricCoord[2];
                    float z_interpolated = 1.0 / (alpha / tri.vertices[0][2] + beta / tri.vertices[1][2] + gamma / tri.vertices[2][2]);

                    int index = (height - 1 - y) * width + x;
                    if (z_interpolated < depth_buf[index] && z_interpolated >= 0.0 && z_interpolated <= 1.0)
                    {
                        if (isConstant)
                            color_buf[index] = tri.getTriangleColor();
                            //color_buf[index] = Vec3f(255, 0, 0);
                        else
                            color_buf[index] = (alpha * tri.color[0] / tri.vertices[0][2] + beta * tri.color[1] / tri.vertices[1][2]
                                                + gamma * tri.color[2] / tri.vertices[2][2]) * z_interpolated;
                        depth_buf[index] = z_interpolated;
                    }
                }
            }
    }

    void triangleRasterize(const std::vector<Vec3f>& vertices, const std::vector<Vec3i>& indices,
                           const std::vector<Vec3f>& colors, const Vec3f& position, const Vec3f& lookat, const Vec3f& upVector,
                           float fovy, float zNear, float zFar, int width, int height, bool isConstant,
                           std::vector<float>& depth_buf, std::vector<Vec3f>& color_buf)
    {
        Matx44f lookAtMatrix = lookAtMatrixCal(position, lookat, upVector);
        Matx44f perspectMatrix = perspectMatrixCal((float)width / (float)height, fovy, zNear, zFar);
        Matx44f mvpMatrix = perspectMatrix * lookAtMatrix;

        depth_buf.resize(width * height);
        color_buf.resize(width * height);

        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(color_buf.begin(), color_buf.end(), Vec3f(0.0, 0.0, 0.0));

        for (int i = 0; i < indices.size(); i++)
        {
            Triangle tri;
            Vec4f ver[3] = {
                mvpMatrix * Vec4f(vertices[indices[i][0]][0], vertices[indices[i][0]][1], vertices[indices[i][0]][2], 1.0),
                mvpMatrix * Vec4f(vertices[indices[i][1]][0], vertices[indices[i][1]][1], vertices[indices[i][1]][2], 1.0),
                mvpMatrix * Vec4f(vertices[indices[i][2]][0], vertices[indices[i][2]][1], vertices[indices[i][2]][2], 1.0)
            };

            for (auto& vertex: ver)
            {
                divide(vertex, vertex[3], vertex);
            }

            for (int j = 0; j < 3; j++)
            {
                auto vertex = ver[j];
                vertex[0] = 0.5 * width * (vertex[0] + 1.0);
                vertex[1] = 0.5 * height * (vertex[1] + 1.0);
                vertex[2] = vertex[2] * 0.5 + 0.5;

                tri.setVertexPosition(j, Vec3f(vertex[0], vertex[1], vertex[2]));
            }

            for (int j = 0; j < 3; j++)
            {
                tri.setVertexColor(j, colors[indices[i][j]]);
            }

            triangle_rendering(tri, width, height, isConstant, depth_buf, color_buf);
        }
    }
}
