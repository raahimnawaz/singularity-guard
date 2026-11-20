// Fixed-size 3-vector and 3x3 matrix arithmetic: no allocation, no dependencies.
#pragma once

#include <array>
#include <cmath>

namespace armctl {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;  // m[row][col]

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 operator*(double s, const Vec3& a) {
    return {s * a[0], s * a[1], s * a[2]};
}

inline Vec3 operator*(const Vec3& a, double s) {
    return s * a;
}

inline Mat3 transpose(const Mat3& m) {
    Mat3 t{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            t[i][j] = m[j][i];
        }
    }
    return t;
}

inline Vec3 matvec(const Mat3& m, const Vec3& v) {
    return {
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    };
}

inline Mat3 matmul(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return out;
}

// Determinant by cofactor expansion, term for term with armsim/controllers.py.
inline double det3(const Mat3& m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Explicit adjugate inverse, unguarded: the naive law is meant to blow up.
inline Mat3 inv3(const Mat3& m) {
    const double d = det3(m);
    const Mat3 adj{{
        {m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2],
         m[0][1] * m[1][2] - m[0][2] * m[1][1]},
        {m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0],
         m[0][2] * m[1][0] - m[0][0] * m[1][2]},
        {m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1],
         m[0][0] * m[1][1] - m[0][1] * m[1][0]},
    }};
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = adj[i][j] / d;
        }
    }
    return out;
}

}  // namespace armctl
