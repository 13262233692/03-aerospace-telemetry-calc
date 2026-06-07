#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace aerospace {

constexpr int STATE_DIM = 6;
constexpr int MEAS_DIM = 3;

using Vector6 = std::array<double, STATE_DIM>;
using Vector3 = std::array<double, MEAS_DIM>;
using Matrix6x6 = std::array<std::array<double, STATE_DIM>, STATE_DIM>;
using Matrix3x6 = std::array<std::array<double, STATE_DIM>, MEAS_DIM>;
using Matrix6x3 = std::array<std::array<double, MEAS_DIM>, STATE_DIM>;
using Matrix3x3 = std::array<std::array<double, MEAS_DIM>, MEAS_DIM>;

namespace math {

inline void matrix_multiply_6x6(const Matrix6x6& A, const Matrix6x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_multiply_6x6_T(const Matrix6x6& A, const Matrix6x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i][k] * B[j][k];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_multiply_T_6x6(const Matrix6x6& A, const Matrix6x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[k][i] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_add_6x6(const Matrix6x6& A, const Matrix6x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

inline void matrix_sub_6x6(const Matrix6x6& A, const Matrix6x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

inline void matrix_identity_6x6(Matrix6x6& A) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            A[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

inline void matrix_scalar_multiply_6x6(Matrix6x6& A, double scalar) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            A[i][j] *= scalar;
        }
    }
}

inline void matrix_multiply_6x3_3x6(const Matrix6x3& A, const Matrix3x6& B, Matrix6x6& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < MEAS_DIM; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_multiply_3x6_6x6(const Matrix3x6& A, const Matrix6x6& B, Matrix3x6& C) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_multiply_3x6_6x3(const Matrix3x6& A, const Matrix6x3& B, Matrix3x3& C) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_add_3x3(const Matrix3x3& A, const Matrix3x3& B, Matrix3x3& C) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

inline bool matrix_inverse_3x3(const Matrix3x3& A, Matrix3x3& inv) {
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (std::abs(det) < 1e-15) {
        return false;
    }

    double inv_det = 1.0 / det;

    inv[0][0] =  (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    inv[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * inv_det;
    inv[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
    inv[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * inv_det;
    inv[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    inv[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) * inv_det;
    inv[2][0] =  (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;
    inv[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) * inv_det;
    inv[2][2] =  (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

    return true;
}

inline void matrix_multiply_6x6_6x3(const Matrix6x6& A, const Matrix6x3& B, Matrix6x3& C) {
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < MEAS_DIM; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

inline void matrix_multiply_T_3x6(const Matrix3x6& A, Matrix6x3& AT) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

inline void matrix_multiply_vector_6x6(const Matrix6x6& A, const Vector6& x, Vector6& y) {
    for (int i = 0; i < STATE_DIM; ++i) {
        double sum = 0.0;
        for (int k = 0; k < STATE_DIM; ++k) {
            sum += A[i][k] * x[k];
        }
        y[i] = sum;
    }
}

inline void vector_add_6(const Vector6& a, const Vector6& b, Vector6& c) {
    for (int i = 0; i < STATE_DIM; ++i) {
        c[i] = a[i] + b[i];
    }
}

inline void vector_sub_6(const Vector6& a, const Vector6& b, Vector6& c) {
    for (int i = 0; i < STATE_DIM; ++i) {
        c[i] = a[i] - b[i];
    }
}

inline void vector_sub_3(const Vector3& a, const Vector3& b, Vector3& c) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        c[i] = a[i] - b[i];
    }
}

inline void matrix_multiply_vector_3x6(const Matrix3x6& A, const Vector6& x, Vector3& y) {
    for (int i = 0; i < MEAS_DIM; ++i) {
        double sum = 0.0;
        for (int k = 0; k < STATE_DIM; ++k) {
            sum += A[i][k] * x[k];
        }
        y[i] = sum;
    }
}

inline void matrix_multiply_vector_6x3(const Matrix6x3& A, const Vector3& x, Vector6& y) {
    for (int i = 0; i < STATE_DIM; ++i) {
        double sum = 0.0;
        for (int k = 0; k < MEAS_DIM; ++k) {
            sum += A[i][k] * x[k];
        }
        y[i] = sum;
    }
}

} // namespace math
} // namespace aerospace
