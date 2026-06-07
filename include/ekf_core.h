#pragma once

#include "matrix_math.h"
#include <atomic>
#include <memory>
#include <cstring>
#include <vector>

#ifdef BUILD_PYTHON_BINDINGS
#include <Python.h>
#endif

namespace aerospace {

struct EKFCoreState {
    Vector6 x;
    Matrix6x6 P;
    double timestamp;
};

#ifdef BUILD_PYTHON_BINDINGS
class GILScopedRelease {
public:
    GILScopedRelease() : saved_(nullptr) {
        if (PyGILState_Check()) {
            saved_ = PyEval_SaveThread();
        }
    }

    ~GILScopedRelease() {
        if (saved_) {
            PyEval_RestoreThread(saved_);
        }
    }

    GILScopedRelease(const GILScopedRelease&) = delete;
    GILScopedRelease& operator=(const GILScopedRelease&) = delete;

private:
    PyThreadState* saved_;
};

class GILScopedAcquire {
public:
    GILScopedAcquire() : state_(PyGILState_Ensure()) {}

    ~GILScopedAcquire() {
        PyGILState_Release(state_);
    }

    GILScopedAcquire(const GILScopedAcquire&) = delete;
    GILScopedAcquire& operator=(const GILScopedAcquire&) = delete;

private:
    PyGILState_STATE state_;
};
#else
class GILScopedRelease {
public:
    GILScopedRelease() = default;
    ~GILScopedRelease() = default;
    GILScopedRelease(const GILScopedRelease&) = delete;
    GILScopedRelease& operator=(const GILScopedRelease&) = delete;
};

class GILScopedAcquire {
public:
    GILScopedAcquire() = default;
    ~GILScopedAcquire() = default;
    GILScopedAcquire(const GILScopedAcquire&) = delete;
    GILScopedAcquire& operator=(const GILScopedAcquire&) = delete;
};
#endif

class ExtendedKalmanFilterCore {
public:
    ExtendedKalmanFilterCore() {
        reset();
    }

    void reset() {
        state_.timestamp = 0.0;
        math::matrix_identity_6x6(state_.P);
        math::matrix_scalar_multiply_6x6(state_.P, 1e6);
        std::fill(state_.x.begin(), state_.x.end(), 0.0);

        math::matrix_identity_6x6(Q_);
        math::matrix_scalar_multiply_6x6(Q_, 1e-6);
    }

    void set_process_noise(const Matrix6x6& Q) {
        Q_ = Q;
    }

    void set_state(const Vector6& x, const Matrix6x6& P, double timestamp) {
        state_.x = x;
        state_.P = P;
        state_.timestamp = timestamp;
    }

    void predict(const Matrix6x6& F, double dt, double timestamp) {
        GILScopedRelease gil_release;

        Matrix6x6 FP;
        math::matrix_multiply_6x6(F, state_.P, FP);

        Matrix6x6 FPT;
        math::matrix_multiply_6x6_T(F, FP, FPT);

        Matrix6x6 new_P;
        math::matrix_add_6x6(FPT, Q_, new_P);

        state_.P = new_P;
        state_.timestamp = timestamp;
    }

    void predict_state(const Vector6& x_new, const Matrix6x6& F, double dt, double timestamp) {
        GILScopedRelease gil_release;

        state_.x = x_new;

        Matrix6x6 FP;
        math::matrix_multiply_6x6(F, state_.P, FP);

        Matrix6x6 FPT;
        math::matrix_multiply_6x6_T(F, FP, FPT);

        Matrix6x6 new_P;
        math::matrix_add_6x6(FPT, Q_, new_P);

        state_.P = new_P;
        state_.timestamp = timestamp;
    }

    bool update_position(const Vector3& measurement, const Matrix3x3& R) {
        GILScopedRelease gil_release;

        Matrix3x6 H;
        for (int i = 0; i < MEAS_DIM; ++i) {
            for (int j = 0; j < STATE_DIM; ++j) {
                H[i][j] = 0.0;
            }
            H[i][i] = 1.0;
        }

        Matrix6x3 HT;
        math::matrix_multiply_T_3x6(H, HT);

        Matrix3x6 HP;
        math::matrix_multiply_3x6_6x6(H, state_.P, HP);

        Matrix3x3 S;
        math::matrix_multiply_3x6_6x3(H, HT, S);
        math::matrix_add_3x3(S, R, S);

        Matrix3x3 S_inv;
        if (!math::matrix_inverse_3x3(S, S_inv)) {
            return false;
        }

        Matrix6x3 K;
        math::matrix_multiply_6x6_6x3(state_.P, HT, K);
        Matrix6x3 K_tmp;
        for (int i = 0; i < STATE_DIM; ++i) {
            for (int j = 0; j < MEAS_DIM; ++j) {
                double sum = 0.0;
                for (int k = 0; k < MEAS_DIM; ++k) {
                    sum += K[i][k] * S_inv[k][j];
                }
                K_tmp[i][j] = sum;
            }
        }
        K = K_tmp;

        Vector3 h_x;
        h_x[0] = state_.x[0];
        h_x[1] = state_.x[1];
        h_x[2] = state_.x[2];

        Vector3 y;
        math::vector_sub_3(measurement, h_x, y);

        Vector6 K_y;
        math::matrix_multiply_vector_6x3(K, y, K_y);

        Vector6 x_new;
        math::vector_add_6(state_.x, K_y, x_new);

        Matrix6x3 KH;
        math::matrix_multiply_6x3_3x6(K, H, KH);

        Matrix6x6 I;
        math::matrix_identity_6x6(I);

        Matrix6x6 I_KH;
        math::matrix_sub_6x6(I, KH, I_KH);

        Matrix6x6 new_P;
        math::matrix_multiply_6x6(I_KH, state_.P, new_P);

        state_.x = x_new;
        state_.P = new_P;

        return true;
    }

    const EKFCoreState& get_state() const {
        return state_;
    }

    void get_state_vector(double* out_x) const {
        std::memcpy(out_x, state_.x.data(), sizeof(double) * STATE_DIM);
    }

    void get_covariance_matrix(double* out_P) const {
        for (int i = 0; i < STATE_DIM; ++i) {
            std::memcpy(out_P + i * STATE_DIM, state_.P[i].data(), sizeof(double) * STATE_DIM);
        }
    }

    double get_timestamp() const {
        return state_.timestamp;
    }

    void predict_batch(const std::vector<Matrix6x6>& F_list,
                       const std::vector<Vector6>& x_list,
                       const std::vector<double>& dt_list,
                       const std::vector<double>& timestamps,
                       std::vector<EKFCoreState>& out_states) {
        GILScopedRelease gil_release;

        size_t n = F_list.size();
        out_states.resize(n);

        EKFCoreState current = state_;

        for (size_t i = 0; i < n; ++i) {
            Matrix6x6 FP;
            math::matrix_multiply_6x6(F_list[i], current.P, FP);

            Matrix6x6 FPT;
            math::matrix_multiply_6x6_T(F_list[i], FP, FPT);

            Matrix6x6 new_P;
            math::matrix_add_6x6(FPT, Q_, new_P);

            current.x = x_list[i];
            current.P = new_P;
            current.timestamp = timestamps[i];

            out_states[i] = current;
        }
    }

private:
    EKFCoreState state_;
    Matrix6x6 Q_;
};

} // namespace aerospace
