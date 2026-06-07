#include "reed_solomon.h"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace aerospace {

ReedSolomon::ReedSolomon()
{
    init_galois_field();
    init_generator_polynomial();
}

void ReedSolomon::init_galois_field()
{
    int field_size = 1 << GF_SYMBOL_SIZE;
    gf_exp_.resize(2 * field_size, 0);
    gf_log_.resize(field_size, 0);

    int x = 1;
    for (int i = 0; i < field_size - 1; ++i) {
        gf_exp_[i] = x;
        gf_log_[x] = i;
        x <<= 1;
        if (x & field_size) {
            x ^= GF_PRIMITIVE_POLY;
        }
    }

    for (int i = field_size - 1; i < 2 * field_size; ++i) {
        gf_exp_[i] = gf_exp_[i - (field_size - 1)];
    }
}

void ReedSolomon::init_generator_polynomial()
{
    int field_size = 1 << GF_SYMBOL_SIZE;
    generator_poly_ = {1};

    for (int i = 0; i < RS_ECC_LENGTH; ++i) {
        std::vector<int> factor = {gf_exp_[i], 1};
        generator_poly_ = poly_mul(generator_poly_, factor);
    }
}

int ReedSolomon::gf_add(int a, int b) const
{
    return a ^ b;
}

int ReedSolomon::gf_sub(int a, int b) const
{
    return a ^ b;
}

int ReedSolomon::gf_mul(int a, int b) const
{
    if (a == 0 || b == 0) return 0;
    int field_size = 1 << GF_SYMBOL_SIZE;
    return gf_exp_[gf_log_[a] + gf_log_[b]];
}

int ReedSolomon::gf_div(int a, int b) const
{
    if (a == 0) return 0;
    if (b == 0) return -1;
    int field_size = 1 << GF_SYMBOL_SIZE;
    return gf_exp_[(gf_log_[a] - gf_log_[b] + (field_size - 1)) % (field_size - 1)];
}

int ReedSolomon::gf_pow(int a, int power) const
{
    int field_size = 1 << GF_SYMBOL_SIZE;
    return gf_exp_[(gf_log_[a] * power) % (field_size - 1)];
}

int ReedSolomon::gf_inverse(int a) const
{
    int field_size = 1 << GF_SYMBOL_SIZE;
    return gf_exp_[(field_size - 1) - gf_log_[a]];
}

std::vector<int> ReedSolomon::poly_mul(const std::vector<int>& a, const std::vector<int>& b) const
{
    std::vector<int> result(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i + j] = gf_add(result[i + j], gf_mul(a[i], b[j]));
        }
    }
    return result;
}

std::vector<int> ReedSolomon::poly_mod(const std::vector<int>& a, const std::vector<int>& b) const
{
    std::vector<int> result = a;
    while (result.size() >= b.size() && !result.empty()) {
        int coeff = result[0];
        if (coeff != 0) {
            for (size_t i = 0; i < b.size(); ++i) {
                result[i] = gf_sub(result[i], gf_mul(coeff, b[i]));
            }
        }
        result.erase(result.begin());
    }
    return result;
}

std::vector<int> ReedSolomon::poly_scale(const std::vector<int>& poly, int scalar) const
{
    std::vector<int> result;
    result.reserve(poly.size());
    for (int c : poly) {
        result.push_back(gf_mul(c, scalar));
    }
    return result;
}

std::vector<int> ReedSolomon::poly_add(const std::vector<int>& a, const std::vector<int>& b) const
{
    size_t max_len = std::max(a.size(), b.size());
    std::vector<int> result(max_len, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        result[i + max_len - a.size()] = a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        result[i + max_len - b.size()] = gf_add(result[i + max_len - b.size()], b[i]);
    }
    return result;
}

int ReedSolomon::poly_eval(const std::vector<int>& poly, int x) const
{
    int result = 0;
    for (int coeff : poly) {
        result = gf_add(gf_mul(result, x), coeff);
    }
    return result;
}

std::vector<int> ReedSolomon::find_error_locator(const std::vector<int>& syndromes) const
{
    std::vector<int> error_locator = {1};
    std::vector<int> old_locator = {1};

    for (size_t i = 0; i < syndromes.size(); ++i) {
        int delta = syndromes[i];
        for (size_t j = 1; j < error_locator.size(); ++j) {
            delta = gf_add(delta, gf_mul(error_locator[error_locator.size() - 1 - j], syndromes[i - j]));
        }

        old_locator.insert(old_locator.begin(), 0);

        if (delta != 0) {
            if (old_locator.size() > error_locator.size()) {
                std::vector<int> new_locator = poly_scale(old_locator, delta);
                old_locator = poly_scale(error_locator, gf_inverse(delta));
                error_locator = new_locator;
            }
            error_locator = poly_add(error_locator, poly_scale(old_locator, delta));
        }
    }

    return error_locator;
}

std::vector<int> ReedSolomon::find_error_positions(const std::vector<int>& error_locator) const
{
    std::vector<int> positions;
    int field_size = 1 << GF_SYMBOL_SIZE;

    for (int i = 1; i < field_size; ++i) {
        if (poly_eval(error_locator, i) == 0) {
            positions.push_back((field_size - 1) - gf_log_[i]);
        }
    }

    return positions;
}

std::vector<int> ReedSolomon::correct_errors(const std::vector<int>& received,
                                              const std::vector<int>& syndromes,
                                              const std::vector<int>& error_positions) const
{
    std::vector<int> result = received;
    int field_size = 1 << GF_SYMBOL_SIZE;

    std::vector<int> error_evaluator = {0};
    for (size_t i = 0; i < syndromes.size(); ++i) {
        error_evaluator = poly_add(error_evaluator, {syndromes[i]});
        if (i < syndromes.size() - 1) {
            error_evaluator.push_back(0);
        }
    }

    std::vector<int> error_locator_deriv;
    for (size_t i = 0; i < error_evaluator.size() - 1; i += 2) {
        error_locator_deriv.push_back(error_evaluator[i + 1]);
    }

    for (int pos : error_positions) {
        if (pos < static_cast<int>(result.size())) {
            int x_inv = gf_exp_[(field_size - 1) - pos];
            int numerator = poly_eval(error_evaluator, x_inv);
            int denominator = 1;
            for (int p : error_positions) {
                if (p != pos) {
                    denominator = gf_mul(denominator, gf_sub(1, gf_mul(gf_exp_[p], x_inv)));
                }
            }
            int error = gf_div(numerator, denominator);
            result[result.size() - 1 - pos] = gf_sub(result[result.size() - 1 - pos], error);
        }
    }

    return result;
}

std::vector<uint8_t> ReedSolomon::encode(const std::vector<uint8_t>& data)
{
    if (data.size() > RS_DATA_LENGTH) {
        return {};
    }

    std::vector<int> msg(RS_BLOCK_LENGTH, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        msg[i] = data[i];
    }

    std::vector<int> padded_msg = msg;
    for (size_t i = 0; i < RS_ECC_LENGTH; ++i) {
        padded_msg.push_back(0);
    }

    std::vector<int> remainder = poly_mod(padded_msg, generator_poly_);

    std::vector<uint8_t> result(RS_BLOCK_LENGTH);
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = static_cast<uint8_t>(msg[i]);
    }
    for (size_t i = 0; i < RS_ECC_LENGTH; ++i) {
        if (i < remainder.size()) {
            result[data.size() + i] = static_cast<uint8_t>(remainder[i]);
        } else {
            result[data.size() + i] = 0;
        }
    }

    return result;
}

std::vector<uint8_t> ReedSolomon::decode(const std::vector<uint8_t>& data_with_ecc, int& corrected_errors)
{
    corrected_errors = 0;

    if (data_with_ecc.size() != RS_BLOCK_LENGTH) {
        return {};
    }

    std::vector<int> received;
    received.reserve(RS_BLOCK_LENGTH);
    for (uint8_t b : data_with_ecc) {
        received.push_back(static_cast<int>(b));
    }

    std::vector<int> syndromes(RS_ECC_LENGTH, 0);
    bool has_error = false;
    for (int i = 0; i < RS_ECC_LENGTH; ++i) {
        syndromes[i] = poly_eval(received, gf_exp_[i]);
        if (syndromes[i] != 0) {
            has_error = true;
        }
    }

    if (!has_error) {
        std::vector<uint8_t> result(RS_DATA_LENGTH);
        for (size_t i = 0; i < RS_DATA_LENGTH; ++i) {
            result[i] = static_cast<uint8_t>(received[i]);
        }
        return result;
    }

    std::vector<int> error_locator = find_error_locator(syndromes);
    std::vector<int> error_positions = find_error_positions(error_locator);

    if (error_positions.empty() || error_positions.size() > static_cast<size_t>(RS_ECC_LENGTH / 2)) {
        return {};
    }

    std::vector<int> corrected = correct_errors(received, syndromes, error_positions);
    corrected_errors = static_cast<int>(error_positions.size());

    std::vector<uint8_t> result(RS_DATA_LENGTH);
    for (size_t i = 0; i < RS_DATA_LENGTH; ++i) {
        result[i] = static_cast<uint8_t>(corrected[i]);
    }

    return result;
}

std::vector<std::vector<uint8_t>> ReedSolomon::encode_burst(const std::vector<std::vector<uint8_t>>& data_blocks)
{
    std::vector<std::vector<uint8_t>> results;
    results.reserve(data_blocks.size());
    for (const auto& block : data_blocks) {
        results.push_back(encode(block));
    }
    return results;
}

std::vector<std::vector<uint8_t>> ReedSolomon::decode_burst(const std::vector<std::vector<uint8_t>>& blocks, std::vector<int>& corrected_counts)
{
    std::vector<std::vector<uint8_t>> results;
    results.reserve(blocks.size());
    corrected_counts.clear();
    corrected_counts.reserve(blocks.size());

    for (const auto& block : blocks) {
        int count = 0;
        results.push_back(decode(block, count));
        corrected_counts.push_back(count);
    }

    return results;
}

}
