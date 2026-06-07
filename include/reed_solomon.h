#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace aerospace {

class ReedSolomon {
public:
    static constexpr size_t RS_BLOCK_LENGTH = 255;
    static constexpr size_t RS_DATA_LENGTH = 223;
    static constexpr size_t RS_ECC_LENGTH = 32;
    static constexpr int GF_SYMBOL_SIZE = 8;
    static constexpr int GF_PRIMITIVE_POLY = 0x11D;

    ReedSolomon();
    ~ReedSolomon() = default;

    std::vector<uint8_t> encode(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decode(const std::vector<uint8_t>& data_with_ecc, int& corrected_errors);

    std::vector<std::vector<uint8_t>> encode_burst(const std::vector<std::vector<uint8_t>>& data_blocks);
    std::vector<std::vector<uint8_t>> decode_burst(const std::vector<std::vector<uint8_t>>& blocks, std::vector<int>& corrected_counts);

    int get_max_correctable_errors() const { return RS_ECC_LENGTH / 2; }

private:
    std::vector<int> gf_exp_;
    std::vector<int> gf_log_;
    std::vector<int> generator_poly_;

    void init_galois_field();
    void init_generator_polynomial();

    int gf_add(int a, int b) const;
    int gf_sub(int a, int b) const;
    int gf_mul(int a, int b) const;
    int gf_div(int a, int b) const;
    int gf_pow(int a, int power) const;
    int gf_inverse(int a) const;

    std::vector<int> poly_mul(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> poly_mod(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> poly_scale(const std::vector<int>& poly, int scalar) const;
    std::vector<int> poly_add(const std::vector<int>& a, const std::vector<int>& b) const;

    int poly_eval(const std::vector<int>& poly, int x) const;
    std::vector<int> find_error_locator(const std::vector<int>& syndromes) const;
    std::vector<int> find_error_positions(const std::vector<int>& error_locator) const;
    std::vector<int> correct_errors(const std::vector<int>& received,
                                     const std::vector<int>& syndromes,
                                     const std::vector<int>& error_positions) const;
};

}
