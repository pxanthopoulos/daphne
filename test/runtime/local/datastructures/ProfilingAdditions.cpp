/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/COOMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/AggCol.h>
#include <runtime/local/kernels/AggRow.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/Transpose.h>

#include <tags.h>

#include <catch.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <type_traits>
#include <filesystem>

template<typename T>
std::string getClassName() {
    if constexpr (std::is_same<T, COOMatrix<typename T::VT>>::value) {
        return "COOMatrix";
    } else if constexpr (std::is_same<T, CSRMatrix<typename T::VT>>::value) {
        return "CSRMatrix";
    } else if constexpr (std::is_same<T, DenseMatrix<typename T::VT>>::value) {
        return "DenseMatrix";
    } else {
        return "Unknown";
    }
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggall_sparsity file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggall_sparsity.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggall_sparsity", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix),
                           (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggall_sparsity.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;

    for (double sparsity: {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m = nullptr;

            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);

            auto start_time = std::chrono::high_resolution_clock::now();

            [[maybe_unused]] VT res = aggAll<VT, DT>(AggOpCode::SUM, m, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggAll, Class: " << getClassName<DT>() << ", Sparsity: " << sparsity << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggcol_sparsity file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggcol_sparsity.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggcol_sparsity", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix),
                           (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggcol_sparsity.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for (double sparsity: {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m = nullptr;

            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);

            DTRes *res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            aggCol<DTRes, DT>(AggOpCode::SUM, res, m, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggCol, Class: " << getClassName<DT>() << ", Sparsity: " << sparsity << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggrow_sparsity file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggrow_sparsity.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggrow_sparsity", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix),
                           (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggrow_sparsity.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for (double sparsity: {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m = nullptr;

            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);

            DTRes *res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            aggRow<DTRes, DT>(AggOpCode::SUM, res, m, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggRow, Class: " << getClassName<DT>() << ", Sparsity: " << sparsity << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_ewbinary_sparsity file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_ewbinary_sparsity.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile ewbinary_sparsity", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix),
                           (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_ewbinary_sparsity.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;

    for (double sparsity: {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m1 = nullptr;
            randMatrix<DT, VT>(m1, numRows, numCols, min, max, sparsity, -1, nullptr);

            DT *m2 = nullptr;
            randMatrix<DT, VT>(m2, numRows, numCols, min, max, sparsity, -1, nullptr);

            DT *res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: EwBinary, Class: " << getClassName<DT>() << ", Sparsity: " << sparsity
                       << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_ewunary_sparsity file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_ewunary_sparsity.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile ewunary_sparsity", TAG_DATASTRUCTURES, (DenseMatrix, COOMatrix),
                           (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_ewunary_sparsity.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for (double sparsity: {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m1 = nullptr;
            randMatrix<DT, VT>(m1, numRows, numCols, min, max, sparsity, -1, nullptr);

            DTRes *res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            ewUnaryMat<DTRes, DT>(UnaryOpCode::SQRT, res, m1, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: EwUnary, Class: " << getClassName<DT>() << ", Sparsity: " << sparsity
                       << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}