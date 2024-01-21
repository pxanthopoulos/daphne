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

template <typename T>
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

TEMPLATE_PRODUCT_TEST_CASE("create log dir", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::string directoryPath = "prof_logs";
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(directoryPath))
            fs::create_directory(directoryPath);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_random file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/logs_random.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile random", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/logs_random.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;

    for (double sparsity: {0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT *m = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();
            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: Random, Class: " << getClassName<DT>() << ", Size: " << numRows << "x" << numCols << ", Sparsity: " << sparsity << ", Valuetype: "
                       << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs"
                       << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_transpose file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_transpose.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile transpose", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_transpose.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 500;
    const size_t numCols = 500;
    const VT min = 100;
    const VT max = 200;

    for(double sparsity : {0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT * m = nullptr;
            DT * res = nullptr;

            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);
            auto start_time = std::chrono::high_resolution_clock::now();
            transpose<DT, DT>(res, m, nullptr);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: Transpose, Class: " << getClassName<DT>() << ", Size: " << numRows << "x" << numCols << ", Sparsity: " << sparsity << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_gengivenvals file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_gengivenvals.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile gengivenvals", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_gengivenvals.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(distrVal(gen));
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            DT * m = genGivenVals<DT>(10, result);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: GenGivenVals, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_getsetappend file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_getsetappend.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile get/set/append/view", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_getsetappend.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;

    const VT min = 0;
    const VT max = 10;
    const size_t size = 250000;
    const size_t numRows = 500;
    const size_t numCols = size / numRows;
    const size_t count = 250000;
    size_t rowLowerIncl = 100;
    size_t rowUpperExcl = 400;

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);

    static_assert(
            std::is_floating_point<VT>::value || std::is_integral<VT>::value,
            "the value type must be either floating point or integral"
    );
    typename std::conditional<
            std::is_floating_point<VT>::value,
            std::uniform_real_distribution<VT>,
            std::uniform_int_distribution<VT>
    >::type distrVal(min, max);

    typename std::conditional<
            std::is_floating_point<VT>::value,
            std::uniform_real_distribution<VT>,
            std::uniform_int_distribution<VT>
    >::type distrValNZ(min + 1, max);

    std::uniform_int_distribution<size_t> distrRow(0, numRows - 1);
    std::uniform_int_distribution<size_t> distrRowView(0, rowUpperExcl - rowLowerIncl - 1);
    std::uniform_int_distribution<size_t> distrCol(0, numCols - 1);

    for(char type : {'g', 's', 'a', 'v'}) {
        DYNAMIC_SECTION("type = " << type) {
            if (type == 'g') {
                std::vector<VT> result;
                result.reserve(size);
                for (size_t i = 0; i < size; ++i) {
                    result.push_back(distrValNZ(gen));
                }
                DT * m = genGivenVals<DT>(numRows, result);

                auto start_time = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < count; ++i) {
                    [[maybe_unused]] VT res = m->get(distrRow(gen), distrCol(gen));
                }
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                outputFile << "Test: Get, Class: " << getClassName<DT>() << ", Size: " << size << ", Get count: " << count << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;
            }
            else if (type == 's') {
                DT * m;

                if constexpr (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
                    m = DataObjectFactory::create<DT>(numRows, numCols, true);
                } else {
                    m = DataObjectFactory::create<DT>(numRows, numCols, size, true);
                }

                auto start_time = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < count; ++i) {
                    m->set(distrRow(gen), distrCol(gen), distrVal(gen));
                }
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                outputFile << "Test: Set, Class: " << getClassName<DT>() << ", Size: " << size << ", Set count: " << count << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;
            }
            else if (type == 'a') {
                std::vector<size_t> rowSequence;
                std::vector<size_t> occurrences(numRows, 0);
                for (size_t i = 0; i < size; ++i) {
                    size_t randomValue = distrRow(gen);

                    while (occurrences[randomValue] >= numCols) {
                        randomValue = distrRow(gen);
                    }

                    occurrences[randomValue]++;
                    rowSequence.push_back(randomValue);
                }
                std::sort(rowSequence.begin(), rowSequence.end());

                std::vector<size_t> colSequence;

                std::vector<size_t> startRow;
                size_t lastRow = -1;
                for (size_t i = 0; i < size; ++i) {
                    if (rowSequence[i] != lastRow) startRow.push_back(i);
                    lastRow = rowSequence[i];
                }
                startRow.push_back(size);

                for (size_t i = 0; i < startRow.size() - 1; i++) {
                    size_t start = startRow[i];
                    size_t end = startRow[i + 1];
                    std::vector<size_t> subSequence;
                    std::unordered_set<size_t> uniqueValues;
                    while (subSequence.size() < end - start) {
                        size_t randomValue = distrCol(gen);
                        if (uniqueValues.find(randomValue) == uniqueValues.end()) {
                            subSequence.push_back(randomValue);
                            uniqueValues.insert(randomValue);
                        }
                    }

                    std::sort(subSequence.begin(), subSequence.end());
                    for (size_t value : subSequence) {
                        colSequence.push_back(value);
                    }
                }

                DT * m;

                if constexpr (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
                    m = DataObjectFactory::create<DT>(numRows, numCols, true);
                } else {
                    m = DataObjectFactory::create<DT>(numRows, numCols, size, true);
                }

                auto start_time = std::chrono::high_resolution_clock::now();
                m->prepareAppend();
                for (size_t i = 0; i < size; ++i) {
                    m->append(rowSequence[i], colSequence[i], distrValNZ(gen));
                }
                m->finishAppend();
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                outputFile << "Test: Append, Class: " << getClassName<DT>() << ", Size: " << size << ", Append count: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;
            }
            else if (type == 'v') {
                DT * m = nullptr;

                randMatrix<DT, VT>(m, numRows, numCols, min, max, 1.0, -1, nullptr);

                DT * view;
                if constexpr (std::is_same<DT, DenseMatrix<typename DT::VT>>::value) {
                    view = DataObjectFactory::create<DT>(m, rowLowerIncl, rowUpperExcl, 0, numCols);
                } else {
                    view = DataObjectFactory::create<DT>(m, rowLowerIncl, rowUpperExcl);
                }

                auto start_time = std::chrono::high_resolution_clock::now();

                bool set = true;
                for (size_t i = 0; i < size; ++i) {
                    if (set) {
                        view->set(distrRowView(gen), distrCol(gen), distrVal(gen));
                    }
                    else {
                        [[maybe_unused]] VT res = view->get(distrRowView(gen), distrCol(gen));
                    }
                    set = !set;
                }

                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

                start_time = std::chrono::high_resolution_clock::now();

                set = true;
                for (size_t i = 0; i < size; ++i) {
                    if (set) {
                        m->set(distrRowView(gen), distrCol(gen), distrVal(gen));
                    }
                    else {
                        [[maybe_unused]] VT res = m->get(distrRowView(gen), distrCol(gen));
                    }
                    set = !set;
                }

                end_time = std::chrono::high_resolution_clock::now();

                double diff = ((double)duration.count() / (double)(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count()) - 1;

                outputFile << "Test: View, Class: " << getClassName<DT>() << ", Size: " << size << ", Count: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Difference from original to view: " << diff * 100 << "%" << std::endl;

                DataObjectFactory::destroy(m);
                DataObjectFactory::destroy(view);
            }
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggall file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggall.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggall", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggall.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(distrVal(gen));
            }

            DT * m1 = genGivenVals<DT>(10, result);

            auto start_time = std::chrono::high_resolution_clock::now();

            [[maybe_unused]] VT res = aggAll<VT, DT>(AggOpCode::SUM, m1, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggAll, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggcol file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggcol.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggcol", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggcol.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(distrVal(gen));
            }

            DT * m1 = genGivenVals<DT>(10, result);
            DTRes * res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            aggCol<DTRes, DT>(AggOpCode::SUM, res, m1, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggCol, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_aggrow file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_aggrow.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile aggrow", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_aggrow.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result;
            result.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result.push_back(distrVal(gen));
            }

            DT * m1 = genGivenVals<DT>(10, result);
            DTRes * res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            aggRow<DTRes, DT>(AggOpCode::SUM, res, m1, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggRow, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_ewbinary file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_ewbinary.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile ewbinary", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_ewbinary.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result1;
            result1.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result1.push_back(distrVal(gen));
            }
            DT * m1 = genGivenVals<DT>(10, result1);

            std::vector<VT> result2;
            result2.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result2.push_back(distrVal(gen));
            }
            DT * m2 = genGivenVals<DT>(10, result2);

            DT * res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            ewBinaryMat<DT, DT, DT>(BinaryOpCode::ADD, res, m1, m2, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggRow, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("empty log_ewunary file", TAG_DATASTRUCTURES, (DenseMatrix), (double)) {
    std::ofstream outputFile("prof_logs/log_ewunary.txt", std::ios::trunc);
    CHECK((outputFile.is_open()));
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}

TEMPLATE_PRODUCT_TEST_CASE("profile ewunary", TAG_DATASTRUCTURES, (DenseMatrix, COOMatrix), (double, uint8_t)) {
    std::ofstream outputFile("prof_logs/log_ewunary.txt", std::ios::app);
    CHECK(outputFile.is_open());

    using DT = TestType;
    using VT = typename DT::VT;
    const VT min = 100;
    const VT max = 200;
    using DTRes = DenseMatrix<VT>;

    for(size_t size : {500, 1000, 10000, 100000, 1000000}) {
        DYNAMIC_SECTION("size = " << size) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            static_assert(
                    std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                    "the value type must be either floating point or integral"
            );
            typename std::conditional<
                    std::is_floating_point<VT>::value,
                    std::uniform_real_distribution<VT>,
                    std::uniform_int_distribution<VT>
            >::type distrVal(min, max);
            std::vector<VT> result1;
            result1.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                result1.push_back(distrVal(gen));
            }
            DT * m1 = genGivenVals<DT>(10, result1);

            DTRes * res = nullptr;

            auto start_time = std::chrono::high_resolution_clock::now();

            ewUnaryMat<DTRes, DT>(UnaryOpCode::SQRT, res, m1, nullptr);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            outputFile << "Test: AggRow, Class: " << getClassName<DT>() << ", Size: " << size << ", Valuetype: " << ValueTypeUtils::cppNameFor<VT> << ", Execution time: " << duration.count() << "μs" << std::endl;

            DataObjectFactory::destroy(m1);
        }
    }
    outputFile.close();
    CHECK(!(outputFile.is_open()));
}
