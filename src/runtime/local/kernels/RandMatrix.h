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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H
#define SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/COOMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <random>
#include <set>
#include <type_traits>
#include <vector>
#include <unordered_set>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct RandMatrix {
    static void apply(DTRes *& res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void randMatrix(DTRes *& res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed, DCTX(ctx)) {
    RandMatrix<DTRes, VTArg>::apply(res, numRows, numCols, min, max, sparsity, seed, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RandMatrix<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, size_t numRows, size_t numCols, VT min, VT max, double sparsity, int64_t seed, DCTX(ctx)) {
        assert(numRows > 0 && "numRows must be > 0");
        assert(numCols > 0 && "numCols must be > 0");
        assert(min <= max && "min must be <= max");
        assert((min != 0 || max != 0) &&
               "min and max must not both be zero, consider setting sparsity to zero instead");
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        if (seed == -1) {
            std::random_device rd;
            std::uniform_int_distribution<int64_t> seedRnd;
            seed = seedRnd(rd);
        }

        std::mt19937 genVal(seed);
        std::mt19937 genIndex(seed * 3);
        
        static_assert(
                std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                "the value type must be either floating point or integral"
        );
        typename std::conditional<
                std::is_floating_point<VT>::value,
                std::uniform_real_distribution<VT>,
                std::uniform_int_distribution<VT>
        >::type distrVal(min, max);
        std::uniform_int_distribution<int> distrIndex(0, numCols * numRows - 1);

        VT * valuesRes = res->getValues();

        // If sparsity >= 0.5, we initialize with random values and insert zeros,
        // else if sparsity < 0.5, it is more efficient to initialize with zero values and insert random.
        size_t insertedValuesLimit;
        if (sparsity >= 0.5) {
            insertedValuesLimit = size_t(round((1 - sparsity) * numCols * numRows));                    
        } else {
            insertedValuesLimit = size_t(round(sparsity * numCols * numRows));
        }
        
        // Fill Matrix with non-zero/random values
        // TODO It might be faster to pull the check on sparsity out of the
        // loop, including a duplication of the loop.
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                if (sparsity >= 0.5) {
                    valuesRes[c] = distrVal(genVal);
                    while (valuesRes[c] == 0)
                        valuesRes[c] = distrVal(genVal);
                } else {
                    valuesRes[c] = VT(0);
                }
            }
            valuesRes += res->getRowSkip();
        }

        // Use Knuth's algorithm to calculate unique random indexes equal to insertedValuesLimit, to be set to zero/random value.
        valuesRes = res->getValues();
        size_t iRange, iSize;
        iSize = 0;
        // TODO It might be faster to pull the check on sparsity out of the
        // loop, including a duplication of the loop.
        // TODO If res->getRowSkip() == res->getNumCols(), it might be faster
        // not to calculate row and col by / and %, but to directly use the
        // generated index.
        for (iRange = 0; iRange < (numCols * numRows) && iSize < insertedValuesLimit; iRange++) {            
            size_t rRange = (numCols * numRows) - iRange;
            size_t rSize = insertedValuesLimit - iSize;
            if (fmod(distrIndex(genIndex), rRange) < rSize) {
                size_t row = iRange / numCols;
                size_t col = iRange % numCols; 
                if (sparsity >= 0.5) {
                    valuesRes[row * res->getRowSkip() + col] = VT(0);
                } else {
                    valuesRes[row * res->getRowSkip() + col] = distrVal(genVal);
                    while (valuesRes[row * res->getRowSkip() + col] == 0)
                        valuesRes[row * res->getRowSkip() + col] = distrVal(genVal);
                }
                iSize++;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RandMatrix<CSRMatrix<VT>, VT> {
    static void apply(CSRMatrix<VT> *& res, size_t numRows, size_t numCols, VT min, VT max, double sparsity, int64_t seed, DCTX(ctx)) {
        assert(numRows > 0 && "numRows must be > 0");
        assert(numCols > 0 && "numCols must be > 0");
        assert(min <= max && "min must be <= max");
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");

        // The exact number of non-zeros to generate.
        // TODO Ideally, it should not be allowed that zero is included in [min, max].
        const size_t nnz = static_cast<size_t>(round(numRows * numCols * sparsity));
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, nnz, false);

        // Initialize pseudo random number generators.
        if (seed == -1)
            seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
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
        
        std::uniform_int_distribution<size_t> distrRow(0, numRows - 1);
        std::uniform_int_distribution<size_t> distrCol(0, numCols - 1);
        
        // Generate non-zero values (positions in the matrix do not matter here).
        VT * valuesRes = res->getValues();
        for(size_t i = 0; i < nnz; i++)
            valuesRes[i] = distrVal(gen);
        
        // Randomly determine the number of non-zeros per row. Store them in
        // the result matrix's rowOffsets array to avoid an additional
        // allocation and to make the prefix sum more cache-efficient.
        size_t * rowOffsetsRes = res->getRowOffsets();
        // We need signed ssize_t for the >0 check.
        ssize_t * nnzPerRow = reinterpret_cast<ssize_t *>(rowOffsetsRes + 1);
        if(sparsity <= 0.5) {
            // Start with empty rows, increment nnz of random row until the
            // desired total number of non-zeros is reached.
            std::fill_n(nnzPerRow, numRows, 0);
            size_t assigned = 0;
            while(assigned < nnz) {
                const size_t r = distrRow(gen);
                if(nnzPerRow[r] < static_cast<ssize_t>(numCols)) {
                    nnzPerRow[r]++;
                    assigned++;
                }
            }
        }
        else {
            // Start with full rows, decrement nnz of random row until the
            // desired total number of non-zeros is reached.
            std::fill_n(nnzPerRow, numRows, numCols);
            size_t assigned = numRows * numCols;
            while(assigned > nnz) {
                const size_t r = distrRow(gen);
                if(nnzPerRow[r] > 0) {
                    nnzPerRow[r]--;
                    assigned--;
                }
            }
        }
        
        // Generate random column indexes, sorted within each row.
        size_t * colIdxsRes = res->getColIdxs();
        if(sparsity <= 0.5) {
            // Use the generated column indexes.
            for(size_t r = 0; r < numRows; r++) {
                std::set<size_t> sortedColIdxs;
                while(static_cast<ssize_t>(sortedColIdxs.size()) < nnzPerRow[r])
                    sortedColIdxs.emplace(distrCol(gen));
                for(auto it = sortedColIdxs.begin(); it != sortedColIdxs.end(); it++)
                    *colIdxsRes++ = *it;
            }
        }
        else {
            // Use all but the generated column indexes.
            for(size_t r = 0; r < numRows; r++) {
                std::set<size_t> sortedColIdxs;
                while(sortedColIdxs.size() < numCols - nnzPerRow[r])
                    sortedColIdxs.emplace(distrCol(gen));
                for(size_t c = 0; c < numCols; c++)
                    if(!sortedColIdxs.count(c))
                        *colIdxsRes++ = c;
            }
        }
        
        // Calculate the row offsets as the prefix sum over the nnz per row.
        rowOffsetsRes[0] = 0;
        for(size_t i = 1; i <= numRows; i++)
            rowOffsetsRes[i] += rowOffsetsRes[i - 1];
    }
};

// ----------------------------------------------------------------------------
// COOMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RandMatrix<COOMatrix<VT>, VT> {
    static void apply(COOMatrix<VT> *& res, size_t numRows, size_t numCols, VT min, VT max, double sparsity, int64_t seed, DCTX(ctx)) {
        assert(numRows > 0 && "numRows must be > 0");
        assert(numCols > 0 && "numCols must be > 0");
        assert(min <= max && "min must be <= max");
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");

        // The exact number of non-zeros to generate.
        // TODO Ideally, it should not be allowed that zero is included in [min, max].
        const auto nnz = static_cast<size_t>(round(numRows * numCols * sparsity));

        if (res == nullptr)
            res = DataObjectFactory::create<COOMatrix<VT>>(numRows, numCols, nnz, false);

        // Initialize pseudo random number generators.
        if (seed == -1)
            seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
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

        VT *valuesRes = res->getValues();
        for (size_t i = 0; i < nnz; i++)
            valuesRes[i] = distrVal(gen);

        std::uniform_int_distribution<size_t> distrRow(0, numRows - 1);
        std::vector<size_t> rowSequence;
        std::vector<size_t> occurrences(numRows, 0);
        for (size_t i = 0; i < nnz; ++i) {
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
        for (size_t i = 0; i < nnz; ++i) {
            if (rowSequence[i] != lastRow) startRow.push_back(i);
            lastRow = rowSequence[i];
        }
        startRow.push_back(nnz);

        std::uniform_int_distribution<size_t> distrCol(0, numCols - 1);

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

        size_t * rowIdxs = res->getRowIdxs();
        size_t * colIdxs = res->getColIdxs();
        for (size_t i = 0; i < nnz; i++) {
            rowIdxs[i] = rowSequence[i];
            colIdxs[i] = colSequence[i];
        }

        valuesRes[nnz] = VT(0);
        rowIdxs[nnz] = size_t(-1);
        colIdxs[nnz] = size_t(-1);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H