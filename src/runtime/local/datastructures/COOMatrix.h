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

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <iomanip>

#include <cassert>
#include <cstddef>
#include <cstring>

/**
 * @brief A sparse matrix in COOrdinate (COO) format.
 *
 */
template<typename ValueType>
class COOMatrix : public Matrix<ValueType> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;

    /**
     * @brief The maximum number of non-zero values this matrix was allocated
     * to accommodate.
     */
    size_t maxNumNonZeros;
    /**
     * @brief The  number of non-zero values this matrix accommodates.
     */
    size_t numNonZeros;

    /**
     * @brief For row-based sub-matrix views, we need this for printing.
     */
     size_t rowOffset;

    std::shared_ptr<ValueType> values;
    std::shared_ptr<size_t> colIdxs;
    std::shared_ptr<size_t> rowIdxs;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType *DataObjectFactory::create(ArgTypes ...);

    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType *obj);

    /**
     * @brief Creates a `COOMatrix` and allocates enough memory for the
     * specified size in the internal `values`, `colIdxs`, and `rowIdxs`
     * arrays.
     * 
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param maxNumNonZeros The maximum number of non-zeros in the matrix.
     * @param zero Whether the allocated memory of the internal arrays shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    COOMatrix(size_t numRows, size_t numCols, size_t maxNumNonZeros, bool zero) :
            Matrix<ValueType>(numRows, numCols),
            maxNumNonZeros(maxNumNonZeros),
            numNonZeros(0),
            rowOffset(0),
            values(new ValueType[maxNumNonZeros], std::default_delete<ValueType[]>()),
            colIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()),
            rowIdxs(new size_t[maxNumNonZeros], std::default_delete<size_t[]>()) {
        if (zero) {
            memset(values.get(), 0, maxNumNonZeros * sizeof(ValueType));
            memset(colIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
            memset(rowIdxs.get(), 0, maxNumNonZeros * sizeof(size_t));
        }
    }

    /**
     * @brief Creates a `COOMatrix` around a sub-matrix of another `COOMatrix`
     * without copying the data.
     *
     * @param src The other `COOMatrix`.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to extract.
     */
    COOMatrix(const COOMatrix<ValueType> *src, size_t rowLowerIncl, size_t rowUpperExcl) :
            Matrix<ValueType>(rowUpperExcl - rowLowerIncl, src->numCols),
            maxNumNonZeros(std::min(src->maxNumNonZeros, src->numCols * (rowUpperExcl - rowLowerIncl))),
            rowOffset(rowLowerIncl) {
        assert(src && "src must not be null");
        assert((rowLowerIncl < src->numRows) && "rowLowerIncl is out of bounds");
        assert((rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds");
        assert((rowLowerIncl < rowUpperExcl) && "rowLowerIncl must be lower than rowUpperExcl");

        std::pair<size_t, size_t> range = src->rowRange(rowLowerIncl, 0);
        size_t rowStart = range.first;
        size_t rowLength = range.second;

        rowIdxs = std::shared_ptr<size_t>(src->rowIdxs, src->rowIdxs.get() + rowStart);
        colIdxs = std::shared_ptr<size_t>(src->colIdxs, src->colIdxs.get() + rowStart);
        values = std::shared_ptr<ValueType>(src->values, src->values.get() + rowStart);

        size_t nonZeros = rowLength;

        for (size_t i = rowLowerIncl + 1; i < rowUpperExcl; i++) {
            range = src->rowRange(i, rowStart + rowLength);
            rowStart = range.first;
            rowLength = range.second;
            nonZeros += rowLength;
        }

        numNonZeros = nonZeros;
    }

    virtual ~COOMatrix() {
        // nothing to do
    }

    [[nodiscard]] std::pair<size_t, size_t> rowRange(size_t rowIdx, size_t start) const {
        size_t rowStart = 0, rowLength = 0, row;
        for (size_t i = start; i < numNonZeros; i++) {
            row = rowIdxs.get()[i];
            if (row > rowIdx) {
                if (rowLength == 0) rowStart = i;
                return std::make_pair(rowStart, rowLength);
            }
            if (row == rowIdx) {
                if (rowLength == 0) rowStart = i;
                rowLength++;
            }
        }

        if (rowLength == 0) return std::make_pair(numNonZeros, rowLength);
        return std::make_pair(rowStart, rowLength);
    }

    void insert(size_t pos, size_t rowIdx, size_t colIdx, ValueType value) {
        assert((numNonZeros < maxNumNonZeros) && "can't add any more nonzero values");

        if (value == ValueType(0)) return;
        for (size_t i = numNonZeros; i > pos; i--) {
            rowIdxs.get()[i] = rowIdxs.get()[i - 1];
            values.get()[i] = values.get()[i - 1];
            colIdxs.get()[i] = colIdxs.get()[i - 1];
        }
        rowIdxs.get()[pos] = rowIdx;
        values.get()[pos] = value;
        colIdxs.get()[pos] = colIdx;
        numNonZeros++;
    }

    void remove(size_t idx) {
        for (size_t i = idx + 1; i < numNonZeros; i++) {
            rowIdxs.get()[i - 1] = rowIdxs.get()[i];
            values.get()[i - 1] = values.get()[i];
            colIdxs.get()[i - 1] = colIdxs.get()[i];
        }
        numNonZeros--;
    }

    [[nodiscard]] std::pair<size_t, bool> findIndex(size_t row, size_t col, size_t start) const {
        size_t i;
        for (i = start; i < numNonZeros; ++i) {
            size_t currentRow = rowIdxs.get()[i];
            size_t currentCol = colIdxs.get()[i];
            if (currentRow > row) return std::make_pair(i, false);
            if (currentRow == row && currentCol > col) return std::make_pair(i, false);
            if (currentRow == row && currentCol == col) return std::make_pair(i, true);
        }
        return std::make_pair(i, false);
    }

public:
    [[nodiscard]] size_t getMaxNumNonZeros() const {
        return maxNumNonZeros;
    }

    [[nodiscard]] size_t getNumNonZeros() const {
        return numNonZeros;
    }

    [[nodiscard]] size_t getNumNonZerosRow(size_t rowIdx) const {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        return range.second;
    }

    [[nodiscard]] size_t getNumNonZerosCol(size_t colIdx) const {
        assert((colIdx < numCols) && "colIdx is out of bounds");

        size_t cnt = 0;
        for (size_t i = 0; i < numNonZeros; i++) {
            if (colIdxs.get()[i] == colIdx) cnt++;
        }
        return cnt;
    }

    void incrNumNonZeros(size_t val) {
        numNonZeros += val;
    }

    [[nodiscard]] ValueType *getValues() {
        return values.get();
    }

    [[nodiscard]] const ValueType *getValues() const {
        return values.get();
    }

    [[nodiscard]] size_t *getColIdxs() {
        return colIdxs.get();
    }

    [[nodiscard]] const size_t *getColIdxs() const {
        return colIdxs.get();
    }

    [[nodiscard]] size_t *getRowIdxs() {
        return rowIdxs.get();
    }

    [[nodiscard]] const size_t *getRowIdxs() const {
        return rowIdxs.get();
    }

    [[nodiscard]] ValueType *getValues(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        size_t rowStart = range.first;

        return values.get() + rowStart;
    }

    [[nodiscard]] const ValueType *getValues(size_t rowIdx) const {
        return const_cast<COOMatrix<ValueType> *>(this)->getValues(rowIdx);
    }

    [[nodiscard]] size_t *getColIdxs(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        size_t rowStart = range.first;

        return colIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getColIdxs(size_t rowIdx) const {
        return const_cast<COOMatrix<ValueType> *>(this)->getColIdxs(rowIdx);
    }

    [[nodiscard]] size_t *getRowIdxs(size_t rowIdx) {
        assert((rowIdx <= numRows) && "rowIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        size_t rowStart = range.first;

        return rowIdxs.get() + rowStart;
    }

    [[nodiscard]] const size_t *getRowIdxs(size_t rowIdx) const {
        return const_cast<COOMatrix *>(this)->getRowIdxs(rowIdx);
    }

    ValueType get(size_t rowIdx, size_t colIdx) const override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        size_t rowStart = range.first;
        size_t rowLength = range.second;

        if (rowLength == 0) {
            return ValueType(0);
        } else {
            for (size_t i = rowStart; i < rowStart + rowLength; i++) {
                size_t col = colIdxs.get()[i];
                if (col > colIdx) return ValueType(0);
                if (col == colIdx) return values.get()[i];
            }
            return ValueType(0);
        }
    }

    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");

        std::pair<size_t, size_t> range = rowRange(rowIdx, 0);
        size_t rowStart = range.first;
        size_t rowLength = range.second;
        size_t rowEnd = rowStart + rowLength;

        if (rowLength == 0) {
            insert(rowEnd, rowIdx, colIdx, value);
            return;
        } else {
            for (size_t i = rowStart; i < rowEnd; i++) {
                if (colIdxs.get()[i] > colIdx) {
                    insert(i, rowIdx, colIdx, value);
                    return;
                }
                if (colIdx == colIdxs.get()[i]) {
                    if (value == ValueType(0)) {
                        remove(i);
                        return;
                    } else {
                        values.get()[i] = value;
                        return;
                    }
                }
            }
            insert(rowEnd, rowIdx, colIdx, value);
            return;
        }
    }

    void prepareAppend() override {
        numNonZeros = 0;
    }

    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        assert((rowIdx < numRows) && "rowIdx is out of bounds");
        assert((colIdx < numCols) && "colIdx is out of bounds");
        assert((numNonZeros < maxNumNonZeros) && "can't add any more nonzero values");

        if (value == ValueType(0)) return;

        rowIdxs.get()[numNonZeros] = rowIdx;
        values.get()[numNonZeros] = value;
        colIdxs.get()[numNonZeros] = colIdx;
        numNonZeros++;
    }

    void finishAppend() override {
        // do nothing
    }

    void printValue(std::ostream &os, ValueType val) const {
        switch (ValueTypeUtils::codeFor<ValueType>) {
            case ValueTypeCode::SI8 :
                os << static_cast<int32_t>(val);
                break;
            case ValueTypeCode::UI8 :
                os << static_cast<uint32_t>(val);
                break;
            default :
                os << val;
                break;
        }
    }

    /**
     * @brief Pretty print of this matrix.
     * @param os The stream to print to.
     */
    void print(std::ostream &os) const override {
        os << "COOMatrix(" << numRows << 'x' << numCols << ", "
           << "double" << ')' << std::endl << std::endl;

        auto *colWidths = new int[numCols];
        for (size_t i = 0; i < numCols; ++i) {
            colWidths[i] = 1;
        }

        for (size_t i = 0; i < numNonZeros; ++i) {
            ValueType value = values.get()[i];
            std::ostringstream oss;
            oss << value;
            std::string strValue = oss.str();
            colWidths[colIdxs.get()[i]] = std::max(static_cast<int>(strValue.length()), colWidths[colIdxs.get()[i]]);
        }

        size_t start = 0;
        for (size_t row = 0; row < numRows; ++row) {
            for (size_t col = 0; col < numCols; ++col) {
                std::pair<size_t, size_t> searchRes = findIndex(row + rowOffset, col, start);
                start = searchRes.first;
                bool found = searchRes.second;
                if (found) {
                    os << std::setw(colWidths[col]) << values.get()[start] << " ";
                } else {
                    os << std::setw(colWidths[col]) << 0 << " ";
                }
            }
            os << std::endl;
        }

        delete[] colWidths;
    }

    /**
     * @brief Prints the internal arrays of this matrix.
     * @param os The stream to print to.
     */
    void printRaw(std::ostream &os) const {
        os << "COOMatrix(" << numRows << 'x' << numCols << ", "
           << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;
        os << "maxNumNonZeros: \t" << maxNumNonZeros << std::endl;
        os << "numNonZeros: \t" << numNonZeros << std::endl;
        os << "values: \t";
        for (size_t i = 0; i < numNonZeros; i++)
            os << values.get()[i] << ", ";
        os << std::endl;
        os << "colIdxs: \t";
        for (size_t i = 0; i < numNonZeros; i++)
            os << colIdxs.get()[i] << ", ";
        os << std::endl;
        os << "rowIdxs: \t";
        for (size_t i = 0; i < numNonZeros; i++)
            os << rowIdxs.get()[i] << ", ";
        os << std::endl;
    }

    COOMatrix *sliceRow(size_t rl, size_t ru) const override {
        return DataObjectFactory::create<COOMatrix>(this, rl, ru);
    }

    COOMatrix *sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("COOMatrix does not support column-based slicing yet");
    }

    COOMatrix *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("COOMatrix does not support slicing yet");
    }

    size_t bufferSize() {
        return this->getNumItems() * sizeof(ValueType);
    }

    size_t serialize(std::vector<char> &buf) const override;
};

template<typename ValueType>
std::ostream &operator<<(std::ostream &os, const COOMatrix<ValueType> &obj) {
    obj.print(os);
    return os;
}
