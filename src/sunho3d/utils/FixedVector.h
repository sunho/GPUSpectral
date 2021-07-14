#pragma once

#include <algorithm>
#include <cstdlib>

template <typename T>
struct FixedVector {
    FixedVector() = default;
    explicit FixedVector(size_t size)
        : size_(size), data_(new T[size]) {
    }
    ~FixedVector() {
        delete[] data_;
    }

    FixedVector(const FixedVector &other)
        : size_(other.size_), data_(new T[other.size_]) {
        memcpy(data_, other.data_, size_ * sizeof(T));
    }
    FixedVector &operator=(const FixedVector &other) {
        size_ = other.size_;
        data_ = new T[size_];
        memcpy(data_, other.data_, size_ * sizeof(T));
        return *this;
    }

    FixedVector(FixedVector &&other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    FixedVector &operator=(FixedVector &&other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
    }

    size_t size() const {
        return size_;
    }
    T *data() const {
        return data_;
    }

  private:
    size_t size_{};
    T *data_{};
};
