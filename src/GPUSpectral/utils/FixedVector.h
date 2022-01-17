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

    template <typename It>
    FixedVector(It it, It end) : size_(std::distance(it, end) * sizeof(T)), data_(new T[size_]){
        std::copy(it, end, data_);
    }

    FixedVector &operator=(const FixedVector &other) {
        size_ = other.size_;
        data_ = new T[size_];
        memcpy(data_, other.data_, size_ * sizeof(T));
        return *this;
    }

    FixedVector(FixedVector &&other) : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    FixedVector &operator=(FixedVector &&other) {
        delete[] data_;

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    size_t size() const {
        return size_;
    }

    T *data() const {
        return data_;
    }

    T &operator[](uint32_t index) {
        return data_[index];
    }

    const T &operator[](uint32_t index) const {
        return data_[index];
    }

  private:
    size_t size_{};
    T *data_{};
};
