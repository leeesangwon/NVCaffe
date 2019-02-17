#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* const_bottom_data = bottom[0]->gpu_data<Ftype>();
  const Ftype* const_top_data = top[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();

  const Ftype* scale_data = this->blobs_[0]->template gpu_data<Ftype>();
  const Ftype* shift_data = this->blobs_[1]->template gpu_data<Ftype>();

  // Mean normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average mean
    caffe_copy(batch_statistic_->count(), this->blobs_[2]->template gpu_data<Ftype>(),
        batch_statistic_->template mutable_gpu_data<Ftype>());
  } else {
    // Compute the mean by averaging over spatial and batch dimensions.
    caffe_gpu_gemv<Ftype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Ftype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_->template gpu_data<Ftype>(), Ftype(0),
        spatial_statistic_->template mutable_gpu_data<Ftype>());
    caffe_gpu_gemv<Ftype>(CblasTrans, num_, channels_,
        Ftype(1) / num_, spatial_statistic_->template gpu_data<Ftype>(),
        batch_sum_multiplier_->template gpu_data<Ftype>(), Ftype(0),
        batch_statistic_->template mutable_gpu_data<Ftype>());
    // Add to the moving average
    if (!frozen_) {
      caffe_gpu_axpby(batch_statistic_->count(),
          Ftype(1) - bn_momentum_, batch_statistic_->template gpu_data<Ftype>(),
          bn_momentum_, this->blobs_[2]->template mutable_gpu_data<Ftype>());
    }
  }
  // Broadcast the mean vector
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template gpu_data<Ftype>(), batch_statistic_->template gpu_data<Ftype>(),
      Ftype(0), spatial_statistic_->template mutable_gpu_data<Ftype>());
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(-1),
      spatial_statistic_->template gpu_data<Ftype>(), spatial_sum_multiplier_->template gpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_gpu_data<Ftype>());
  // Subtract
  caffe_gpu_add(broadcast_buffer_->count(), const_bottom_data,
      broadcast_buffer_->template gpu_data<Ftype>(), top_data);

  // Variance normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average variance
    caffe_copy(batch_statistic_->count(), this->blobs_[3]->template gpu_data<Ftype>(),
        batch_statistic_->template mutable_gpu_data<Ftype>());
  } else {
    caffe_gpu_powx(broadcast_buffer_->count(), const_top_data, Ftype(2),
        broadcast_buffer_->template mutable_gpu_data<Ftype>());
    caffe_gpu_gemv<Ftype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Ftype(1) / (height_ * width_), broadcast_buffer_->template gpu_data<Ftype>(),
        spatial_sum_multiplier_->template gpu_data<Ftype>(), Ftype(0),
        spatial_statistic_->template mutable_gpu_data<Ftype>());
    caffe_gpu_gemv<Ftype>(CblasTrans, num_, channels_, Ftype(1) / num_,
        spatial_statistic_->template gpu_data<Ftype>(), batch_sum_multiplier_->template gpu_data<Ftype>(),
        Ftype(0), batch_statistic_->template mutable_gpu_data<Ftype>());

    // Add to the moving average
    caffe_gpu_axpby(batch_statistic_->count(),
        Ftype(1) - bn_momentum_, batch_statistic_->template gpu_data<Ftype>(),
        bn_momentum_, this->blobs_[3]->template mutable_gpu_data<Ftype>());
  }

  // Add eps
  caffe_gpu_add_scalar(batch_statistic_->count(), bn_eps_,
        batch_statistic_->template mutable_gpu_data<Ftype>());
  // Inverse standard deviation
  caffe_gpu_powx(batch_statistic_->count(), batch_statistic_->template gpu_data<Ftype>(),
        Ftype(-0.5), batch_statistic_->template mutable_gpu_data<Ftype>());
  // Broadcast the inverse std
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template gpu_data<Ftype>(), batch_statistic_->template gpu_data<Ftype>(),
      Ftype(0), spatial_statistic_->template mutable_gpu_data<Ftype>());
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template gpu_data<Ftype>(), spatial_sum_multiplier_->template gpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_gpu_data<Ftype>());
  // Multiply with the inverse std
  caffe_gpu_mul(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template gpu_data<Ftype>(), top_data);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_->count(), const_top_data,
        x_norm_->template mutable_gpu_data<Ftype>());
    caffe_copy(batch_statistic_->count(), batch_statistic_->template gpu_data<Ftype>(),
        x_inv_std_->template mutable_gpu_data<Ftype>());
  }

  // Scale
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template gpu_data<Ftype>(), scale_data,
      Ftype(0), spatial_statistic_->template mutable_gpu_data<Ftype>());
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template gpu_data<Ftype>(), spatial_sum_multiplier_->template gpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_gpu_data<Ftype>());
  caffe_gpu_mul(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template gpu_data<Ftype>(), top_data);

  // Shift
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template gpu_data<Ftype>(), shift_data,
      Ftype(0), spatial_statistic_->template mutable_gpu_data<Ftype>());
  caffe_gpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template gpu_data<Ftype>(), spatial_sum_multiplier_->template gpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_gpu_data<Ftype>());
  caffe_gpu_add(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template gpu_data<Ftype>(), top_data);
}

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
  const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (frozen_) {
    if (propagate_down[0]) {
      const Btype* const_top_diff = top[0]->gpu_diff<Btype>();
      Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
      // Use the moving average variance
      caffe_copy(batch_statistic_->count(), this->blobs_[3]->template gpu_data<Btype>(),
          batch_statistic_->template mutable_gpu_data<Btype>());
      caffe_gpu_add_scalar(batch_statistic_->count(), bn_eps_,
          batch_statistic_->template mutable_gpu_data<Btype>());
      caffe_gpu_powx(batch_statistic_->count(), batch_statistic_->template gpu_data<Btype>(),
          Btype(-0.5), batch_statistic_->template mutable_gpu_data<Btype>());
      // Multiple slope with inverse std
      caffe_gpu_mul(batch_statistic_->count(), this->blobs_[0]->template gpu_data<Btype>(),
          batch_statistic_->template gpu_data<Btype>(), batch_statistic_->template mutable_gpu_data<Btype>());
      // Broadcast
      caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
          Btype(1), batch_sum_multiplier_->template gpu_data<Btype>(), batch_statistic_->template gpu_data<Btype>(),
          Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
      caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
          height_ * width_, 1, Btype(1),
          spatial_statistic_->template gpu_data<Btype>(), spatial_sum_multiplier_->template gpu_data<Btype>(),
          Btype(0), broadcast_buffer_->template mutable_gpu_data<Btype>());
      // Elementwise multiply top grad with (slope / std)
      caffe_gpu_mul(broadcast_buffer_->count(), const_top_diff,
          broadcast_buffer_->template gpu_data<Btype>(), bottom_diff);
    }
    return;
  }

  // gradient w.r.t. slope
  if (this->param_propagate_down_[0]) {
    const Btype* const_top_diff = top[0]->gpu_diff<Btype>();
    Btype* scale_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    caffe_gpu_mul(broadcast_buffer_->count(), x_norm_->template gpu_data<Btype>(), const_top_diff,
        broadcast_buffer_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), broadcast_buffer_->template gpu_data<Btype>(),
        spatial_sum_multiplier_->template gpu_data<Btype>(), Btype(0),
        spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), batch_sum_multiplier_->template gpu_data<Btype>(),
        Btype(1), scale_diff);
  }

  // gradient w.r.t. bias
  if (this->param_propagate_down_[1]) {
    const Btype* const_top_diff = top[0]->gpu_diff<Btype>();
    Btype* shift_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    caffe_gpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), const_top_diff, spatial_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), batch_sum_multiplier_->template gpu_data<Btype>(),
        Btype(1), shift_diff);
  }

  // gradient w.r.t. normalized inputs
  if (propagate_down[0]) {
    const Btype* const_top_diff = top[0]->gpu_diff<Btype>();
    const Btype* const_bottom_diff = bottom[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const Btype* scale_data = this->blobs_[0]->tempalte gpu_data<Btype>();
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template gpu_data<Btype>(), scale_data,
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1), spatial_statistic_->template gpu_data<Btype>(),
        spatial_sum_multiplier_->template gpu_data<Btype>(), Btype(0),
        broadcast_buffer_->template mutable_gpu_data<Btype>());
    caffe_gpu_mul(broadcast_buffer_->count(), const_top_diff,
        broadcast_buffer_->template gpu_data<Btype>(), broadcast_buffer_->template mutable_gpu_data<Btype>());

    // sum of x_hat * (dl / dx_hat)
    caffe_gpu_mul(broadcast_buffer_->count(), x_norm_->template gpu_data<Btype>(),
        broadcast_buffer_->template gpu_data<Btype>(), bottom_diff);
    caffe_gpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), const_bottom_diff, spatial_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), batch_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), batch_statistic_->template mutable_gpu_data<Btype>());

    // x_hat times the sum
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template gpu_data<Btype>(), batch_statistic_->template gpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), spatial_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), bottom_diff);
    caffe_gpu_mul(broadcast_buffer_->count(), x_norm_->template gpu_data<Btype>(),
        const_bottom_diff, bottom_diff);

    // Subtract the average of x_hat times the sum
    caffe_gpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), broadcast_buffer_->template gpu_data<Btype>(),
        spatial_sum_multiplier_->template gpu_data<Btype>(), Btype(0),
        spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), batch_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), batch_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template gpu_data<Btype>(), batch_statistic_->template gpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), spatial_sum_multiplier_->template gpu_data<Btype>(),
        Btype(1), bottom_diff);
    caffe_gpu_axpby(broadcast_buffer_->count(), Btype(1),
        broadcast_buffer_->template gpu_data<Btype>(), Btype(-1) / (num_ * height_ * width_),
        bottom_diff);

    // Multiply with the inverse std
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template gpu_data<Btype>(), x_inv_std_->template gpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_gpu_data<Btype>());
    caffe_gpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template gpu_data<Btype>(), spatial_sum_multiplier_->template gpu_data<Btype>(),
        Btype(0), broadcast_buffer_->template mutable_gpu_data<Btype>());
    caffe_gpu_mul(broadcast_buffer_->count(), const_bottom_diff,
        broadcast_buffer_->template gpu_data<Btype>(), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(BNLayer);

}  // namespace caffe
