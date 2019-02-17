#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  frozen_ = this->layer_param_.bn_param().frozen();
  bn_momentum_ = this->layer_param_.bn_param().momentum();
  bn_eps_ = this->layer_param_.bn_param().eps();
  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);

    const Type btype = blobs_type();

    vector<int> shape;
    shape.push_back(1);
    shape.push_back(bottom[0]->channels());
    shape.push_back(1);
    shape.push_back(1);
    // slope
    this->blobs_[0] = Blob::create(btype, btype);
    this->blobs_[0]->Reshape(shape);
    shared_ptr<Filler<btype> > slope_filler(GetFiller<btype>(
        this->layer_param_.bn_param().slope_filler()));
    slope_filler->Fill(this->blobs_[0].get());
    // bias
    this->blobs_[1] = Blob::create(btype, btype);
    this->blobs_[1]->Reshape(shape);
    shared_ptr<Filler<btype> > bias_filler(GetFiller<btype>(
        this->layer_param_.bn_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
    // moving average mean
    this->blobs_[2] = Blob::create(btype, btype);
    this->blobs_[2]->Reshape(shape);
    this->blobs_[2]->set_data(0.);
    // moving average variance
    this->blobs_[3] = Blob::create(btype, btype);
    this->blobs_[3]->Reshape(shape);
    this->blobs_[3]->set_data(frozen_ ? 1. : 0.);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // runing average stats does not use weight decay and learning rate
  while (this->layer_param_.param_size() < 4){
    this->layer_param_.mutable_param()->Add();
  }
  this->layer_param_.mutable_param(2)->set_lr_mult(btype(0));
  this->layer_param_.mutable_param(2)->set_decay_mult(btype(0));

  this->layer_param_.mutable_param(3)->set_lr_mult(btype(0));
  this->layer_param_.mutable_param(3)->set_decay_mult(btype(0));

  // shutdown scale and bias update in frozen mode
  if (this->frozen_){
    // slope
    this->layer_param_.mutable_param(0)->set_lr_mult(btype(0));
    this->layer_param_.mutable_param(0)->set_decay_mult(btype(0));

    // bias
    this->layer_param_.mutable_param(1)->set_lr_mult(btype(0));
    this->layer_param_.mutable_param(1)->set_decay_mult(btype(0));
  }

  // =====================================
  int N = bottom[0]->shape(0);
  int C = bottom[0]->shape(1);
  int H = bottom[0]->shape(2);
  int W = bottom[0]->shape(3);

  broadcast_buffer_ = Blob::create<Ftype>(N, C, H, W);
  spatial_statistic_ = Blob::create<Ftype>(N, C, 1, 1);
  batch_statistic_ = Blob::create<Ftype>(1, C, 1, 1);
  x_norm_ = Blob::create<Ftype>(N, C, H, W);
  x_inv_std_ = Blob::create<Ftype>(1, C, 1, 1);
  spatial_sum_multiplier_ = Blob::create<Ftype>(1, 1, H, W);
  batch_sum_multiplier_ = Blob::create<Ftype>(N, 1, 1, 1);
}

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  broadcast_buffer_->ReshapeLike(*(bottom[0]));
  spatial_statistic_->Reshape(num_, channels_, 1, 1);
  batch_statistic_->Reshape(1, channels_, 1, 1);

  x_norm_->ReshapeLike(*(bottom[0]));
  x_inv_std_->ReshapeLike(batch_statistic_);

  spatial_sum_multiplier_->Reshape(1, 1, height_, width_);
  spatial_sum_multiplier_->set_data(1.)
  batch_sum_multiplier_->Reshape(num_, 1, 1, 1);
  batch_sum_multiplier_->set_data(1.)
}

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
  const vector<Blob*>& top) {
  const Ftype* const_bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* const_top_data = top[0]->cpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_cpu_data<Ftype>();

  const Ftype* scale_data = this->blobs_[0]->template cpu_data<Ftype>();
  const Ftype* shift_data = this->blobs_[1]->template cpu_data<Ftype>();

  // Mean normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average mean
    caffe_copy(batch_statistic_->count(), this->blobs_[2]->template cpu_data<Ftype>(),
        batch_statistic_->template mutable_cpu_data<Ftype>());
  } else {
    // Compute the mean by averaging over spatial and batch dimensions.
    caffe_cpu_gemv<Ftype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Ftype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_->template cpu_data<Ftype>(), Ftype(0),
        spatial_statistic_->template mutable_cpu_data<Ftype>());
    caffe_cpu_gemv<Ftype>(CblasTrans, num_, channels_,
        Ftype(1) / num_, spatial_statistic_->template cpu_data<Ftype>(),
        batch_sum_multiplier_->template cpu_data<Ftype>(), Ftype(0),
        batch_statistic_->template mutable_cpu_data<Ftype>());
    // Add to the moving average
    if (!frozen_) {
      caffe_cpu_axpby(batch_statistic_->count(),
          Ftype(1) - bn_momentum_, batch_statistic_->template cpu_data<Ftype>(),
          bn_momentum_, this->blobs_[2]->template mutable_cpu_data<Ftype>());
    }
  }
  // Broadcast the mean vector
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template cpu_data<Ftype>(), batch_statistic_->template cpu_data<Ftype>(),
      Ftype(0), spatial_statistic_->template mutable_cpu_data<Ftype>());
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(-1),
      spatial_statistic_->template cpu_data<Ftype>(), spatial_sum_multiplier_->template cpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_cpu_data<Ftype>());
  // Subtract
  caffe_add(broadcast_buffer_->count(), const_bottom_data,
      broadcast_buffer_->template cpu_data<Ftype>(), top_data);

  // Variance normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average variance
    caffe_copy(batch_statistic_->count(), this->blobs_[3]->template cpu_data<Ftype>(),
        batch_statistic_->template mutable_cpu_data<Ftype>());
  } else {
    // calculate batch variance
    caffe_powx(broadcast_buffer_->count(), const_top_data, Ftype(2),
        broadcast_buffer_->template mutable_cpu_data<Ftype>());
    caffe_cpu_gemv<Ftype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Ftype(1) / (height_ * width_), broadcast_buffer_->template cpu_data<Ftype>(),
        spatial_sum_multiplier_->template cpu_data<Ftype>(), Ftype(0),
        spatial_statistic_->template mutable_cpu_data<Ftype>());
    caffe_cpu_gemv<Ftype>(CblasTrans, num_, channels_, Ftype(1) / num_,
        spatial_statistic_->template cpu_data<Ftype>(), batch_sum_multiplier_->template cpu_data<Ftype>(),
        Ftype(0), batch_statistic_->template mutable_cpu_data<Ftype>());

    // Add to the moving average
    caffe_cpu_axpby(batch_statistic_->count(),
        Ftype(1) - bn_momentum_, batch_statistic_->template cpu_data<Ftype>(),
        bn_momentum_, this->blobs_[3]->template mutable_cpu_data<Ftype>());
  }

  // Add eps
  caffe_add_scalar(batch_statistic_->count(), bn_eps_,
                   batch_statistic_->template mutable_cpu_data<Ftype>());
  // Inverse standard deviation
  caffe_powx(batch_statistic_->count(), batch_statistic_->template cpu_data<Ftype>(),
             Ftype(-0.5), batch_statistic_->template mutable_cpu_data<Ftype>());

  // Broadcast the inverse std
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Ftype(1), batch_sum_multiplier_->template cpu_data<Ftype>(), batch_statistic_->template cpu_data<Ftype>(),
        Ftype(0), spatial_statistic_->template mutable_cpu_data<Ftype>());
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template cpu_data<Ftype>(), spatial_sum_multiplier_->template cpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_cpu_data<Ftype>());
  // Multiply with the inverse std
  caffe_mul(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template cpu_data<Ftype>(), top_data);

  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_->count(), const_top_data,
        x_norm_->template mutable_cpu_data<Ftype>());
    caffe_copy(batch_statistic_->count(), batch_statistic_->template cpu_data<Ftype>(),
        x_inv_std_->template mutable_cpu_data<Ftype>());
  }

  // Scale
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template cpu_data<Ftype>(), scale_data,
      Ftype(0), spatial_statistic_->template mutable_cpu_data<Ftype>());
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template cpu_data<Ftype>(), spatial_sum_multiplier_->template cpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_cpu_data<Ftype>());
  caffe_mul(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template cpu_data<Ftype>(), top_data);

  // Shift
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Ftype(1), batch_sum_multiplier_->template cpu_data<Ftype>(), shift_data,
      Ftype(0), spatial_statistic_->template mutable_cpu_data<Ftype>());
  caffe_cpu_gemm<Ftype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Ftype(1),
      spatial_statistic_->template cpu_data<Ftype>(), spatial_sum_multiplier_->template cpu_data<Ftype>(),
      Ftype(0), broadcast_buffer_->template mutable_cpu_data<Ftype>());
  caffe_add(broadcast_buffer_->count(), const_top_data,
      broadcast_buffer_->template cpu_data<Ftype>(), top_data);
}

template <typename Ftype, typename Btype>
void BNLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
  const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (frozen_) {
    if (propagate_down[0]) {
      const Btype* const_top_diff = top[0]->cpu_diff<Btype>();
      Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
      // Use the moving average variance
      caffe_copy(batch_statistic_->count(), this->blobs_[3]->template cpu_data<Btype>(),
          batch_statistic_->template mutable_cpu_data<Btype>());
      caffe_add_scalar(batch_statistic_->count(), bn_eps_,
          batch_statistic_->template mutable_cpu_data<Btype>());
      caffe_powx(batch_statistic_->count(), batch_statistic_->template cpu_data<Btype>(),
          Btype(-0.5), batch_statistic_->template mutable_cpu_data<Btype>());
      // Divide slope with std
      caffe_mul(batch_statistic_->count(), this->blobs_[0]->template cpu_data<Btype>(),
          batch_statistic_->template cpu_data<Btype>(), batch_statistic_->template mutable_cpu_data<Btype>());
      // Broadcast
      caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
          Btype(1), batch_sum_multiplier_->template cpu_data<Btype>(), batch_statistic_->template cpu_data<Btype>(),
          Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
      caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
          height_ * width_, 1, Btype(1),
          spatial_statistic_->template cpu_data<Btype>(), spatial_sum_multiplier_->template cpu_data<Btype>(),
          Btype(0), broadcast_buffer_->template mutable_cpu_data<Btype>());
      // Elementwise multiply top grad with (slope / std)
      caffe_mul(broadcast_buffer_->count(), const_top_diff,
          broadcast_buffer_->template cpu_data<Btype>(), bottom_diff);
    }
    return;
  }

  // gradient w.r.t. slope
  if (this->param_propagate_down_[0]) {
    const Btype* const_top_diff = top[0]->cpu_diff<Btype>();
    Btype* scale_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
    caffe_mul(broadcast_buffer_->count(), x_norm_->template cpu_data<Btype>(), const_top_diff,
        broadcast_buffer_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), broadcast_buffer_->template cpu_data<Btype>(),
        spatial_sum_multiplier_->template cpu_data<Btype>(), Btype(0),
        spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), batch_sum_multiplier_->template cpu_data<Btype>(),
        Btype(1), scale_diff);
  }

  // gradient w.r.t. bias
  if (this->param_propagate_down_[1]) {
    const Btype* const_top_diff = top[0]->cpu_diff<Btype>();
    Btype* shift_diff = this->blobs_[1]->template mutable_cpu_diff<Btype>();
    caffe_cpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), const_top_diff, spatial_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), batch_sum_multiplier_->template cpu_data<Btype>(),
        Btype(1), shift_diff);
  }

  // gradient w.r.t. normalized inputs
  if (propagate_down[0]) {
    const Btype* const_top_diff = top[0]->cpu_diff<Btype>();
    const Btype* const_bottom_diff = bottom[0]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const Btype* scale_data = this->blobs_[0]->template cpu_data<Btype>();

    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template cpu_data<Btype>(), scale_data,
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1), spatial_statistic_->template cpu_data<Btype>(),
        spatial_sum_multiplier_->template cpu_data<Btype>(), Btype(0),
        broadcast_buffer_->template mutable_cpu_data<Btype>());
    caffe_mul(broadcast_buffer_->count(), const_top_diff,
        broadcast_buffer_->template cpu_data<Btype>(), broadcast_buffer_->template mutable_cpu_data<Btype>());

    // sum of x_hat * (dl / dx_hat)
    caffe_mul(broadcast_buffer_->count(), x_norm_->template cpu_data<Btype>(),
        broadcast_buffer_->template cpu_data<Btype>(), bottom_diff);
    caffe_cpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), const_bottom_diff, spatial_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), batch_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), batch_statistic_->template mutable_cpu_data<Btype>());

    // x_hat times the sum
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template cpu_data<Btype>(), batch_statistic_->template cpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), spatial_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), bottom_diff);
    caffe_mul(broadcast_buffer_->count(), x_norm_->template cpu_data<Btype>(),
        const_bottom_diff, bottom_diff);

    // Subtract the average of x_hat times the sum
    caffe_cpu_gemv<Btype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Btype(1), broadcast_buffer_->template cpu_data<Btype>(),
        spatial_sum_multiplier_->template cpu_data<Btype>(), Btype(0),
        spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemv<Btype>(CblasTrans, num_, channels_, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), batch_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), batch_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template cpu_data<Btype>(), batch_statistic_->template cpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), spatial_sum_multiplier_->template cpu_data<Btype>(),
        Btype(1), bottom_diff);
    caffe_cpu_axpby(broadcast_buffer_->count(), Btype(1),
        broadcast_buffer_->template cpu_data<Btype>(), Btype(-1) / (num_ * height_ * width_),
        bottom_diff);

    // Multiply with the inverse std
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Btype(1), batch_sum_multiplier_->template cpu_data<Btype>(), x_inv_std_->template cpu_data<Btype>(),
        Btype(0), spatial_statistic_->template mutable_cpu_data<Btype>());
    caffe_cpu_gemm<Btype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Btype(1),
        spatial_statistic_->template cpu_data<Btype>(), spatial_sum_multiplier_->template cpu_data<Btype>(),
        Btype(0), broadcast_buffer_->template mutable_cpu_data<Btype>());
    caffe_mul(broadcast_buffer_->count(), const_bottom_diff,
        broadcast_buffer_->template cpu_data<Btype>(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

INSTANTIATE_CLASS_FB(BNLayer);
REGISTER_LAYER_CLASS(BN);
}  // namespace caffe
