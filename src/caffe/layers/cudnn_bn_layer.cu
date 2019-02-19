#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cudnn_bn_layer.hpp"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNBNLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const Ftype* scale_data = this->blobs_[0]->template gpu_data<Ftype>();
  const Ftype* bias_data = this->blobs_[1]->template gpu_data<Ftype>();

  const double epsilon = max(this->bn_eps_, CUDNN_BN_MIN_EPSILON);

  if (this->phase_ == TEST || this->frozen_) {
    const Ftype* running_mean_data = this->blobs_[2]->template gpu_data<Ftype>();
    const Ftype* running_variance_data = this->blobs_[3]->template gpu_data<Ftype>();
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(Caffe::cudnn_handle(0),
        CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<Ftype>::one,
        cudnn::dataType<Ftype>::zero,
        fwd_bottom_desc_, bottom_data,
        fwd_top_desc_, top_data,
        fwd_bn_param_desc_, scale_data, bias_data,
        running_mean_data, running_variance_data,
        epsilon));
  } else {
    Ftype* running_mean_data = this->blobs_[2]->template mutable_gpu_data<Ftype>();
    Ftype* running_variance_data = this->blobs_[3]->template mutable_gpu_data<Ftype>();
    Ftype* save_mean_data = save_mean_->template mutable_gpu_data<Ftype>();
    Ftype* save_inv_variance_data = save_inv_variance_->template mutable_gpu_data<Ftype>();
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(0),
        mode_,
        cudnn::dataType<Ftype>::one,
        cudnn::dataType<Ftype>::zero,
        fwd_bottom_desc_, bottom_data,
        fwd_top_desc_, top_data,
        fwd_bn_param_desc_, scale_data, bias_data,
        1 - this->bn_momentum_,
        running_mean_data, running_variance_data,
        epsilon,
        save_mean_data, save_inv_variance_data));
  }
}

template <typename Ftype, typename Btype>
void CuDNNBNLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
   if (this->frozen_){
     BNLayer<Ftype, Btype>::Backward_gpu(top, propagate_down, bottom);
     return;
   }
  if (propagate_down[0] || this->param_propagate_down_[0] ||
      this->param_propagate_down_[1]) {
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    const Btype* bottom_data = bottom[0]->gpu_data<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const Btype* scale_data = this->blobs_[0]->template gpu_data<Btype>();
    Btype* scale_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    const Btype* save_mean_data = save_mean_->template gpu_data<Btype>();
    const Btype* save_inv_variance_data = save_inv_variance_->template gpu_data<Btype>();

    const double epsilon = max(this->bn_eps_, CUDNN_BN_MIN_EPSILON);

    CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(0),
        mode_,
        cudnn::dataType<Btype>::one,
        cudnn::dataType<Btype>::zero,
        cudnn::dataType<Btype>::one,
        cudnn::dataType<Btype>::one,
        bwd_bottom_desc_, bottom_data,
        bwd_top_desc_, top_diff,
        bwd_bottom_desc_, bottom_diff,
        bwd_bn_param_desc_, scale_data,
        scale_diff, bias_diff,
        epsilon,
        save_mean_data, save_inv_variance_data));

  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
