#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cudnn_bn_layer.hpp"

#if CUDNN_VERSION_MIN(5, 0, 0)

namespace caffe {

template <typename Ftype, typename Btype>
void CuDNNBNLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  BNLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  save_mean_->ReshapeLike(*(this->blobs_[2]));
  save_inv_variance_->ReshapeLike(*(this->blobs_[3]));

  // Initialize CUDNN.
  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_bn_param_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bn_param_desc_);
#if CUDNN_VERSION_MIN(7, 0, 0)
  mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  mode_ = CUDNN_BATCHNORM_SPATIAL;      // only SPATIAL mode is supported
#endif

  handles_setup_ = true;
  
  LOG(INFO)<<"using cuDNN BN engine";
}

template <typename Ftype, typename Btype>
void CuDNNBNLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  // Do not call BNLayer::Reshape function as some members are unnecessary
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  // CUDNN tensors
  cudnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  cudnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  cudnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  cudnn::setTensor4dDesc<Btype>(&bwd_top_desc_, this->num_, this->channels_,
                                this->height_, this->width_);
  // Fix to the spatial mode
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(fwd_bn_param_desc_,
      fwd_bottom_desc_, mode_));
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bwd_bn_param_desc_,
      bwd_bottom_desc_, mode_));

  if (this->frozen_){
    this->broadcast_buffer_->ReshapeLike(*(bottom[0]));
    this->spatial_statistic_->Reshape(this->num_, this->channels_, 1, 1);
    this->batch_statistic_->Reshape(1, this->channels_, 1, 1);

    this->spatial_sum_multiplier_->Reshape(1, 1, this->height_, this->width_);
    this->spatial_sum_multiplier_->set_data(1.F);
    this->batch_sum_multiplier_->Reshape(this->num_, 1, 1, 1);
    this->batch_sum_multiplier_->set_data(1.F);
  }
}

template <typename Ftype, typename Btype>
CuDNNBNLayer<Ftype, Btype>::~CuDNNBNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(fwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(bwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(fwd_top_desc_);
  cudnnDestroyTensorDescriptor(bwd_top_desc_);
  cudnnDestroyTensorDescriptor(fwd_bn_param_desc_);
  cudnnDestroyTensorDescriptor(bwd_bn_param_desc_);
}

INSTANTIATE_CLASS_FB(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
