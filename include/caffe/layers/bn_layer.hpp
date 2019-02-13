#ifndef CAFFE_BN_LAYER_HPP_
#define CAFFE_BN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Batch normalization the input blob along the channel axis while
 *        averaging over the spatial axes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class BNLayer : public Layer<Ftype, Btype> {
  typedef Ftype Dtype;

 public:
  explicit BNLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "BN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  void AverageAllExceptChannel(const Dtype* input, Dtype* output);
  void BroadcastChannel(const Dtype* input, Dtype* output);

  bool frozen_;
  Dtype bn_momentum_;
  Dtype bn_eps_;

  int num_;
  int channels_;
  int height_;
  int width_;

  TBlob<Dtype> broadcast_buffer_;
  TBlob<Dtype> spatial_statistic_;
  TBlob<Dtype> batch_statistic_;

  TBlob<Dtype> x_norm_;
  TBlob<Dtype> x_inv_std_;

  TBlob<Dtype> spatial_sum_multiplier_;
  TBlob<Dtype> batch_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BN_LAYER_HPP_
