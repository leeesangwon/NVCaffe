#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/interp_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  pad_beg_ = interp_param.pad_beg();
  pad_end_ = interp_param.pad_end();
  CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
  InterpParameter interp_param = this->layer_param_.interp_param();
  if (interp_param.has_shrink_factor() &&
      !interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (interp_param.has_zoom_factor() &&
             !interp_param.has_shrink_factor()) {
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (interp_param.has_height() && interp_param.has_width()) {
    height_out_  = interp_param.height();
    width_out_  = interp_param.width();
  } else if (interp_param.has_shrink_factor() &&
             interp_param.has_zoom_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  } else {
    LOG(FATAL);
  }
  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  caffe_cpu_interp2<Ftype,false>(num_ * channels_,
    bottom[0]->cpu_data<Ftype>(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_cpu_data<Ftype>(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  bottom[0]->set_diff(0.);
  caffe_cpu_interp2_backward<Btype,false>(num_ * channels_,
    bottom[0]->mutable_cpu_diff<Btype>(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->cpu_diff<Btype>(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

#ifndef CPU_ONLY
template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  caffe_gpu_interp2<Ftype,false>(num_ * channels_,
    bottom[0]->gpu_data<Ftype>(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_gpu_data<Ftype>(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Ftype, typename Btype>
void InterpLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  bottom[0]->set_diff(0.);
  caffe_gpu_interp2_backward<Btype,false>(num_ * channels_,
    bottom[0]->mutable_gpu_diff<Btype>(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->gpu_diff<Btype>(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}
#endif

#ifdef CPU_ONLY
STUB_GPU(InterpLayer);
#endif

INSTANTIATE_CLASS_FB(InterpLayer);
REGISTER_LAYER_CLASS(Interp);

}  // namespace caffe
