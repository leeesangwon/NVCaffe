#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/parallel.hpp"

#define IDL_CACHE_PROGRESS 0.05F

namespace caffe {

static std::mutex idl_mutex_;

static size_t idl_id(const string& ph, const string& name) {
  std::lock_guard<std::mutex> lock(idl_mutex_);
  static size_t id = 0UL;
  static map<string, size_t> ph_names;
  string ph_name = ph + name;
  auto it = ph_names.find(ph_name);
  if (it != ph_names.end()) {
    return it->second;
  }
  CHECK_LT(id, MAX_IDL_CACHEABLE);
  ph_names.emplace(ph_name, id);
  return id++;
};

template <typename Ftype, typename Btype>
ImageSegDataLayer<Ftype, Btype>::ImageSegDataLayer(const LayerParameter& param, size_t sover_rank)
    : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank),
      id_(idl_id(Phase_Name(this->phase_), this->name())),
      epoch_count_(0UL) {
  LOG_IF(INFO, P2PManager::global_rank() == 0)
             << this->print_current_device() << " ImageSegDataLayer: " << this
             << " name: " << this->name()
             << " id: " << id_
             << " threads: " << this->threads_num();
}

template <typename Ftype, typename Btype>
ImageSegDataLayer<Ftype, Btype>::~ImageSegDataLayer<Ftype, Btype>() {
  if (layer_inititialized_flag_.is_set()) {
    this->StopInternalThread();
  }
}

template <typename Ftype, typename Btype>
void ImageSegDataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  TBlob<Btype> transformed_data;
  TBlob<Btype> transformed_label;

  const ImageDataParameter& image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color  = image_data_param.is_color();
  const string& root_folder = image_data_param.root_folder();
  const int label_type = image_data_param.label_type();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  if (this->rank_ % Caffe::device_in_use_per_host_count() == 0) {
    // Read the file with filenames and labels
    lines_[id_].clear();
    const string &source = image_data_param.source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    CHECK(infile.good()) << "File " << source;
    string linestr;
    while (std::getline(infile, linestr)) {
      std::istringstream iss(linestr);
      string imgfn;
      iss >> imgfn;
      string segfn = "";
      if (label_type != ImageDataParameter_LabelType_NONE) {
        iss >> segfn;
      }
      lines_[id_].emplace_back(std::make_pair(imgfn, segfn));
    }
    if (image_data_param.shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
      ShuffleImages();
    }
    if (image_data_param.dataset_share_per_node() > 1.F
        && this->phase_ == TRAIN) {
      lines_[id_].resize(std::lround(lines_[id_].size()
          / image_data_param.dataset_share_per_node()));
    }
  }
  LOG_IF(INFO, P2PManager::global_rank() == 0)
  << this->print_current_device() << " A total of " << lines_[id_].size() << " images.";

  size_t skip = 0UL;
  // Check if we would need to randomly skip a few data points
  if (image_data_param.rand_skip()) {
    if (Caffe::gpus().size() > 1) {
      LOG(WARNING) << "Skipping data points is not supported in multiGPU mode";
    } else {
      skip = caffe_rng_rand() % image_data_param.rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points";
      CHECK_GT(lines_[id_].size(), skip) << "Not enough points to skip";
    }
  }
  line_ids_.resize(this->threads_num());
  for (size_t i = 0; i < this->threads_num(); ++i) {
    line_ids_[i] = (this->rank_ % Caffe::device_in_use_per_host_count()) *
        this->threads_num() + i + skip;
  }

  // Read an image, and use it to initialize the top blob.
  const string& imgfn = lines_[id_][line_ids_[0]].first;
  bool from_cache = false;
  cv::Mat cv_img = next_mat(root_folder, file_name, new_height, new_width ,is_color, short_side,
      from_cache);
  CHECK(cv_img.data) << "Could not load " << root_folder + imgfn;
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = image_data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  int crop_height = 0;
  int crop_width = 0;
  CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
	|| (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (transform_param.has_crop_size()) {
    crop_height = crop;
    crop_width = crop;
  } 
  if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
    crop_height = transform_param.crop_height();
    crop_width = transform_param.crop_width();
  }
  if (crop_width <= 0 || crop_height <= 0) {
    LOG_FIRST_N(INFO, 1) << "Crop is not set. Using '" << root_folder + file_name
                         << "' as a model, w=" << cv_img.rows << ", h=" << cv_img.cols;
    crop_height = cv_img.rows;
    crop_width = cv_img.cols;
  }
  vector<int> top_shape { batch_size, cv_image.channels(), crop_height, crop_width };
  transformed_data.Reshape(top_shape);
  top[0]->Reshape(top_shape);
  LOG_IF(INFO, P2PManager::global_rank() == 0) << "output data size: " << top[0]->num() << ", "
    << top[0]->channels() << ", " << top[0]->height() << ", "
    << top[0]->width();
  // label image
  vector<int> label_shape { batch_size, 1, crop_height, crop_width };
  transformed_label.Reshape(label_shape);
  top[1]->Reshape(label_shape);
  this->batch_transformer_->reshape(top_shape, label_shape);
  layer_inititialized_flag_.set();
}

template <typename Ftype, typename Btype>
void ImageSegDataLayer<Ftype, Btype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_[id_].begin(), lines_[id_].end(), prefetch_rng);
}

template<typename Ftype, typename Btype>
void ImageSegDataLayer<Ftype, Btype>::InitializePrefetch() {}

template<typename Ftype, typename Btype>
std::vector<cv::Mat> ImageSegDataLayer<Ftype, Btype>::next_mat_vector(
      const string& root_folder, const std::pair<std::string, std::string> file_names,
      int height, int width, bool is_color, int short_side, bool& from_cache, 
      const int label_type, const int ignore_label) {
  from_cache = false;
  if (this->layer_param_.image_data_param().cache()) {
    std::lock_guard<std::mutex> lock(cache_mutex_[id_]);
    if (cache_[id_].size() > 0) {
      auto it = cache_[id_].find(file_names);
      if (it != cache_[id_].end()) {
        from_cache = true;
        return it->second;
      }
    }
  }
  std::vector<cv::Mat> cv_img_seg;
  // img
  cv_img_seg.push_back(ReadImageToCVMat(root_folder + file_names.first, height, width, is_color, short_side));
  if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + file_names.first;
  }
  // seg
  if (label_type == ImageDataParameter_LabelType_PIXEL) {
    cv_img_seg.push_back(ReadImageToCVMat(root_folder + filenames.second,
            new_height, new_width, false, short_side));
    if (!cv_img_seg[1].data) {
        DLOG(INFO) << "Fail to load seg: " << root_folder + filenames.second;
    }
  }
  else if (label_type == ImageDataParameter_LabelType_IMAGE) {
    const int label = atoi(filenames.second.c_str());
    cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
                CV_8UC1, cv::Scalar(label));
    cv_img_seg.push_back(seg);      
  }
  else {
    cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
                CV_8UC1, cv::Scalar(ignore_label));
    cv_img_seg.push_back(seg);
  }
  return cv_img_seg;
}

template <typename Ftype, typename Btype>
bool ImageSegDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t queue_id) {
  TBlob<Btype> transformed_data;
  TBlob<Btype> transformed_label;

  CHECK(batch->data_->count());
  const ImageDataParameter& image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  const bool cache_on = image_data_param.cache();
  const bool shuffle = image_data_param.shuffle();
  const string& root_folder = image_data_param.root_folder();
  const int label_type = image_data_param.label_type();
  const int ignore_label = image_data_param.ignore_label();
  unordered_map<std::pair<std::string, std::string>, std::pair<cv::Mat, cv::Mat>>& cache = cache_[id_];

  size_t line_id = line_ids_[thread_id];
  const size_t line_bucket = Caffe::device_in_use_per_host_count() * this->threads_num();
  const size_t lines_size = lines_[id_].size();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  const string& file_name = lines_[id_][line_id].first;
  bool from_cache = false;
  cv::Mat cv_img = next_mat(root_folder, file_name, new_height, new_width, is_color, short_side,
      from_cache);

  CHECK(cv_img.data) << "Could not load " << (root_folder + file_name);
  int crop_height = 0;
  int crop_width = 0;
  CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
	|| (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (transform_param.has_crop_size()) {
    crop_height = crop;
    crop_width = crop;
  } 
  if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
    crop_height = transform_param.crop_height();
    crop_width = transform_param.crop_width();
  }
  if (crop_width <= 0 || crop_height <= 0) {
    LOG_FIRST_N(INFO, 1) << "Crop is not set. Using '" << root_folder + file_name
                         << "' as a model, w=" << cv_img.rows << ", h=" << cv_img.cols;
    crop_height = cv_img.rows;
    crop_width = cv_img.cols;
  }

  // Infer the expected blob shape from a cv_img.
  vector<int> top_shape { batch_size, cv_img.channels(), crop_height, crop_width };
  transformed_data.Reshape(top_shape);
  batch->data_->Reshape(top_shape);
  vector<int> label_shape {batch_size, 1, crop_height, crop_width };
  transformed_label.Reshape()
  batch->label_->Reshape(label_shape);
  vector<Btype> tmp(top_shape[1] * top_shape[2] * top_shape[3]);
  Btype* prefetch_data = batch->data_->mutable_cpu_data<Btype>();
  Btype* prefetch_label = batch->label_->mutable_cpu_data<Btype>();
  Packing packing = NHWC;

  // datum scales
  const size_t buf_len = batch->data_->offset(1);
  const size_t buf_len_label = batch->label_->offset(1);
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, line_id);
    const string& file_names = lines_[id_][line_id]
    from_cache = false;
    vector<cv::Mat> cv_img_seg;
    cv_img_seg = next_mat_vector(root_folder, file_names, new_height, new_width, is_color, short_side, 
                                 from_cache, label_type, ignore_label);
    if (cv_img_seg[0].data) {
      int offset;
      offset = batch->data_->offset(item_id);
      transformed_data.set_cpu_data(prefetch_data + offset)

      offset = batch->label_.offset(item_id);
      transformed_label.set_cpu_data(prefetch_label + offset)

#if defined(USE_CUDNN)
      this->bdt(thread_id)->TransformImgAndSeg(cv_img_seg, 
          &(transformed_data), &(transformed_label), ignore_label);
#else
      CHECK_EQ(buf_len, tmp.size());
      this->bdt(thread_id)->TransformImgAndSeg(cv_img_seg, 
          &(transformed_data), &(transformed_label), ignore_label);
      hwc2chw(top_shape[1], top_shape[3], top_shape[2], tmp.data(), prefetch_data + offset);
      packing = NCHW;
#endif
    }
    if (cache_on && !cached_[id_] && !from_cache) {
      std::lock_guard<std::mutex> lock(cache_mutex_[id_]);
      if (cv_img_seg[0].data != nullptr && cv_img_seg[1].data != nullptr) {
        auto em = cache.emplace(file_names, cv_img_seg);
        if (em.second) {
          ++cached_num_[id_];
        } else {
          DLOG(WARNING) << this->print_current_device()
                        << " Duplicate @ " << line_id << " " 
                        << file_names[0] << " " << file_names[1];
        }
      } else {
        ++failed_num_[id_];
      }
      if (cached_num_[id_] + failed_num_[id_] >= lines_size) {
        cached_[id_] = true;
        LOG_IF(INFO, P2PManager::global_rank() == 0) << cache.size()
                  << " objects cached for " << Phase_Name(this->phase_)
                  << " by layer " << this->name();
      } else if ((float) cached_num_[id_] / lines_size >=
          cache_progress_[id_] + IDL_CACHE_PROGRESS) {
        cache_progress_[id_] = (float) cached_num_[id_] / lines_size;
        LOG_IF(INFO, P2PManager::global_rank() == 0)     << this->print_current_device() << " "
                  << std::setw(2) << std::setfill(' ') << f_round1(cache_progress_[id_] * 100.F)
                  << "% of objects cached for "
                  << Phase_Name(this->phase_) << " by layer '" << this->name() << "' ("
                  << cached_num_[id_] << "/" << lines_size << ")";
      }
    }

    // go to the next iter
    line_ids_[thread_id] += line_bucket;
    if (line_ids_[thread_id] >= lines_size) {
      while (line_ids_[thread_id] >= lines_size) {
        line_ids_[thread_id] -= lines_size;
      }
      if (thread_id == 0 &&
          this->rank_ % Caffe::device_in_use_per_host_count() == 0) {
        if (this->phase_ == TRAIN) {
          // We have reached the end. Restart from the first.
          LOG_IF(INFO, P2PManager::global_rank() == 0)
          << this->print_current_device() << " Restarting data prefetching (" << lines_size << ")";
          if (epoch_count_ == 0UL) {
            epoch_count_ += std::lround(lines_[id_].size()
                * image_data_param.dataset_share_per_node());
            Caffe::report_epoch_count(epoch_count_);
          }
        }
        if (shuffle) {
          LOG_IF(INFO, P2PManager::global_rank() == 0) << "Shuffling data";
          ShuffleImages();
        }
      }
    }
    line_id = line_ids_[thread_id];
  }
  batch->set_data_packing(packing);
  batch->set_id(this->batch_id(thread_id));
  return chached_[id_];
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(ImageSegData);

}  // namespace caffe
