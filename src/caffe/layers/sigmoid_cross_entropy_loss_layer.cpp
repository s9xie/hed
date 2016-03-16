#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;

  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
      temp_loss_pos = 0;
      temp_loss_neg = 0;
      count_pos = 0;
      count_neg = 0;
      for (int j = 0; j < dim; j ++) {
         if (target[i*dim+j] == 1) {
        	count_pos ++;
        	temp_loss_pos -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
    	else if (target[i*dim+j] == 0) {
        	count_neg ++;
        	temp_loss_neg -= input_data[i*dim + j] * (target[i*dim+j] - (input_data[i*dim + j] >= 0)) -
                	log(1 + exp(input_data[i*dim + j] - 2 * input_data[i*dim + j] * (input_data[i*dim + j] >= 0)));
    	}
     } 
     loss_pos += temp_loss_pos * count_neg / (count_pos + count_neg);
     loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
  }
  top[0]->mutable_cpu_data()[0] = (loss_pos * 1 + loss_neg) / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    Dtype count_pos = 0;
    Dtype count_neg = 0;
    int dim = bottom[0]->count() / bottom[0]->num();

    for (int i = 0; i < num; ++i) {
    	count_pos = 0;
    	count_neg = 0;
    	for (int j = 0; j < dim; j ++) {
           	if (target[i*dim+j] == 1) {
                	count_pos ++;
        	}
        	else if (target[i*dim+j] == 0) {
                	count_neg ++;
        	}
     	}
    	for (int j = 0; j < dim; j ++) {
        	if (target[i*dim+j] == 1) {
               		bottom_diff[i * dim + j] *= 1 * count_neg / (count_pos + count_neg);
        	}
        	else if (target[i*dim+j] == 0) {
                	bottom_diff[i * dim + j] *= count_pos / (count_pos + count_neg);
        	}
     	}
    }
    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
