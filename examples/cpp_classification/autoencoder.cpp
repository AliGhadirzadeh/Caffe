#include <iostream>
#include <caffe/caffe.hpp>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace std;
using std::string;
/*************************************************/
class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);

  std::vector<float> Predict(const cv::Mat& img);

  private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void SetMean(const string& mean_file);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 public:
 	std::vector<float> encoded_data_;
 	int num_outputs_;
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  };
/*************************************************/
Classifier::Classifier(const string& model_file,
                       const string& trained_file)
{
	Caffe::set_mode(Caffe::CPU);
	/* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  num_outputs_ = net_ ->num_outputs();
	cout << "net inputs x outputs dimension: " << net_ ->num_inputs() << "x" << num_outputs_ << endl;

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();

  cout << "net input layer (channels x width x height): " << num_channels_ <<"x"<<input_layer->width()<<"x"<<input_layer->height() << endl;

  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  Blob<float>* output_layer = net_->output_blobs()[0];
  cout << "net output layer (channels x width x height): " << output_layer->channels()<< "x" << output_layer->width() << "x" << output_layer->height() << endl;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}


std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the encoded output layer to a std::vector */
  if (num_outputs_ > 1)
  {
  	Blob<float>* output_layer2 = net_->output_blobs()[1];
  	const float* begin2 = output_layer2->cpu_data();
  	const float* end2 = begin2 + output_layer2->channels();
		encoded_data_ = std::vector<float>(begin2, end2);
	}
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cout << "image channels: " << img.channels() << endl;
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

	cout << "sample channels: " << sample.channels() << endl;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  sample_normalized = (1.0 / 256.0) * sample_float;
  //cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */

  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv)
{
	std::cout << "\n\n\nProgram is started" << std::endl;

	::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string file         = argv[4];

  Classifier classifier(model_file, trained_file);

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<float> p_img_v = classifier.Predict(img);
	if(classifier.num_outputs_ > 1)
	{
		cout << "The encoded layer is:" << endl;
		for (int i = 0; i < classifier.encoded_data_.size(); i++)
		{
			cout << classifier.encoded_data_[i] << " ";
		}
		cout << endl;
	}
	cout << "showing the output" << endl;
	cv::Mat p_img = cv::Mat(p_img_v).reshape(0,88);
	cv::namedWindow("Image predicted", cv::WINDOW_AUTOSIZE );
  cv::imshow("Image predicted", p_img);
  cv::waitKey(0);
  p_img.convertTo(p_img, CV_8UC1, 255.0);
  cv::imwrite("results/autoencoder.jpg", p_img);
	return 0;
}
