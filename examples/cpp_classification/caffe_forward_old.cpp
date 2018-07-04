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

// definitions
#define NO_OPERATION      0
#define MEAN_SUBTRACTION  1
#define SCALE_DATA        2

using namespace caffe;
using namespace std;
using std::string;

void WriteCVMatToFile(string fileName, cv::Mat data);

/*************************************************/
class CaffeForward {
 public:
  CaffeForward( const string& model_file,
                const string& weight_file,
                const string& prep_op,
                const string& prep_file);

  std::vector<std::vector<float> > ForwardProcess(const cv::Mat& img);
  void SaveChannels(std::vector<std::vector<float> > outputs , const string& output_dir);

  private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void SetMean(const string& mean_file);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 public:
 	int num_outputs_;
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  double scale_;
  unsigned char prep_op_;
};
/*************************************************/
CaffeForward::CaffeForward( const string& model_file,
                            const string& weight_file,
                            const string& prep_op,
                            const string& prep_file)
{
  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
  #endif

	/* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weight_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

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


  //if ( strcmp(prep_op, "MEAN_SUBTRACTION") == 0 )
  //{
    /* Load the binaryproto mean file. */
    SetMean(prep_file);
    prep_op_ = MEAN_SUBTRACTION;

  //}
  /*
  else if( strcmp(prep_op, "SCALE_DATA") == 0 )
  {
    scale_  = boost::lexical_cast<double>(prep_file);
    prep_op_ = SCALE_DATA;
  } */

}

/* Load the mean file in binaryproto format. */
void CaffeForward::SetMean(const string& mean_file) {
  cv::Mat mean_img = cv::imread(mean_file.c_str(), -1);
  cv::resize(mean_img, mean_, input_geometry_);
  mean_.convertTo(mean_, CV_32FC3);
  return;

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
  cv::imwrite("/home/ali/Codes/Caffe/mean.PNG", mean);
  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  mean_ = mean;
  cv::imwrite("/home/ali/Codes/Caffe/mean2.PNG", mean_);
}

std::vector<std::vector<float> > CaffeForward::ForwardProcess(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  std::vector<std::vector<float> > outputs;

  /* Copy the encoded output layer to a std::vector */
  //Blob<float>* input_layer = net_->input_blobs()[0];
  /*cout << input_layer->width() << "x"<< input_layer->height() << "x" <<input_layer->channels() << endl;
  const float* begin = input_layer->cpu_data();
  const float* end = begin + input_layer->channels()*input_layer->width()*input_layer->height();
  outputs.push_back(std::vector<float>(begin, end));*/

  for (int i = 0; i < num_outputs_; i++)
  {
    Blob<float>* output_layer = net_->output_blobs()[i];
    cout << output_layer->width() << "x"<< output_layer->height() << "x" <<output_layer->channels() << endl;
  	const float* begin = output_layer->cpu_data();
  	const float* end = begin + output_layer->channels()*output_layer->width()*output_layer->height();
		outputs.push_back(std::vector<float>(begin, end));
  }
  return outputs;
}

void CaffeForward::SaveChannels(std::vector<std::vector<float> > outputs, const string& output_dir )
{
  for (int i = 0; i < outputs.size(); i++)
  {
      Blob<float>* output_layer = net_->output_blobs()[i];
      char filename[100];
      for (int ch = 0; ch < output_layer->channels(); ch++)
      {
        std::vector<float> channel;
        int p_start = ch*output_layer->width()*output_layer->height();
        int length = output_layer->width()*output_layer->height();
        channel.assign(outputs[i].begin()+p_start, outputs[i].begin()+p_start+length);
        cv::Mat img = cv::Mat(channel).reshape(0,output_layer->height());
        sprintf(filename,"%s/o_%02d_ch_%02d.txt", output_dir.c_str(),i,ch);
        WriteCVMatToFile(filename, img);
        double min, max;
        cv::minMaxLoc(img, &min, &max);
        img = img - min;
        img = img / (max - min);
        img.convertTo(img, CV_8UC1, 255.0);
        sprintf(filename,"%s/o_%02d_ch_%02d.jpg", output_dir.c_str(),i,ch);
        cv::imwrite(filename, img);
      }
  }

}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeForward::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void CaffeForward::Preprocess(const cv::Mat& img,
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

  cv::Mat sample_preprocessed;


  if (prep_op_ == SCALE_DATA)
    sample_preprocessed = (1.0 / 256.0) * sample_float;
  else if(prep_op_ == MEAN_SUBTRACTION)
    cv::subtract(sample_float, mean_, sample_preprocessed);
  //sample_preprocessed = sample_float;


  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */

  cv::split(sample_preprocessed, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv)
{
	std::cout << "\n\n\nProgram is started" << std::endl;

	::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];    // deploy file
  string weight_file  = argv[2];    // weights
  string prep_op      = argv[3];    // preprocess operation
  string prep_file    = argv[4];    // preprocess file
  string input_file   = argv[5];    // input image
  string output_dir   = argv[6];    // output directory to save the files

  CaffeForward CaffeForward(model_file, weight_file, prep_op, prep_file);

  cv::Mat img = cv::imread(input_file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << input_file;

  std::vector< std::vector<float> > outputs = CaffeForward.ForwardProcess(img);
  CaffeForward.SaveChannels(outputs, output_dir);

  /*
	cv::namedWindow("Image ForwardProcessed", cv::WINDOW_AUTOSIZE );
  cv::imshow("Image ForwardProcessed", p_img);
  cv::waitKey(0);
  */
	return 0;
}

void WriteCVMatToFile(string fileName, cv::Mat data)
{
    int nRow = data.size().height;
    int nCol = data.size().width;
    ofstream fileWriter;
    fileWriter.open(fileName.c_str());

    fileWriter << nRow << endl;
    fileWriter << nCol << endl;

    for(int i = 0; i < nRow; i++)
        {
            for(int j = 0; j < nCol; j++)
                {
                    fileWriter << std::fixed << std::setprecision(20) << data.at<float>(i,j) << endl;
                }
        }

    fileWriter.close();
}
