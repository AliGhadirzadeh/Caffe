#include <iostream>
#include <caffe/caffe.hpp>
#include <boost/lexical_cast.hpp>
#include "boost/algorithm/string.hpp"
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
#include <fstream>
#include <unistd.h>

// definitions
#define NO_OPERATION      0
#define MEAN_SUBTRACTION  1
#define SCALE_DATA        2

using namespace caffe;
using namespace std;
using std::string;

void WriteCVMatToFile(string fileName, cv::Mat data);
cv::Mat ReadCVMatFromFile(string filename);
cv::Mat ReadInputFile(string filename);
/*************************************************/
class CaffeForward {
 public:
  CaffeForward( const string& model_file,const string& weight_file,const string& preprocess,const string& preprocess_data);

  std::vector<std::vector<float> > ForwardProcess(std::vector <cv::Mat> &inputs);
  void SaveChannels(std::vector<std::vector<float> > outputs , const string& output_dir);
  private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int input_idx);
  void SetMean(const string& mean_file);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, int input_idx);

 public:
 	int num_outputs_, num_inputs_;
 private:
  shared_ptr<Net<float> > net_;
  std::vector<cv::Size> input_geometry_;
  std::vector<int> num_channels_;
  cv::Mat mean_;
  double scale_;
  unsigned char preprocess_;
};
/*************************************************/
/* CAFFE FORWARD */
CaffeForward::CaffeForward( const string& model_file,const string& weight_file,const string& preprocess,const string& preprocess_data)
{
  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
  #endif
	/* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  std::vector<std::string> model_names;
  boost::split(model_names, weight_file, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    net_->CopyTrainedLayersFrom(model_names[i]);
  }

  num_outputs_ = net_->num_outputs();
  num_inputs_ = net_->num_inputs();
	//cout << "net inputs x outputs dimension: " << num_inputs_ << "x" << num_outputs_ << endl;

  for(int i = 0; i < num_inputs_; i++)
  {
    Blob<float>* input_layer = net_->input_blobs()[i];
    num_channels_.push_back(input_layer->channels());
    input_geometry_.push_back(cv::Size(input_layer->width(), input_layer->height()));
    //cout << "input " << i << " (channels x width x height): " << num_channels_[i] <<"x"<<input_geometry_[i].width<<"x"<<input_geometry_[i].height << endl;
    CHECK(num_channels_[i] == 3 || num_channels_[i] == 1) << "Input layer should have 1 or 3 channels.";
  }
  /*
  for(int i = 0; i < num_outputs_; i++)
  {
    Blob<float>* output_layer = net_->output_blobs()[i];
    cout << "output " << i << " (channels x width x height): " << output_layer->channels()<< "x" << output_layer->width() << "x" << output_layer->height() << endl;
  }*/

  if ( preprocess == "MEAN_SUBTRACTION" )
  {
    SetMean(preprocess_data);
    preprocess_ = MEAN_SUBTRACTION;
  }
  else if( preprocess == "SCALE_DATA" )
  {
    scale_  = boost::lexical_cast<double>(preprocess_data);
    preprocess_ = SCALE_DATA;
  }
  else
    preprocess_ = NO_OPERATION;
}

/*************************************************/
/* SET MEAN */
void CaffeForward::SetMean(const string& mean_file) {
  /*cv::Mat mean_img = cv::imread(mean_file.c_str(), -1);
  cv::resize(mean_img, mean_, input_geometry_);
  mean_.convertTo(mean_, CV_32FC3);
  return;*/
  /*
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  // Convert from BlobProto to Blob<float>
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
  // The format of the mean file is planar 32-bit float BGR or grayscale.
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    // Extract an individual channel.
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }
  // Merge the separate channels into a single image.
  cv::Mat mean;
  cv::merge(channels, mean);
  cv::imwrite("/home/ali/Codes/Caffe/mean.PNG", mean);
  // Compute the global mean pixel value and create a mean image filled with this value.
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  mean_ = mean;
  cv::imwrite("/home/ali/Codes/Caffe/mean2.PNG", mean_);*/
}

/*************************************************/
/* FORWARD PROCESS */
std::vector <std::vector<float> > CaffeForward::ForwardProcess(std::vector <cv::Mat> & inputs)
{
  for (int i = 0; i < inputs.size(); i++)
  {
    Blob<float>* input_layer = net_->input_blobs()[i];
    input_layer->Reshape(1, input_layer->channels(), input_layer->height(), input_layer->width());
  }
  net_->Reshape();
  for (int i = 0; i < inputs.size(); i++)
  {
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels, i);
    Preprocess(inputs[i], &input_channels, i);
  }

  net_->Forward();

  std::vector<std::vector<float> > outputs;

  for (int i = 0; i < num_outputs_; i++)
  {
    Blob<float>* output_layer = net_->output_blobs()[i];
    // cout << output_layer->width() << "x"<< output_layer->height() << "x" <<output_layer->channels() << endl;
  	const float* begin = output_layer->cpu_data();
  	const float* end = begin + output_layer->channels()*output_layer->width()*output_layer->height();
		outputs.push_back(std::vector<float>(begin, end));
  }
  return outputs;
}

/*************************************************/
/* SAVE CHANNELS */
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
 /*************************************************/
 /* WRAP INPUT LAYER */
void CaffeForward::WrapInputLayer(std::vector<cv::Mat>* input_channels, int input_idx)
{
  Blob<float>* input_layer = net_->input_blobs()[input_idx];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}
/*************************************************/
/* PREPROCESS */
void CaffeForward::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, int input_idx)
{
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_[input_idx] == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_[input_idx] == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_[input_idx] == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_[input_idx] == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_[input_idx])
  {
    cout << "dimension mismatch" << endl;
    cout << "input geometry: " << input_geometry_[input_idx].width << " x "<< input_geometry_[input_idx].height << endl;
    cout << "input data:     " << sample.size().width << " x " << sample.size().height << endl;
    cv::resize(sample, sample_resized, input_geometry_[input_idx]);
  }
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_[input_idx] == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_preprocessed;

  if (preprocess_ == SCALE_DATA)
    sample_preprocessed = (1.0 / 256.0) * sample_float;
  else if(preprocess_ == MEAN_SUBTRACTION)
    cv::subtract(sample_float, mean_, sample_preprocessed);
  else
    sample_preprocessed = sample_float;

  cv::split(sample_preprocessed, *input_channels);
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[input_idx]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

/*************************************************/
/* MAIN */
int main(int argc, char** argv)
{
	::google::InitGoogleLogging(argv[0]);

  string model_file       = argv[1];    // deploy file
  string weight_file      = argv[2];    // weights
  string preprocess       = argv[3];    // preprocess operation
  string preprocess_data  = argv[4];    // preprocess file
  string input_files      = argv[5];    // input files, each separated by ','
  string output_dir       = argv[6];    // output directory to save the files

  std::vector<cv::Mat> inputs;
  string filename;
  std::size_t fname_pos1 = 0;
  std::size_t fname_pos2 = 0;
  bool break_loop = false;
  // open input_files one by one
  while(!break_loop)
  {
    fname_pos2 = input_files.find(",", fname_pos1);
    if(fname_pos2 != string::npos)
      filename = input_files.substr(fname_pos1, fname_pos2-fname_pos1);
    else
    {
      filename = input_files.substr(fname_pos1);
      break_loop = true;
    }
    cout << filename << endl;
    cv::Mat input = ReadInputFile(filename);
    CHECK(!input.empty()) << "Unable to decode input: " << filename;
    inputs.push_back(input);
    fname_pos1 = fname_pos2+1;
  }

  CaffeForward CaffeForward(model_file, weight_file, preprocess, preprocess_data);

  std::vector< std::vector<float> > outputs = CaffeForward.ForwardProcess(inputs);
  CaffeForward.SaveChannels(outputs, output_dir);

  return 0;
}

cv::Mat ReadInputFile(string filename)
{
  if(filename.find(".txt") != string::npos)
    return ReadCVMatFromFile(filename);
  else
    return cv::imread(filename, -1);
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
cv::Mat ReadCVMatFromFile(string filename)
{
  ifstream reader(filename.c_str());
  int rows, cols;
  string line;

  getline(reader, line);
  rows = boost::lexical_cast<int>(line);
  getline(reader, line);
  cols = boost::lexical_cast<int>(line);

  cv::Mat im( cv::Size(rows, cols), CV_32FC1, cv::Scalar(255) );

  for(int i = 0; i < rows; i++)
  {
    for(int j = 0; j < cols; j++)
    {
      getline(reader, line);
      im.at<float>(j, i) = boost::lexical_cast<float>(line);
    }
  }

  reader.close();
  return im;
}
