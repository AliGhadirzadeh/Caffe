#include <iostream>
#include <stdlib.h>
#include <caffe/caffe.hpp>
#include <Eigen/Eigen/Dense>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/lexical_cast.hpp>


using namespace caffe;
using namespace std;
using std::string;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/*************************************************/
class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& limit_file,
             const int output_rows);
  std::vector<float> Predict(const cv::Mat& img);
  private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
  std::vector<float> Postprocess(std::vector<float> raw_prediction);
 public:
 	std::vector<float> encoded_data_;
 	int num_outputs_;
  cv::Size output_geometry_;
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
  std::vector<float> min_;
  std::vector<float> max_;
  };
/*************************************************/
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& limit_file,
                       const int output_rows)
{
	Caffe::set_mode(Caffe::CPU);
	/* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  num_outputs_ = net_ ->num_outputs();
	//cout << "net inputs x outputs dimension: " << net_ ->num_inputs() << "x" << num_outputs_ << endl;

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();

  //cout << "net input layer (channels x width x height): " << num_channels_ <<"x"<<input_layer->width()<<"x"<<input_layer->height() << endl;

  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  Blob<float>* output_layer = net_->output_blobs()[0];
  //cout << "net output layer (channels x width x height): " << output_layer->channels()<< "x" << output_layer->width() << "x" << output_layer->height() << endl;
  output_geometry_ = cv::Size(output_layer->channels()/output_rows,output_rows);
  //cout << "output geometry: " <<  output_geometry_.height << " x " << output_geometry_.width << endl;

  ifstream file;
  file.open(limit_file.c_str());
  string line;
  if (file.is_open())
  {
    while(std::getline(file, line))
    {
        min_.push_back(boost::lexical_cast<float>(line));
        std::getline(file, line);
        max_.push_back(boost::lexical_cast<float>(line));
    }
  }
  else
  {
    std::cout << "Error - could not open limit_file" << endl;
  }
  file.close();
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

  std::vector<float> predictions = Postprocess(std::vector<float>(begin, end));
  return predictions;
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
  //cout << "image channels: " << img.channels() << endl;
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

	//cout << "sample channels: " << sample.channels() << endl;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
  {
    cout << "input does not match the network geometry" << endl;
    cv::resize(sample, sample_resized, input_geometry_);
  }
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  //sample_normalized = (1.0 / 256.0) * sample_float;
  sample_normalized = (1.0) * sample_float;

  //cout << sample_normalized << endl;

  //cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */

  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

}

std::vector<float> Classifier::Postprocess(std::vector<float> raw_prediction)
{
  std::vector<float> predictions;
  int k;
  for (int j = 0; j < output_geometry_.height; j++)
  {
    for(int i = 0; i < output_geometry_.width; i++)
    {
      k = i + output_geometry_.width;
      float scaled_data;
      scaled_data = raw_prediction[i + output_geometry_.width*j] * (max_[k] - min_[k]) + min_[k];
      predictions.push_back(scaled_data);
    }
  }
  return predictions;
}


/*************************************************/
void Encode(const string& filename_root, Classifier& classifier, int n_files );
vector<vector<int> > GetIndecisFromFile(const string& filename);
MatrixXd Reconstruct(cv::Mat img, Classifier& classifier);
void WriteMatrixToFile(string fileName, MatrixXd data);
MatrixXd ReadMatrixFromFile(string fileName);
void ReconstructActionGroups(Classifier& classifier);
void MakeDescriminatorDatabase(Classifier& classifier);
void Encode2ImageFile(Classifier& classifier, const string& input_image_filename, const string & output_image_filename);
void Random2ImageFile(int dim, const string & output_image_filename);
void CreateTextFileName(const string & output_filename,const string & data_root_file, const int n_neg_samples, const int n_pos_samples);
void calcReconstructError(Classifier& classifier, string data_file, int n_data);
void Decoder(Classifier& classifier, MatrixXd data);

int main(int argc, char** argv)
{
	std::cout << "Caffe action_autoencoder is started" << std::endl;
  srand (time(NULL));
	::google::InitGoogleLogging(argv[0]);
  const int action_rows = 20;

  const bool calc_reconstruct_err = false;
  const bool decode_input_data = true;
  const bool encode_image_data = false;

  string model_file;

  if (calc_reconstruct_err || encode_image_data)
    model_file =  "models/action_vae/train.prototxt";
  else if (decode_input_data)
    model_file =  "models/action_vae/decoder.prototxt";

  const string trained_file = "models/action_vae/snapshot_reach/vae_iter_44534.caffemodel";
  const string limit_file = "data/reach_action/data_limits.txt";
  const string actimg_file = "data/reach_action/img";
  const string data_file = "data/reach_action";
  /*
  const string trained_file = "models/action_vae/snapshot_throw/_iter_32329.caffemodel";
  const string limit_file = "data/throw_action/data_limits.txt";
  const string actimg_file = "data/throw_action/img";
  const string data_file = "data/throw_action";
  */

  Classifier classifier(model_file, trained_file, limit_file, action_rows);
  if (calc_reconstruct_err)
    calcReconstructError(classifier, data_file, 5501);
  else if(decode_input_data)
  {
    MatrixXd enc_samples = ReadMatrixFromFile("actions.txt");
    cout << enc_samples << endl;
    Decoder(classifier, enc_samples);
  }
  else if(encode_image_data)
    Encode(actimg_file, classifier, 5501 );

	return 0;
}

// Decode the encoded 'data'
// each row of data corresponds to one encoded sample
void Decoder(Classifier& classifier, MatrixXd data)
{
  int n_samples = data.rows();
  cv::Mat m;
  MatrixXd rowMat;
  char save_filename[100];
  for(int i = 0; i < n_samples; i++)
  {
    rowMat = data.block(i,0,1,data.cols());
    eigen2cv(rowMat, m);
    //cout << m << endl;
    MatrixXd cmd = Reconstruct(m, classifier);
    //cout << cmd << endl << endl << endl;
    sprintf(save_filename, "commands/%04d.txt", i);
    WriteMatrixToFile(save_filename, cmd);
  }
}

void ReconstructActionGroups(Classifier& classifier)
{
  const string ind_filename = "results/reconst_actions/ind.txt";
  const string action_rootfile = "data/toss_action/image0/";
  const string result_rootfile = "results/reconst_actions/";
  char img_filename[100];
  char res_filename[100];
  vector< vector < int > > indecis =  GetIndecisFromFile(ind_filename);

  for(int i = 0; i < indecis.size(); i++)
  {
    for(int j = 0; j < indecis[i].size(); j++)
    {
      sprintf(img_filename, "%sact_img%04d.jpg", action_rootfile.c_str(), (indecis[i][j] - 1));
      cv::Mat img = cv::imread(img_filename, -1);
      CHECK(!img.empty()) << "Unable to decode image " << img_filename;
      MatrixXd reconst_action = Reconstruct(img, classifier);
      sprintf(res_filename, "%s%d/%04d.txt", result_rootfile.c_str(), i+1, j);
      WriteMatrixToFile(res_filename, reconst_action);
    }
  }
}

// This function calculates reconstruction error for all data in a folder
void calcReconstructError(Classifier& classifier, string data_file, int n_data)
{
  int cols = classifier.output_geometry_.width;
  int rows = classifier.output_geometry_.height;
  char file[100];
  MatrixXd rec_act, org_act;
  RowVectorXd err = RowVectorXd::Zero(cols);
  RowVectorXd max_err = RowVectorXd::Zero(cols);
  for(int n = 0; n < n_data; n++)
  {
    // finding the original action
    sprintf(file, "%s/raw/act_log%04d.txt", data_file.c_str(), n);
    org_act = ReadMatrixFromFile(file);
    org_act = org_act.block(0,7,20,7);
    // reconstructing the action
    sprintf(file, "%s/img/act_img%04d.png", data_file.c_str(), n);
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    rec_act = Reconstruct(img, classifier);
    double col_err;
    for(int j = 0; j < cols; j++)
    {
      VectorXd diff = org_act.col(j) - rec_act.col(j);
      col_err = diff.cwiseAbs().sum()/rows;
      err(j) += col_err;
      if ( col_err > max_err(j) )
        max_err(j) = col_err;
    }
  }
  err = err / n_data;
  cout << err <<endl;
  cout << max_err <<endl;
}

MatrixXd Reconstruct(cv::Mat img, Classifier& classifier)
{
  int cols = classifier.output_geometry_.width;
  int rows = classifier.output_geometry_.height;

  MatrixXd reconst_out(rows, cols);
  std::vector<float> pred = classifier.Predict(img);

  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      reconst_out(i,j) = pred[j + i*cols];

  return reconst_out;
}

// This function opens act_imgs in a directory and save the encoded space
// into a file
void Encode(const string& filename_root, Classifier& classifier, int n_files )
{
  char filename[100];
  ofstream file_writer;
  file_writer.open("enc_data.txt");
  if(file_writer.is_open())
  {
    file_writer << n_files << endl;
    for(int i = 0; i < n_files; i++)
    {
      sprintf(filename, "%s/act_img%04d.png",filename_root.c_str(), i);
      cv::Mat img = cv::imread(filename, -1);
      CHECK(!img.empty()) << "Unable to decode image " << filename;
      classifier.Predict(img);
      // displaying the encoded data
      cout << "i = " << i << " enc_data = " ;
      for (int j = 0; j < classifier.encoded_data_.size(); j++)
        cout << classifier.encoded_data_[j]<< "\t\t\t";
      cout << endl;
      // saving the data
      for (int j = 0; j < classifier.encoded_data_.size(); j++)
        file_writer << classifier.encoded_data_[j] << endl;
    }
    file_writer.close();
  }
  else
    cout << "Error - the file_writer could not be opened. " << endl;
}

void MakeDescriminatorDatabase(Classifier& classifier)
{
  char in_filename[100];
  char out_filename[100];
  const string act_img_root_filename = "data/toss_action/image0";
  const string enc_img_root_filename = "data/enc_toss_action/images";
  const string labels_filename = "data/enc_toss_action/train.txt";
  const int n_pos_samples = 2200;      // randomly distributed numbers
  const int n_neg_samples = 2200;      // encoded data
  const int dim_enc_space = 3;

  // generating the encoded data
  for(int i = 0; i < n_neg_samples; i++)
  {
    sprintf(in_filename, "%s/act_img%04d.jpg",act_img_root_filename.c_str(), i);
    sprintf(out_filename, "%s/enc%04d.PNG",enc_img_root_filename.c_str(), i);
    Encode2ImageFile(classifier, in_filename, out_filename);
  }
  // generating random data
  for(int i = 0; i < n_pos_samples; i++)
  {
    sprintf(out_filename, "%s/enc%04d.PNG",enc_img_root_filename.c_str(), i+n_neg_samples);
    Random2ImageFile(dim_enc_space, out_filename);
  }
  // saving the text file indicating filenames and labels
  CreateTextFileName(labels_filename, enc_img_root_filename, n_neg_samples, n_pos_samples);
}

void Encode2ImageFile(Classifier& classifier, const string& input_image_filename, const string & output_image_filename)
{
  cv::Mat input_image = cv::imread(input_image_filename, -1);
  CHECK(!input_image.empty()) << "Unable to decode image " << input_image_filename;
  classifier.Predict(input_image);
  cv::Mat enc_image = cv::Mat(classifier.encoded_data_);
  enc_image.convertTo(enc_image, CV_8UC1, 255.0);
	imwrite(output_image_filename, enc_image);
}
void Random2ImageFile(int dim, const string & output_image_filename)
{
  std::vector<float> v;
  float r;
  for (int i = 0; i < dim; i++)
  {
    r = (rand()%1000+1)/1000.0;
    v.push_back(r);
  }
  cv::Mat rand_image = cv::Mat(v);
  rand_image.convertTo(rand_image, CV_8UC1, 255.0);
	imwrite(output_image_filename, rand_image);
}
void CreateTextFileName(const string & output_filename,const string & data_root_file, const int n_neg_samples, const int n_pos_samples)
{
  ofstream file_writer;
  file_writer.open(output_filename.c_str());
  char line[100];
  for(int i = 0; i < n_neg_samples; i++)
  {
    sprintf(line, "%s/enc%04d.PNG 0\n",data_root_file.c_str(), i);
    file_writer << line;
  }
  for(int i = 0; i < n_pos_samples; i++)
  {
    sprintf(line, "%s/enc%04d.PNG 1\n",data_root_file.c_str(), i + n_neg_samples);
    file_writer << line;
  }
  file_writer.close();
}

vector<vector<int> > GetIndecisFromFile(const string& filename)
{
  vector<vector<int> > indecis;
  ifstream fileReader;
  fileReader.open(filename.c_str());

  string line;
  int n_data;
  int data;
  if (fileReader.is_open())
  {
    while(getline(fileReader, line))
    {
      vector<int> row;
      n_data = boost::lexical_cast<int>(line);
      for(int i=0; i < n_data; i++)
      {
        getline(fileReader, line);
        data = boost::lexical_cast<int>(line);
        row.push_back(data);
      }
      indecis.push_back(row);
    }
  }
  else
    cout << "Error - could not open the file " << filename << endl;
  return indecis;
}

void WriteMatrixToFile(string fileName, MatrixXd data)
{
    int nRow = data.rows();
    int nCol = data.cols();
    ofstream fileWriter;
    fileWriter.open(fileName.c_str());

    fileWriter << nRow << endl;
    fileWriter << nCol << endl;

    for(int i = 0; i < nRow; i++)
        {
            for(int j = 0; j < nCol; j++)
                {
                    fileWriter << data(i,j) << endl;
                }
        }

    fileWriter.close();
}
MatrixXd ReadMatrixFromFile(string fileName)
{
    MatrixXd data;
    ifstream fileReader;
    fileReader.open(fileName.c_str());
    int nRow, nCol;
    string line;

    if (fileReader.is_open())
        {

            getline(fileReader, line);
            nRow = boost::lexical_cast<int>(line);
            getline(fileReader, line);
            nCol = boost::lexical_cast<int>(line);
            data.resize(nRow, nCol);
            for(int i = 0; i < nRow; i++)
                {
                    for(int j = 0; j < nCol; j++)
                        {
                            getline(fileReader, line);
                            data(i,j) = boost::lexical_cast<double>(line);
                        }
                }
        }

    fileReader.close();
    return data;
}
