#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

// ファイルからクラスの名前のリストを読み込む関数
std::vector<std::string> read_classes( const std::string file )
{
    std::vector<std::string> classes;
    std::string line;
    std::ifstream ifs( file );
    while( std::getline( ifs, line ) ){
        classes.push_back( line );
    }
    return classes;
}

// クラスの数だけカラーテーブルを生成する関数
std::vector<cv::Scalar> get_colors( const int32_t num )
{
    std::vector<cv::Scalar> colors;
    cv::RNG random( 0 );
    for( int32_t i = 0; i < num; i++ ){
        const cv::Scalar color = cv::Scalar( random.uniform( 0, 255 ), random.uniform( 0, 255 ), random.uniform( 0, 255 ) );
        colors.push_back( color );
    }
    return colors;
}

int main(int argc, char* argv[])
{
    // キャプチャを開く
    cv::VideoCapture capture = cv::VideoCapture( "../bicycle.jpg" ); // 画像ファイル（自転車）
    //cv::VideoCapture capture = cv::VideoCapture( "../city.jpg" ); // 画像ファイル（道路）
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // モデルを読み込む
    const std::string weights = "../optimized_graph_voc.pb"; // VOC
    //const std::string weights = "../optimized_graph_cityscapes.pb"; // Cityscapes
    cv::dnn::SegmentationModel model = cv::dnn::SegmentationModel( weights );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0 / 127.5;                          // スケールファクター
    const cv::Size size = cv::Size( 513, 513 );                // 入力サイズ（VOC）
    //const cv::Size size = cv::Size( 2049, 1025 );            // 入力サイズ（CityScapes）
    const cv::Scalar mean = cv::Scalar( 127.5, 127.5, 127.5 ); // 差し引かれる平均値
    const bool swap = true;                                    // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = false;                                   // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

    // クラスリストとカラーテーブルを取得する
    const std::string names = "../voc.names"; // VOC
    //const std::string names = "../cityscapes.names"; // CityScapes
    const std::vector<std::string> classes = read_classes( names );
    std::vector<cv::Scalar> colors = get_colors( classes.size() );
    if( names == "voc.names" ){
        colors[0] = cv::Scalar( 0, 0, 0 );
    }

    while( true ){
        // フレームをキャプチャして画像を読み込む
        cv::Mat image;
        capture >> image;
        if( image.empty() ){
            cv::waitKey( 0 );
            break;
        }

        // 画像が3チャンネル以外の場合は3チャンネルに変換する
        if( image.channels() == 1 ){
            cv::cvtColor( image, image, cv::COLOR_GRAY2BGR );
        }
        if( image.channels() == 4 ){
            cv::cvtColor( image, image, cv::COLOR_BGRA2BGR );
        }

        // セグメンテーションしてマスクを取得する
        cv::Mat mask;
        model.segment( image, mask );

        // カラーテーブルを参照してマスクに色を付ける
        cv::Mat color_mask = cv::Mat( mask.rows, mask.cols, CV_8UC3 );
        #pragma omp parallel for
        for( int32_t i = 0; i < mask.total(); i++ ){
            const uint8_t class_id = mask.at<uint8_t>( i );
            const cv::Scalar color = colors[class_id];
            color_mask.at<cv::Vec3b>( i ) = cv::Vec3b( color[0], color[1], color[2] );
        }

        // マスクを入力画像と同じサイズに拡大する
        cv::resize( color_mask, color_mask, cv::Size( image.cols, image.rows ), 0.0, 0.0, cv::INTER_LINEAR );

        // 画像とマスクをアルファブレンドする
        const double alpha = 0.5;
        const double beta = 1.0 - alpha;
        cv::addWeighted( image, alpha, color_mask, beta, 0.0, image );

        // 画像を表示する
        cv::imshow( "segmentation", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}