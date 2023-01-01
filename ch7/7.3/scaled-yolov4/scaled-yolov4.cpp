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
    cv::VideoCapture capture = cv::VideoCapture( "../dog.jpg" ); // 画像ファイル
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // カメラ
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // モデルを読み込む
    //const std::string weights = "../yolov4-csp.weights"; // YOLOv4-csp (512x512, 640x640)
    //const std::string config = "../yolov4-csp.cfg";
    const std::string weights = "../yolov4x-mish.weights"; // YOLOv4x-mish (640x640)
    const std::string config = "../yolov4x-mish.cfg";
    //const std::string weights = "../yolov4-p5.weights"; // YOLOv4-P5 (896x896)
    //const std::string config = "../yolov4-p5.cfg";
    //const std::string weights = "../yolov4-p6.weights"; // YOLOv4-P6 (1280x1280)
    //const std::string config = "../yolov4-p6.cfg";
    cv::dnn::DetectionModel model = cv::dnn::DetectionModel( weights, config );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0 / 255.0;                    // スケールファクター
    //const cv::Size size = cv::Size( 512, 512 );        // 入力サイズ（YOLOv4-csp）
    const cv::Size size = cv::Size( 640, 640 );          // 入力サイズ（YOLOv4-csp、YOLOv4x-mish）
    //const cv::Size size = cv::Size( 896, 896 );        // 入力サイズ（YOLOv4-P5）
    //const cv::Size size = cv::Size( 1280, 1280 );      // 入力サイズ（YOLOv4-P6）
    const cv::Scalar mean = cv::Scalar( 0.0, 0.0, 0.0 ); //差し引かれる平均値
    const bool swap = true;                              // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = false;                             // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

    // NMS（Non - Maximum Suppression）をクラスごとに処理する
    model.setNmsAcrossClasses( false ); // （True: 全体、False: クラスごと）

    // クラスリストとカラーテーブルを取得する
    const std::string names = "../coco.names";
    const std::vector<std::string> classes = read_classes( names );
    const std::vector<cv::Scalar> colors = get_colors( classes.size() );

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

        // オブジェクトを検出する
        const float confidence_threshold = 0.5f;
        const float nms_threshold = 0.4f;
        std::vector<int32_t> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        model.detect( image, class_ids, confidences, boxes, confidence_threshold, nms_threshold );

        // 検出されたオブジェクトを描画する
        for( int32_t i = 0; i < boxes.size(); i++ ){
            const int32_t class_id = class_ids[i];
            const float confidence = confidences[i];
            const cv::Rect box = boxes[i];

            const std::string class_name = classes[class_id];
            const cv::Scalar color = colors[class_id];
            const int32_t thickness = 2;
            cv::rectangle( image, box, color, thickness, cv::LineTypes::LINE_AA );

            const std::string result = cv::format( "%s (%.3f)", class_name.c_str(), confidence );
            const cv::Point point = cv::Point( box.tl().x, box.tl().y - 5 );
            const int32_t font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
            const double scale = 0.5;
            cv::putText( image, result, point, font, scale, color, thickness, cv::LineTypes::LINE_AA );
        }

        // 画像を表示する
        cv::imshow( "object detection", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}