#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <valarray>
#include <iostream>
#include <algorithm>
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

std::vector<float> softmax( const std::vector<float> confidences )
{
    std::valarray<float> exps( confidences.data(), confidences.size() );
    exps = exps - exps.max();
    exps = exps.apply( []( float value ){ return std::exp( value ); } );
    exps = exps / exps.sum();
    return std::vector<float>( std::begin( exps ), std::end( exps ) );
}

int main(int argc, char* argv[])
{
    // キャプチャを開く
    cv::VideoCapture capture = cv::VideoCapture( "../yorkie.jpg" ); // 画像ファイル
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // カメラ
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // モデルを読み込む
    //const std::string weights = "../efficientnet-b0.onnx";
    //const std::string weights = "../efficientnet-b1.onnx";
    //const std::string weights = "../efficientnet-b2.onnx";
    //const std::string weights = "../efficientnet-b3.onnx";
    //const std::string weights = "../efficientnet-b4.onnx";
    //const std::string weights = "../efficientnet-b5.onnx";
    //const std::string weights = "../efficientnet-b6.onnx";
    const std::string weights = "../efficientnet-b7.onnx";
    cv::dnn::ClassificationModel model = cv::dnn::ClassificationModel( weights );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0 / 255.0;                              // スケールファクター
    //const cv::Size size = cv::Size( 224, 224 );                  // 入力サイズ (b0)
    //const cv::Size size = cv::Size( 240, 240 );                  // 入力サイズ (b1)
    //const cv::Size size = cv::Size( 260, 260 );                  // 入力サイズ (b2)
    //const cv::Size size = cv::Size( 300, 300 );                  // 入力サイズ (b3)
    //const cv::Size size = cv::Size( 380, 380 );                  // 入力サイズ (b4)
    //const cv::Size size = cv::Size( 456, 456 );                  // 入力サイズ (b5)
    //const cv::Size size = cv::Size( 528, 528 );                  // 入力サイズ (b6)
    const cv::Size size = cv::Size( 600, 600 );                    // 入力サイズ (b7)
    const cv::Scalar mean = cv::Scalar( 123.675, 116.28, 103.53 ); //差し引かれる平均値
    const bool swap = true;                                        // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = true;                                        // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

    // 後処理のSoftmaxを有効にする（OpenCV 4.6.0以降）
    model.setEnableSoftmaxPostProcessing( true );

    // クラスリストとカラーテーブルを取得する
    const std::string names = "../imagenet.names";
    const std::vector<std::string> classes = read_classes( names );

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

        // クラスに分類して信頼度が最も高いクラスを取得する
        int32_t class_id = 0;
        float confidence = 0.0f;
        model.classify( image, class_id, confidence );

        // 信頼度が最も高いクラスを描画する
        const std::string class_name = classes[class_id];
        const std::string result = cv::format( "%s (%.3f)", class_name.c_str(), confidence);
        const cv::Point point = cv::Point( 30, 30 );
        const int32_t font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.5;
        const cv::Scalar color = cv::Scalar( 255, 255, 255 );
        const int32_t thickness = 1;
        cv::putText( image, result, point, font, scale, color, thickness, cv::LineTypes::LINE_AA );

        /*
        // 推論結果を取得する
        std::vector<cv::Mat> outputs;
        model.predict( image, outputs );

        std::vector<float> output;
        outputs.front().copyTo( output );

        // 必要な場合はSoftmax関数で信頼度を[0.0-1.0]の範囲に変換する
        // ここで扱うEfficientNetの学習済みモデルにはSoftmaxレイヤーが含まれないため必要
        output = softmax( output );

        // 信頼度が上位5個のクラスを取得する
        const int32_t top_n = 5;
        std::vector<size_t> indices( output.size() );
        std::iota( indices.begin(), indices.end(), 0 );
        std::sort( indices.begin(), indices.end(),
            [&output]( int32_t left, int32_t right ) -> bool {
                return output[left] > output[right];
            } );

        std::vector<int32_t> class_ids( indices.begin(), indices.begin() + top_n );
        std::vector<float> confidences( top_n );
        for( int32_t i = 0; i < top_n; i++ ){
            confidences[i] = output[class_ids[i]];
        }

        // 信頼度が上位5個のクラスを表示する
        for( int32_t i = 0; i < top_n; i++ )
        {
            const int32_t class_id = class_ids[i];
            const float confidence = confidences[i];
            const std::string class_name = classes[class_id];
            const std::string result = cv::format( "top-%d %s (%.3f)", i + 1, class_name.c_str(), confidence );
            std::cout << result << std::endl;
        }
        */

        // 画像を表示する
        cv::imshow( "classification", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}