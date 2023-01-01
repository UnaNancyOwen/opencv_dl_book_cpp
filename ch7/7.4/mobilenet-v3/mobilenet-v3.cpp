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
    const std::string weights = "../v3-large_224_1.0_float.pb"; // Large
    //const std::string weights = "../v3-small_224_1.0_float.pb"; // Small
    cv::dnn::ClassificationModel model = cv::dnn::ClassificationModel( weights );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0 / 127.5;                          // スケールファクター
    const cv::Size size = cv::Size( 224, 224 );                // 入力サイズ
    const cv::Scalar mean = cv::Scalar( 127.5, 127.5, 127.5 ); // 差し引かれる平均値
    const bool swap = true;                                    // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = true;                                    // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

    // 後処理のSoftmaxを無効にする（OpenCV 4.6.0以降）
    model.setEnableSoftmaxPostProcessing( false );

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
        const std::string class_name = classes[class_id - 1];
        const std::string result = cv::format( "%s (%.3f)", class_name.c_str(), confidence);
        const cv::Point point = cv::Point( 30, 30 );
        const int32_t font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.5;
        const cv::Scalar color = cv::Scalar( 255, 255, 255 );
        const int32_t thickness = 1;
        cv::putText( image, result, point, font, scale, color, thickness, cv::LINE_AA );

        /*
        // 推論結果を取得する
        std::vector<cv::Mat> outputs;
        model.predict( image, outputs );

        std::vector<float> output;
        outputs.front().copyTo( output );

        // 必要な場合はSoftmax関数で信頼度を[0.0-1.0]の範囲に変換する
        // ここで扱うMobileNet v3の学習済みモデルにはSoftMaxレイヤーが含まれるため不要
        //output = softmax( output );

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
            const int32_t class_id = class_ids[i] - 1;
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