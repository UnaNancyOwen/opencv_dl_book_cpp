#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    // キャプチャを開く
    cv::VideoCapture capture = cv::VideoCapture( "../face.jpg" ); // 画像ファイル
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // カメラ
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // モデルを読み込む
    //const std::string weights = "../opencv_face_detector.caffemodel"; // float32
    //const std::string config = "../opencv_face_detector.prototxt";
    const std::string weights = "../opencv_face_detector_fp16.caffemodel"; // float16
    const std::string config = "../opencv_face_detector_fp16.prototxt";
    //const std::string weights = "../opencv_face_detector_uint8.pb"; // uint8
    //const std::string config = "../opencv_face_detector_uint8.pbtxt";
    cv::dnn::DetectionModel model = cv::dnn::DetectionModel( weights, config );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0;                                  // スケールファクター
    const cv::Size size = cv::Size( 300, 300 );                // 入力サイズ
    const cv::Scalar mean = cv::Scalar( 104.0, 177.0, 123.0 ); // 差し引かれる平均値
    const bool swap = false;                                   // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = false;                                   // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

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

        // 顔を検出する
        const float confidence_threshold = 0.6;
        const float nms_threshold = 0.4;
        std::vector<int32_t> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        model.detect( image, class_ids, confidences, boxes, confidence_threshold, nms_threshold );

        // 検出した顔のバウンディングボックスを描画する
        for( const cv::Rect& box : boxes ){
            const cv::Scalar color = cv::Scalar( 0, 0, 255 );
            const int32_t thickness = 2;
            cv::rectangle( image, box, color, thickness, cv::LINE_AA );
        }

        // 画像を表示する
        cv::imshow( "face detection", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}