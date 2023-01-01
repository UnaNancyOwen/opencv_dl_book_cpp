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

    // 分類器を読み込む
    const std::string path = "../haarcascade_frontalface_default.xml";
    cv::CascadeClassifier cascade = cv::CascadeClassifier( path );
    if( cascade.empty() ){
        throw std::runtime_error( "can't read cascade!" );
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

        // グレースケールに変換する
        cv::Mat gray_image;
        cv::cvtColor( image, gray_image, cv::COLOR_BGR2GRAY );

        // 顔を検出する
        const int32_t width = gray_image.cols;
        const int32_t height = gray_image.rows;
        const cv::Size min_size = cv::Size( width / 10, height / 10 );
        std::vector<cv::Rect> boxes;
        cascade.detectMultiScale( gray_image, boxes, 1.1, 3, 0, min_size );

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