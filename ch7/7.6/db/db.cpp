#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

class text_detector
{
private:
     cv::dnn::TextDetectionModel_DB model;

public:
    // コンストラクタ
    text_detector()
    {
        init_model();
    }

private:
    // モデルを準備する
    void init_model()
    {
        // モデルを読み込む
        //const std::string weights = "../DB_TD500_resnet50.onnx"; // 英語, 中国語, 数字
        //const std::string weights = "../DB_TD500_resnet18.onnx"; // 英語, 中国語, 数字
        const std::string weights = "../DB_IC15_resnet50.onnx";    // 英語, 数字
        //const std::string weights = "../DB_IC15_resnet18.onnx";  // 英語, 数字
        model = cv::dnn::TextDetectionModel_DB( weights );

        // モデルの推論に使用するエンジンとデバイスを設定する
        model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
        model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

        // モデルの入力パラメーターを設定する
        const double scale = 1.0 / 255.0;                                               // スケールファクター
        const cv::Size size = cv::Size( 736, 736 );                                     // 入力サイズ（MSRA-TD500）
        //const cv::Size size = cv::Size( 736, 1280 );                                  // 入力サイズ（ICDAR2015）
        const cv::Scalar mean = cv::Scalar( 122.67891434, 116.66876762, 104.00698793 ); //差し引かれる平均値
        const bool swap = false;                                                        // チャンネルの順番（True: RGB、False: BGR）
        const bool crop = false;                                                        // クロップ
        model.setInputParams( scale, size, mean, swap, crop );

        // テキスト検出のパラメーターを設定する
        const float binary_threshold = 0.3f;  // 二値化の閾値
        const float polygon_threshold = 0.5f; // テキスト輪郭スコアの閾値
        const int32_t max_candidates = 200;   // テキスト候補領域の上限値
        const double unclip_ratio = 2.0;      // アンクリップ率
        model.setBinaryThreshold( binary_threshold );
        model.setPolygonThreshold( polygon_threshold );
        model.setMaxCandidates( max_candidates );
        model.setUnclipRatio( unclip_ratio );
    }

public:
    // 画像からテキストを検出する（座標）
    void detect_vertices( const cv::Mat& image, std::vector<std::vector<cv::Point>>& vertices, std::vector<float>& confidencies )
    {
        if( model.getNetwork_().empty() )
        {
            throw std::runtime_error( "failed model has not been created!" );
        }

        if( image.empty() )
        {
            throw std::runtime_error( "failed image is empty!" );
        }

        model.detect( image, vertices, confidencies );
    }

    // 画像からテキストを検出する（中心座標、領域サイズ、回転角度）
    void detect_rotated_rectangles( const cv::Mat& image, std::vector<cv::RotatedRect>& rotated_rectangles, std::vector<float>& confidencies )
    {
        if( model.getNetwork_().empty() )
        {
            throw std::runtime_error( "failed model has not been created!" );
        }

        if( image.empty() )
        {
            throw std::runtime_error( "failed image is empty!" );
        }

        model.detectTextRectangles( image, rotated_rectangles, confidencies );
    }
};

// 回転矩形から矩形四隅の頂点座標（左下から時計回り）を取得する
std::vector<std::vector<cv::Point>> get_vertices( const std::vector<cv::RotatedRect>& rotated_rectangles )
{
    std::vector<std::vector<cv::Point>> vertices;
    for( const cv::RotatedRect& rotated_rectangle : rotated_rectangles ){
        cv::Mat points;
        cv::boxPoints( rotated_rectangle, points );
        std::vector<cv::Point> vertex;
        vertex.push_back( cv::Point( static_cast<int32_t>( points.at<float>( 0, 0 ) ), static_cast<int32_t>( points.at<float>( 0, 1 ) ) ) );
        vertex.push_back( cv::Point( static_cast<int32_t>( points.at<float>( 1, 0 ) ), static_cast<int32_t>( points.at<float>( 1, 1 ) ) ) );
        vertex.push_back( cv::Point( static_cast<int32_t>( points.at<float>( 2, 0 ) ), static_cast<int32_t>( points.at<float>( 2, 1 ) ) ) );
        vertex.push_back( cv::Point( static_cast<int32_t>( points.at<float>( 3, 0 ) ), static_cast<int32_t>( points.at<float>( 3, 1 ) ) ) );
        vertices.push_back( vertex );
    }
    return vertices;
}

int main(int argc, char* argv[])
{
    // キャプチャを開く
    cv::VideoCapture capture = cv::VideoCapture( "../text.jpg" ); // 画像ファイル
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // カメラ
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // テキスト検出器の生成
    text_detector detector = text_detector();

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

        // テキスト検出（座標）
        std::vector<std::vector<cv::Point>> vertices;
        std::vector<float> confidencies;
        detector.detect_vertices( image, vertices, confidencies );

        /*
        // テキスト検出（中心座標、領域サイズ、回転角度）
        std::vector<cv::RotatedRect> rotated_rectangles;
        std::vector<float> confidencies;
        detector.detect_rotated_rectangles( image, rotated_rectangles, confidencies );
        std::vector<std::vector<cv::Point>> vertices = get_vertices( rotated_rectangles ); // テキスト検出（座標）と同じ
        */

        // 検出したテキスト領域の矩形を描画する
        for( std::vector<cv::Point>& vertex : vertices ){
            const bool close = true;
            const cv::Scalar color = cv::Scalar( 0, 255, 0 );
            const int32_t thickness = 2;
            cv::polylines( image, vertex, close, color, thickness, cv::LineTypes::LINE_AA );
        }

        // 画像を表示する
        cv::imshow( "text detection", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}