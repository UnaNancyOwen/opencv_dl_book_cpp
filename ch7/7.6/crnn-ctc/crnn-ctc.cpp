#include <vector>
#include <string>
#include <fstream>
#include <iostream>
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

class text_recognizer
{
private:
    cv::dnn::TextRecognitionModel model;
    bool require_gray = false;

public:
    // コンストラクタ
    text_recognizer()
    {
        init_model();
    }

private:
    // モデルを準備する
    void init_model()
    {
        // モデルを読み込む
        //const std::string weights = "../crnn.onnx";       // 英語, 数字
        const std::string weights = "../crnn_cs.onnx";      // 英語, 数字, 記号
        //const std::string weights = "../crnn_cs_CN.onnx"; // 英語, 中国語, 数字, 記号
        model = cv::dnn::TextRecognitionModel( weights );

        // モデルの推論に使用するエンジンとデバイスを設定する
        model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
        model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

        // グレースケール画像を要求する（CRNNのみ）
        if( weights == "crnn.onnx" ){
            require_gray = true;
        }

        // モデルの入力パラメーターを設定する
        const double scale = 1.0 / 127.5;                          // スケールファクター
        const cv::Size size = cv::Size( 100, 32 );                 // 入力サイズ
        const cv::Scalar mean = cv::Scalar( 127.5, 127.5, 127.5 ); //差し引かれる平均値
        const bool swap = true;                                    // チャンネルの順番（True: RGB、False: BGR）
        const bool crop = false;                                   // クロップ
        model.setInputParams( scale, size, mean, swap, crop );

        // デコードタイプを設定する
        const std::string type = "CTC-greedy";               // 貪欲法
        //const std::string type = "CTC-prefix-beam-search"; // ビーム探索
        model.setDecodeType( type );

        // 語彙リストを設定する
        //const std::string vocabulary_file = "../alphabet_36.txt";   // 英語, 数字
        const std::string vocabulary_file = "../alphabet_94.txt";     // 英語, 数字, 記号
        //const std::string vocabulary_file = "../alphabet_3944.txt"; // 英語, 中国語, 数字, 記号
        const std::vector<std::string> vocabularies = read_vocabularies( vocabulary_file );
        model.setVocabulary( vocabularies );
    }

    // ファイルから語彙リストを読み込む
    std::vector<std::string> read_vocabularies( const std::string file )
    {
        std::vector<std::string> vocabularies;
        std::string line;
        std::ifstream ifs( file );
        while( std::getline( ifs, line ) ){
            vocabularies.push_back( line );
        }
        return vocabularies;
    }

public:
    // 画像からテキストを認識する
    std::string recognize( cv::Mat& image )
    {
        if( model.getNetwork_().empty() )
        {
            throw std::runtime_error( "failed model has not been created!" );
        }

        if( image.empty() )
        {
            throw std::runtime_error( "failed image is empty!" );
        }

        // グレースケール画像に変換する
        if( require_gray && image.channels() != 1 ){
            cv::cvtColor( image, image, cv::COLOR_BGR2GRAY );
        }

        // テキストを認識する
        std::string text = model.recognize( image );

        return text;
    }
};

// 画像とテキスト領域の座標リストからテキスト領域の画像を切り出す関数
std::vector<cv::Mat> get_text_images( const cv::Mat& image, const std::vector<std::vector<cv::Point>> vertices )
{
    std::vector<cv::Mat> text_images;
    const cv::Size size = cv::Size( 100, 32 );

    for( const std::vector<cv::Point>& vertex : vertices ){
        const cv::Point2f source_poins[4] = { cv::Point2f( vertex[0].x, vertex[0].y ), cv::Point2f( vertex[1].x, vertex[1].y ), 
                                              cv::Point2f( vertex[2].x, vertex[2].y ), cv::Point2f( vertex[3].x, vertex[3].y ) };
        const cv::Point2f target_poins[4] = { cv::Point2f( 0, size.height ), cv::Point2f( 0, 0 ),
                                              cv::Point2f( size.width, 0 ), cv::Point2f( size.width, size.height ) };
        const cv::Mat transform_matrix = cv::getPerspectiveTransform( source_poins, target_poins );

        cv::Mat text_image;
        cv::warpPerspective( image, text_image, transform_matrix, size );

        text_images.push_back( text_image );
    }

    return text_images;
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

    // テキスト認識器の生成
    text_recognizer recognizer = text_recognizer();

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

        // テキスト領域の画像を切り出す
        std::vector<cv::Mat> text_images = get_text_images( image, vertices );

        /*
        // テキスト領域の画像を保存する
        for( int32_t i = 0; i < text_images.size(); i++ ){
            std::string file_name = cv::format( "text_%03d.jpg", i );
            cv::imwrite( file_name, text_images[i] );
        }
        */

        // テキストを認識する
        std::vector<std::string> texts;
        for( cv::Mat& text_image : text_images ){
            const std::string text = recognizer.recognize( text_image );
            texts.push_back( text );
        }

        // 検出したテキスト領域の矩形を描画する
        for( std::vector<cv::Point>& vertex : vertices ){
            const bool close = true;
            const cv::Scalar color = cv::Scalar( 0, 255, 0 );
            const int32_t thickness = 2;
            cv::polylines( image, vertex, close, color, thickness, cv::LineTypes::LINE_AA );
        }

        // テキスト認識の結果を描画する
        for( int32_t i = 0; i < texts.size(); i++ ){
            const std::string text = texts[i];
            const std::vector<cv::Point> vertex = vertices[i];

            const cv::Point position = vertex[1] - cv::Point( 0, 10 );
            const int32_t font = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
            const double scale = 0.5;
            const cv::Scalar color = cv::Scalar( 255, 255, 255 );
            const int32_t thickness = 1;
            cv::putText( image, text, position, font, scale, color, thickness, cv::LineTypes::LINE_AA );

            // OpenCVのcv2.putText()では中国語（漢字）は描画できないので標準出力に表示する
            std::cout << text << std::endl;
        }

        // 画像を表示する
        cv::imshow( "text recognition", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}