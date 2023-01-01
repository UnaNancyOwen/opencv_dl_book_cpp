#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

// ジョイントタイプ（関節の名前）
enum class joints : int32_t
{
    NOSE,           // 0
    SPINE_SHOULDER, // 1
    SHOULDER_RIGHT, // 2
    ELBOW_RIGHT,    // 3
    HAND_RIGHT,     // 4
    SHOULDER_LEFT,  // 5
    ELBOW_LEFT,     // 6
    HAND_LEFT,      // 7
    HIP_RIGHT,      // 8
    KNEE_RIGHT,     // 9
    FOOT_RIGHT,     // 10
    HIP_LEFT,       // 11
    KNEE_LEFT,      // 12
    FOOT_LEFT,      // 13
    EYE_RIGHT,      // 14
    EYE_LEFT,       // 15
    EAR_RIGHT,      // 16
    EAR_LEFT        // 17
};

// ボーンリスト（関節の接続関係）
std::vector<std::pair<joints, joints>> bones = {
    { joints::SPINE_SHOULDER, joints::SHOULDER_RIGHT },
    { joints::SPINE_SHOULDER, joints::SHOULDER_LEFT  },
    { joints::SHOULDER_RIGHT, joints::ELBOW_RIGHT    },
    { joints::ELBOW_RIGHT,    joints::HAND_RIGHT     },
    { joints::SHOULDER_LEFT,  joints::ELBOW_LEFT     },
    { joints::ELBOW_LEFT,     joints::HAND_LEFT      },
    { joints::SPINE_SHOULDER, joints::HIP_RIGHT      },
    { joints::HIP_RIGHT,      joints::KNEE_RIGHT     },
    { joints::KNEE_RIGHT,     joints::FOOT_RIGHT     },
    { joints::SPINE_SHOULDER, joints::HIP_LEFT       },
    { joints::HIP_LEFT,       joints::KNEE_LEFT      },
    { joints::KNEE_LEFT,      joints::FOOT_LEFT      },
    { joints::SPINE_SHOULDER, joints::NOSE           },
    { joints::NOSE,           joints::EYE_RIGHT      },
    { joints::EYE_RIGHT,      joints::EAR_RIGHT      },
    { joints::NOSE,           joints::EYE_LEFT       },
    { joints::EYE_LEFT,       joints::EAR_LEFT       }
};

// カラーテーブルを生成する関数
std::vector<cv::Scalar> get_colors()
{
    std::vector<cv::Scalar> colors = {
        cv::Scalar( 255, 0, 0 ), cv::Scalar( 255, 85, 0 ), cv::Scalar( 255, 170, 0 ), cv::Scalar( 255, 255, 0 ), cv::Scalar( 170, 255, 0 ),
        cv::Scalar( 85, 255, 0 ), cv::Scalar( 0, 255, 0 ), cv::Scalar( 0, 255, 85 ), cv::Scalar( 0, 255, 170 ), cv::Scalar( 0, 255, 255 ),
        cv::Scalar( 0, 170, 255 ), cv::Scalar( 0, 85, 255 ), cv::Scalar( 0, 0, 255 ), cv::Scalar( 85, 0, 255 ), cv::Scalar( 170, 0, 255 ),
        cv::Scalar( 255, 0, 255 ), cv::Scalar( 255, 0, 170 ), cv::Scalar( 255, 0, 85 ) };
    return colors;
}

// ボーンを描画する関数
void draw_bone( cv::Mat& image, const cv::Point start_point, const cv::Point end_point, const cv::Scalar color, int32_t thickness = 4 )
{
    const int32_t mean_x = static_cast<int32_t>( ( start_point.x + end_point.x ) / 2.0f );
    const int32_t mean_y = static_cast<int32_t>( ( start_point.y + end_point.y ) / 2.0f );
    const cv::Point center = cv::Point( mean_x, mean_y );
    const cv::Point diff = start_point - end_point;
    const double length = std::sqrt( diff.x * diff.x + diff.y * diff.y );
    const cv::Size axes = cv::Size( static_cast<int32_t>( length / 2.0 ), thickness );
    const int32_t angle = static_cast<int32_t>( std::atan2( diff.y, diff.x ) * 180.0 / CV_PI );
    std::vector<cv::Point> polygon;
    cv::ellipse2Poly( center, axes, angle, 0, 360, 1, polygon );
    cv::fillConvexPoly( image, polygon, color, cv::LINE_AA );
}

int main(int argc, char* argv[])
{
    // キャプチャを開く
    cv::VideoCapture capture = cv::VideoCapture( "pose.jpg" ); // 画像ファイル
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // カメラ
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // モデルを読み込む
    const std::string weights = "human-pose-estimation.onnx";
    cv::dnn::KeypointsModel model = cv::dnn::KeypointsModel( weights );

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // モデルの入力パラメーターを設定する
    const double scale = 1.0 / 255.0;                          // スケールファクター
    const cv::Size size = cv::Size( 256, 456 );                // 入力サイズ
    const cv::Scalar mean = cv::Scalar( 128.0, 128.0, 128.0 ); // 差し引かれる平均値
    const bool swap = false;                                   // チャンネルの順番（True: RGB、False: BGR）
    const bool crop = false;                                   // クロップ
    model.setInputParams( scale, size, mean, swap, crop );

    // カラーテーブルを取得する
    std::vector<cv::Scalar> colors = get_colors();

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

        // モデルの入力サイズを設定する
        const cv::Size size = cv::Size( 256, static_cast<int32_t>( ( 256.0f / image.cols ) * image.rows ) ); // アスペクト比を保持する
        model.setInputSize( size );

        // キーポイントを検出する
        const float confidence_threshold = 0.6;
        const std::vector<cv::Point2f> keypoints = model.estimate( image, confidence_threshold );

        // キーポイントを描画する
        for( int32_t i = 0; i < keypoints.size(); i++ ){
            const cv::Point point = cv::Point( keypoints[i].x, keypoints[i].y );
            const int32_t radius = 5;
            const cv::Scalar color = colors[i];
            const int32_t thickness = -1;
            cv::circle( image, point, radius, color, thickness, cv::LINE_AA );
        }

        // ボーンを描画する
        for( const std::pair<joints, joints>& bone : bones ){
            const cv::Point point1 = cv::Point( keypoints[static_cast<int32_t>( bone.first )].x, keypoints[static_cast<int32_t>( bone.first )].y );
            const cv::Point point2 = cv::Point( keypoints[static_cast<int32_t>( bone.second )].x, keypoints[static_cast<int32_t>( bone.second )].y );
            if( point1 == cv::Point( -1, -1 ) && point2 == cv::Point( -1, -1 ) ){
                continue;
            }
            draw_bone( image, point1, point2, colors[static_cast<int32_t>( bone.second )] );
        }

        // 画像を表示する
        cv::imshow( "keypoints", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}