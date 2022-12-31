#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    // �L���v�`�����J��
    cv::VideoCapture capture = cv::VideoCapture( "../face.jpg" ); // �摜�t�@�C��
    //cv::VideoCapture capture = cv::VideoCapture( 0 ); // �J����
    if( !capture.isOpened() ){
        throw std::runtime_error( "can't open capture!" );
    }

    // ���f����ǂݍ���
    // const std::string weights = "../opencv_face_detector.caffemodel"; // float32
    // const std::string config = "../opencv_face_detector.prototxt";
    const std::string weights = "../opencv_face_detector_fp16.caffemodel"; // float16
    const std::string config = "../opencv_face_detector_fp16.prototxt";
    // const std::string weights = "../opencv_face_detector_uint8.pb"; // uint8
    // const std::string config = "../opencv_face_detector_uint8.pbtxt";
    cv::dnn::DetectionModel model = cv::dnn::DetectionModel( weights, config );

    // ���f���̐��_�Ɏg�p����G���W���ƃf�o�C�X��ݒ肷��
    model.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    model.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );

    // ���f���̓��̓p�����[�^�[��ݒ肷��
    const double scale = 1.0;                                  // �X�P�[���t�@�N�^�[
    const cv::Size size = cv::Size( 300, 300 );                // ���̓T�C�Y
    const cv::Scalar mean = cv::Scalar( 104.0, 177.0, 123.0 ); //����������镽�ϒl
    const bool swap = false;                                   // �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
    const bool crop = false;                                   // �N���b�v
    model.setInputParams( scale, size, mean, swap, crop );

    while( true ){
        // �t���[�����L���v�`�����ĉ摜��ǂݍ���
        cv::Mat image;
        capture >> image;
        if( image.empty() ){
            cv::waitKey( 0 );
            break;
        }

        // �摜��3�`�����l���ȊO�̏ꍇ��3�`�����l���ɕϊ�����
        if( image.channels() == 1 ){
            cv::cvtColor( image, image, cv::COLOR_GRAY2BGR );
        }
        if( image.channels() == 4 ){
            cv::cvtColor( image, image, cv::COLOR_BGRA2BGR );
        }

        // ������o����
        const float confidence_threshold = 0.6;
        const float nms_threshold = 0.4;
        std::vector<int32_t> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        model.detect( image, class_ids, confidences, boxes, confidence_threshold, nms_threshold );

        // ���o������̃o�E���f�B���O�{�b�N�X��`�悷��
        const cv::Scalar color = cv::Scalar( 0, 0, 255 );
        const int32_t thickness = 2;
        for( const cv::Rect& box : boxes ){
            cv::rectangle( image, box, color, thickness, cv::LINE_AA );
        }

        // �摜��\������
        cv::imshow( "face detection", image );
        const int32_t key = cv::waitKey( 10 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}