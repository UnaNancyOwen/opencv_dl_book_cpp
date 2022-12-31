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

    // ���ފ��ǂݍ���
    const std::string path = "../haarcascade_frontalface_default.xml";
    cv::CascadeClassifier cascade = cv::CascadeClassifier( path );
    if( cascade.empty() ){
        throw std::runtime_error( "can't read cascade!" );
    }

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

        // �O���[�X�P�[���ɕϊ�����
        cv::Mat gray_image;
        cv::cvtColor( image, gray_image, cv::COLOR_BGR2GRAY );

        // ������o����
        const int32_t width = gray_image.cols;
        const int32_t height = gray_image.rows;
        const cv::Size min_size = cv::Size( width / 10, height / 10 );
        std::vector<cv::Rect> boxes;
        cascade.detectMultiScale( gray_image, boxes, 1.1, 3, 0, min_size);

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