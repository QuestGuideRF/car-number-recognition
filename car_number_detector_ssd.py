import logging
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pytesseract import image_to_string
from path import Path
from object_detection .utils import label_map_util
from object_detection .utils import visualization_utils as vis_util
logging .getLogger ('tensorflow').setLevel (logging .ERROR )
def detection_params (image ,detection_graph ,sess ):
    image_np_expanded =np .expand_dims (image ,axis =0 )
    image_tensor =detection_graph .get_tensor_by_name ('image_tensor:0')
    boxes =detection_graph .get_tensor_by_name ('detection_boxes:0')
    scores =detection_graph .get_tensor_by_name ('detection_scores:0')
    classes =detection_graph .get_tensor_by_name ('detection_classes:0')
    num_detections =detection_graph .get_tensor_by_name ('num_detections:0')
    (boxes ,scores ,classes ,num_detections )=sess .run ([boxes ,scores ,classes ,num_detections ],
    feed_dict ={image_tensor :image_np_expanded })
    return boxes ,scores ,classes ,num_detections
def ocr_tesseract (image ):
    img_after_resize =cv2 .resize (image ,None ,fx =2 ,fy =2 ,
    interpolation =cv2 .INTER_CUBIC )
    img_after_bilateral_filter =cv2 .bilateralFilter (img_after_resize ,10 ,20 ,20 )
    img_gray =cv2 .cvtColor (img_after_bilateral_filter ,cv2 .COLOR_BGR2GRAY )
    numbers =image_to_string (img_gray ,lang ='rus',config ='--псм 6'
    '--oem 0'
    '-c tessedit_char_whitelist=0123456789ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ'
    '-c tessedit_char_blacklist=.,|><:;~`/][}{-=)($%^-!')
    return numbers
def cut_detect_numbers (scores ,boxes ,image_np ):
    numbers =[]
    for i in range (scores [0 ].shape [0 ]):
        if int (100 *scores [0 ][i ])>50 :
            img_screenshot =image_np
            box =boxes [0 ][i ]
            dx =int (box [1 ]*img_screenshot .shape [1 ])
            dy =int (box [0 ]*img_screenshot .shape [0 ])
            dw =int ((box [3 ]-box [1 ])*img_screenshot .shape [1 ])
            dh =int ((box [2 ]-box [0 ])*img_screenshot .shape [0 ])
            crop_img =img_screenshot [dy :dy +dh ,dx :dx +dw ]
            number =ocr_tesseract (crop_img )
            numbers .append (number )
    return numbers
def detector (detection_graph ,category_index ,mode ,image_path ):
    with detection_graph .as_default ():
        with tf .Session (graph =detection_graph )as sess :
            if mode =='0':
                path_to_image =Path (image_path )
                image_np =cv2 .imread (path_to_image )
                boxes ,scores ,classes ,num_detections =detection_params (image_np ,detection_graph ,sess )
                image_np_for_vis =image_np
                vis_util .visualize_boxes_and_labels_on_image_array (
                image_np_for_vis ,np .squeeze (boxes ),
                np .squeeze (classes ).astype (np .int32 ),
                np .squeeze (scores ),
                category_index ,
                use_normalized_coordinates =True ,
                line_thickness =2 )
                numbers =cut_detect_numbers (scores ,boxes ,image_np )
                print ('Обнаруженные номера --- \n',numbers ,'\n')
                while True :
                    cv2 .imshow ('обнаружение объектов',cv2 .resize (image_np_for_vis ,(800 ,600 )))
                    if cv2 .waitKey (25 )&0xFF ==ord ('q'):
                        cv2 .destroyAllWindows ()
                        break
            elif mode =='1':
                cap =cv2 .VideoCapture (0 )
                while True :
                    ret ,image_np =cap .read ()
                    boxes ,scores ,classes ,num_detections =detection_params (image_np ,detection_graph ,sess )
                    vis_util .visualize_boxes_and_labels_on_image_array (
                    image_np ,np .squeeze (boxes ),
                    np .squeeze (classes ).astype (np .int32 ),
                    np .squeeze (scores ),
                    category_index ,
                    use_normalized_coordinates =True ,
                    line_thickness =2 )
                    numbers =cut_detect_numbers (scores ,boxes ,image_np )
                    print ('Обнаруженные номера --- \n',numbers ,'\n')
                    cv2 .imshow ('обнаружение объектов',cv2 .resize (image_np ,(800 ,600 )))
                    if cv2 .waitKey (25 )&0xFF ==ord ('q'):
                        cv2 .destroyAllWindows ()
                        break
            else :
                print ('Режим детектора не выбран, попробуйте еще раз!')
def main ():
    PATH_TO_CKPT =Path ('detector_model/frozen_inference_graph.pb')
    PATH_TO_LABELS =Path ('object_detection.pbtxt')
    NUM_CLASSES =1
    detection_graph =tf .Graph ()
    with detection_graph .as_default ():
        od_graph_def =tf .GraphDef ()
        with tf .gfile .GFile (PATH_TO_CKPT ,'rb')as fid :
            serialized_graph =fid .read ()
            od_graph_def .ParseFromString (serialized_graph )
            tf .import_graph_def (od_graph_def ,name ='')
    label_map =label_map_util .load_labelmap (PATH_TO_LABELS )
    categories =label_map_util .convert_label_map_to_categories (label_map ,
    max_num_classes =NUM_CLASSES ,
    use_display_name =True )
    category_index =label_map_util .create_category_index (categories )
    detector (detection_graph ,category_index ,args .mode ,args .image_path )
if __name__ =='__main__':
    parser =argparse .ArgumentParser (description ='Обнаружить и прочитать номерной знак автомобиля на загруженном изображении')
    parser .add_argument ('-m','--mode',help ='режим работы обнаружение видео или изображения')
    parser .add_argument ('-i','--image_path',help ='путь к изображению')
    args =parser .parse_args ()
    main ()