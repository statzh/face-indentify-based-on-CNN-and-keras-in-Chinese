#-*- coding: utf-8 -*-
  
import cv2
import sys
import gc
from face_train_by_LeNet import Model
from load_face_dataset import get_classes

if __name__ == '__main__':
    '''if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
'''
        #加载模型
    model = Model()
    model.load_model(file_path = './model/me.face.model.h5')    

        #框住人脸的矩形边框颜色       
    color = (0, 255, 0)

        #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

        #人脸识别分类器本地存储路径
    cascade_path = "C:/Users/Tiago Haw/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"    
    
    n, f_dict = get_classes('F:\\face_identify')

        #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频

            #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

            #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect

                    #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                
                if faceID != -1:
                    name = f_dict[faceID]
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,name, 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'Stranger', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (128,0,128),                               #颜色红
                                2)                                     #字的线宽

                    
                '''
                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'kun', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽
                elif faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'me', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,255,0),                             #颜色红
                                2)                                     #字的线宽
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'Wenkai', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (128,0,128),                               #颜色红
                                2)                                     #字的线宽
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'Stranger', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (128,0,128),                               #颜色红
                                2)                                     #字的线宽
        '''
        cv2.imshow("Recognise myself", frame)

            #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
            #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

        #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()