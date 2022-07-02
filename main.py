import numpy as np
import numba
import serial
import time
import pyautogui
import threading

data_send= np.zeros((1024), dtype= np.uint8)
could_send= False

@numba.jit(nopython= True, cache= True)
def resize_and_rgb2gray(img, h_new= 64, w_new= 128):
    h_old= img.shape[0]
    w_old= img.shape[1]
    alpha_w= w_old/w_new
    alpha_h= h_old/h_new
    rgb_y= np.array([0.299, 0.587, 0.114])
    h_i= 0
    img_new= np.zeros((h_new, w_new), dtype= np.uint8)
    for i in range(h_new):
        w_i= 0
        for j in range(w_new):
            img_new[i, j]= int(np.sum(img[int(h_i), int(w_i), :]*rgb_y)/255.0+0.3+0.4*np.random.random())#*
            w_i+= alpha_w
        h_i+= alpha_h
    return img_new

@numba.jit(nopython= True, cache= True)
def img2_BMP(img_light):
    img_bmp= np.zeros((16*64), dtype= np.uint8)
    for j in range(64):
        for i in range(16):
            bit_ij = 0
            for k in range(8):
                bit_ij = bit_ij << 1
                bit_ij = bit_ij | img_light[j, i * 8 + (7 - k)]
            img_bmp[j*16+i]= bit_ij
    return img_bmp

def fun_send_data():
    global could_send, data_send
    with serial.Serial(port="COM7", baudrate= 921600, timeout=5) as ser:
        while True:
            if could_send and ser.out_waiting==0:
                ser.write(bytes(data_send))
                could_send= False
            else:
                time.sleep(0.01)


def main():
    global could_send, data_send
    px0= 400
    py0= 300
    t1= threading.Thread(target=fun_send_data)
    t1.start()

    while True:
        img_rgb = np.array(pyautogui.screenshot())[py0:py0 + 300, px0:px0 + 600, :]
        img_gray = resize_and_rgb2gray(img_rgb)
        img_gray[img_gray > 1] = 1
        img_BMP = img2_BMP(img_gray)
        data_send= np.copy(img_BMP)
        could_send= True

        # 如出现画面撕裂，可加大此延时解决
        # 减小此延时可提高显示帧率
        time.sleep(0.03)

if __name__ == '__main__':
    main()

