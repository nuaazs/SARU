import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
ROOT = "/home/iint/lmz/ct_t1/SyNRA_png_0318_003/train/ct" # test
SAVE_PATH = "/home/iint/lmz/ct_t1/SyNRA_png_0318_003/test/skin_npy"
os.makedirs(SAVE_PATH,exist_ok=True)

def get_slice_num(filename):
    return int(filename.split("/")[-1].split("_")[1].split(".")[0])
# 均值迁移去噪声+二值化 Mean shift denoise + binarization
def threshold_demo(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow("mask", binary)
    return binary

# for pname in ['070', '071', '073', '074', '076', '081', '082', '084', '085', '087', '089', '091', '093', '094', '096', '100', '101', '104', '105', '106', '107', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '127', '128', '129', '130', '131', '134', '135', '137', '138', '139', '140', '141', '142', '143']:
#     _list = sorted([file for file in os.listdir(ROOT) if "png" in file and pname+"_" in file],key = get_slice_num)

def get_skin(png_list,plot=True,save_path="/home/zhaosheng/paper2/online_code/response/skins"):
    p_list = []
    all_area = 0
    for filename in png_list:
        # print(f"Filename:{filename}")
        pname = filename.split("/")[-1].split("_")[0]
        filepath = os.path.join(filename)

        src1 = cv.imread(filepath)
        src2 = cv.imread(filepath)
        binary = threshold_demo(src2)
        # contour discovery
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)# cv.CHAIN_APPROX_SIMPLE
        temp_array = np.zeros((256,256))
        
        for c in range(len(contours)):
            area = cv.contourArea(contours[c])
            if area < 3600:
                continue
            # print(area)
            all_area += area
            cv.drawContours(src1, contours, c, (0, 0, 255), 2, 8)
            for onedot in contours[c]:
                x = onedot[0,0]
                y = onedot[0,1]
                temp_array[y,x]=1
        if plot:
            plt.figure(dpi=200)
            plt.imshow(temp_array)
            plt.show()
        p_list.append(temp_array)
    p_list = np.array(p_list)
    p_list = p_list.transpose(1,2,0)
    # print(f"{pname}:{p_list.shape}")
    np.save(os.path.join(save_path,f"{pname}_skin.npy"),p_list)
    return 
    

def get_skin(img_array,axis=2):
    skin = []
    head = []
    slices = img_array.shape[axis]
    for slice in range(slices):
        slice_array = img_array[:,:,slice]
        slice_array = (slice_array-slice_array.min())*255/(slice_array.max()-slice_array.min())
        im = Image.fromarray(slice_array)
        im = im.convert('L')
        im.save(f"/tmp/temp_{slice}.png")
        src1 = cv2.imread(f"/tmp/temp_{slice}.png")
        binary = threshold_demo(src1)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        temp_array = np.zeros(slice_array.shape)
        empty_pic = Image.fromarray(temp_array)
        empty_pic = empty_pic.convert('L')
        empty_pic.save(f"/tmp/temp_{slice}_empty_pic.png")
        skin_src = cv2.imread(f"/tmp/temp_{slice}_empty_pic.png")
        head_src = cv2.imread(f"/tmp/temp_{slice}_empty_pic.png")
        
        lagest_area = 25
        largest_index = 999
        for c in range(len(contours)):
            area = cv2.contourArea(contours[c])
            if area > lagest_area:
                lagest_area = area
                largest_index = c
                largest_contour=contours[c]
            else:
                continue
            # print(area)
        cv2.drawContours(skin_src,[largest_contour],-1,(0,0,255),1)  #绘制轮廓
        cv2.fillPoly(head_src, pts =[largest_contour], color=(0,0,255))
        

            # for onedot in contours[c]:
            #     x = onedot[0,0]
            #     y = onedot[0,1]
            #     temp_array[y,x]=1
        skin_image = cv2.cvtColor(skin_src, cv2.COLOR_BGR2GRAY)
        head_image = cv2.cvtColor(head_src, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(f"/tmp/temp_{slice}_skin.png", skin_image)
        cv2.imwrite(f"/tmp/temp_{slice}_head.png", head_image)
        skin_image_pil = Image.open(f"/tmp/temp_{slice}_skin.png")
        skin_image_pil = skin_image_pil.convert('L')
        skin_array = np.asarray(skin_image_pil)
        skin_array[skin_array>0]=1
        skin.append(skin_array)

        head_image_pil = Image.open(f"/tmp/temp_{slice}_head.png")
        head_image_pil = head_image_pil.convert('L')
        head_array = np.asarray(head_image_pil)
        head_array[head_array>0]=1
        head.append(head_array)
    head = np.array(head)
    skin = np.array(skin)

    skin = skin.transpose(1,2,0)
    head = head.transpose(1,2,0)
    return skin,head
