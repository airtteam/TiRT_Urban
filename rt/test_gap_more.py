import numpy as np

from utils.utils import *



length1 = 10
width1 = 10
height1 = 30
alpha1 = 0.003
n_part = 10
vza = 5
Ewall = 0.90
Estreet = 0.92
for vza in range(10,51,40):
    tantv = np.tan(vza*rd)

    S = length1 * height1 * tantv * alpha1
    Sroof = length1 * width1 * alpha1
    f = np.exp(-S-Sroof)
    ff = np.exp(-Sroof)
    f0 = np.exp(-S)
    # i0 = f-ff
    i0 = 1  - f0

    if2 = 1.0
    n_hza0 = 90
    n_haa0 = 60
    hza0 = np.linspace(0, 89, n_hza0)  # hemisphere space
    haa0 = np.linspace(0, 360, n_haa0)
    hza, haa = np.meshgrid(hza0, haa0)
    hza = np.reshape(hza, -1)  ### to linear
    haa = np.reshape(haa, -1)  ### to linear

    ### step of hza in rad unit rather zhan degree unit
    dangle = np.pi / 2.0 / n_hza0  # 角度的步长
    tantemp = np.tan(hza * rd)  ### 半球的正切
    fweiplus = np.sin(hza * rd) * dangle * np.cos(hza * rd)  # 角度的归一化方式, 一般是0.5
    # fweiplus = np.sin(hza * rd) * dangle  ##### 一般是1.0
    fweiplusSum = np.sum(if2 * fweiplus)  ### 权重累计，如果没有meshgrid，应该是0.5，现在总和是0.5*n_haa0

    dheight = height1/n_part
    Sum = 0
    eu = 0
    ed = 0
    au = 0.5
    ad = 0.5
    for kh in range(n_part):
        height11 = (kh+0.5) *dheight
        Sv = (height1 - height11)*length1 * tantv*alpha1
        projStemp1 = (height1 - height11)*length1 *alpha1 * tantemp
        projStemp2 = (height11) * length1  * alpha1 * tantemp
        ftemp1 = np.exp(-projStemp1)
        ftemp2 = np.exp(-projStemp2)
        #### 看到这里的概率
        fv = np.exp(-Sv)
        #### 这里有地物的概率
        dS = dheight * length1 * alpha1* tantv
        ### 半球等效
        hee1 = np.sum(ftemp1 * fweiplus / fweiplusSum * if2)
        hee2 = np.sum(ftemp2 * fweiplus / fweiplusSum * if2)
        eu = eu + fv * dS*hee1
        print('hee1:',hee1,'hee2:',hee2)
        ed = ed + fv*dS*hee2

    ed = ed/i0*0.5
    eu = eu/i0*0.5
    p = 1- ed - eu


    dS = 1.0
    projStemp1 = (height1) * length1 * alpha1 * tantemp
    ftemp1 = np.exp(-projStemp1)
    hee1 = np.sum(ftemp1 * fweiplus / fweiplusSum * if2)
    euu = f0 * dS*(1-hee1)
    print('hee1:',hee1)

    eww = Ewall *(1- Ewall)* i0*p
    esw = Estreet * (1-Ewall) * i0 * ed
    ews = Ewall *(1-Estreet) * euu
    ew = i0 * Ewall
    es = f0 * Estreet
    print(eww,esw,ews,eww+esw+ews+ew+es)


