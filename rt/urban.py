import numpy as np





class Urban:


    shapes = np.asarray([[10,10,10,0.001],
                         [10,10,30,0.0005]])
    n_shape,n_dim = np.shape(shapes)

    Eroof = 0.90
    Ewall = 0.85
    Estreat = 0.92


    Troof_sunlit = 310
    Troof_shaded = 300
    Twall_sunlit = 310
    Twall_shaded = 300
    Tstreat_sunlit = 310
    Tstreat_shaded = 300
    vza = 40
    vaa = 90
    sza = 20
    saa = 0
    raa = vaa - saa
    n_part = 500



    ### 第一种楼
    rd = np.pi/180.0
    ui = np.cos(sza*rd)
    uv = np.cos(vza*rd)
    si = np.sin(sza*rd)
    sv = np.sin(vza*rd)
    up = np.cos(raa*rd)
    tantv = np.tan(vza*rd)
    tants = np.tan(sza*rd)


    ### 街道的可视比例
    height1r = 0
    projv = 0
    projs = 0
    projvs = 0
    laa = 0
    waa = laa + 90
    for kshape2 in range(n_shape):
        shape2 = shapes[kshape2]
        length2 = shape2[0]
        width2 = shape2[1]
        height2 = shape2[2]
        alpha2 = shape2[3]
        height2m1 = height2 - height1r
        if height2m1 <= 0: continue
        proj_roof = alpha2 * length2 * width2
        projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa)*rd)) + height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa)*rd)))
        projv_temp = projv_wall + proj_roof
        projv = projv + projv_temp
        projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(np.cos((saa - laa)*rd)) + height2m1 * width2 * tants * np.abs(np.cos((saa - waa)*rd)))
        projs_temp = projv_wall + proj_roof
        projs = projs + projs_temp
    Overlapping = np.sqrt(tantv*tantv+tants*tants-2*tantv*tants*up) /(tantv+tants)
    projvs = projv + projs*Overlapping

    pStreatV = np.exp(-projv)
    pStreatS = np.exp(-projs)
    pStreat_sunlit = np.exp(-projvs)
    pStreat_shaded = pStreatV - pStreat_sunlit

    ## 计算屋顶的可视比例
    pRoofV = 0
    pRoofS = 0
    pRoof_sunlit = 0
    for kshape1 in range(n_shape):
        shape1 = shapes[kshape1]
        length1 = shape1[0]
        width1 = shape1[1]
        height1 = shape1[2]
        alpha1 = shape1[3]
        height1r = height1
        projv = 0
        projs = 0
        for kshape2 in range(n_shape):
            shape2 = shapes[kshape2]
            length2 = shape2[0]
            width2 = shape2[1]
            height2 = shape2[2]
            alpha2 = shape2[3]
            height2m1 = height2 - height1r
            if height2m1 <= 0: continue
            proj_roof = alpha2 * length2 * width2
            projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa)*rd)) + height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa)*rd)))
            projv_temp = projv_wall + proj_roof
            projv = projv + projv_temp

            projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(
                np.cos((saa - laa) * rd)) + height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
            projs_temp = projs_wall + proj_roof
            projs = projs + projs_temp
        Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
        projvs =  projv + projs*Overlapping
        gapv_roof = np.exp(-projv)
        gaps_roof = np.exp(-projs)
        gapvs_roof = np.exp(-projvs)
        pRoofV_temp = alpha1 * gapv_roof * (length1) * (width1)
        pRoofV = pRoofV + pRoofV_temp
        pRoofS_temp = alpha1 * gaps_roof * (length1) * (width1)
        pRoofS = pRoofS + pRoofS_temp
        pRoof_sunlit_temp = alpha1 * gapvs_roof * (length1) * (width1)
        pRoof_sunlit = pRoof_sunlit + pRoof_sunlit_temp

    pWall_sunlit_fraction = 0
    pWall_sunlit_fraction_Weight = 0
    for kshape1 in range(n_shape):
        shape1 = shapes[kshape1]
        length1 = shape1[0]
        width1 = shape1[1]
        height1 = shape1[2]
        alpha1 = shape1[3]
        height1r = height1
        dheight = height1/n_part
        projv = 0
        projs = 0
        for kh in range(n_part):
            height_temp = dheight * (kh+0.5)

            proj = (shapes[:, 2] - height_temp) *shapes[:,3] * tantv
            ind = proj < 0
            proj[ind] = 0

            proj_roof = np.sum(shapes[:, 3] * shapes[:, 2] * shapes[:, 0])
            projv_wall =  np.sum(proj * shapes[:, 0]) * np.abs(np.cos((vaa - laa) * rd)) + np.sum(proj * shapes[:, 1]) * np.abs(np.cos((vaa - waa) * rd))
            projv_temp = projv_wall + proj_roof
            projv = projv + projv_temp

            proj = (shapes[:, 2] - height_temp) *shapes[:,3]  * tants
            ind = proj < 0
            proj[ind] = 0
            projs_wall =  np.sum(proj * shapes[:, 0]) * np.abs(np.cos((saa - laa) * rd)) + np.sum(proj * shapes[:, 1]) * np.abs(np.cos((saa - waa) * rd))
            projs_temp = projs_wall + proj_roof
            projs = projs + projs_temp
        projv = projv / n_part
        projs = projs / n_part
        Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
        projvs = projv + projs*Overlapping
        gapv_wall= np.exp(-projv)
        gapvs_wall = np.exp(-projvs)

        lsunlit = 0
        part1 = np.abs(laa-saa)
        if part1 > 180: part1 = 360 -part1
        part2 = np.abs(laa-vaa)
        if part2 > 180: part2 = 360 -part2
        if (part1<90)*(part2<90) > 0: lsunlit = 1
        wsunlit = 0
        part1 = np.abs(waa-saa)
        if part1 > 180: part1 = 360 -part1
        part2 = np.abs(waa-vaa)
        if part2 > 180: part2 = 360 -part2
        if (part1<90)*(part2<90) > 0: wsunlit = 1

        pWall_sunlit_fraction = pWall_sunlit_fraction + alpha1 * gapvs_wall * ((height1*length1*tantv*np.cos((vaa - laa) * rd))*lsunlit + (height1*length1*tantv*np.cos((vaa - laa) * rd)*wsunlit))
        pWall_sunlit_fraction_Weight = pWall_sunlit_fraction_Weight + alpha1  * gapv_wall * ((height1*length1*tantv*np.cos((vaa - laa) * rd)) + (height1*length1*tantv*np.cos((vaa - laa) * rd)))

    pWall_sunlit_fraction = pWall_sunlit_fraction/pWall_sunlit_fraction_Weight

    pWallV = 1 - pStreatV - pRoofV
    pWall_sunlit = pWallV * pWall_sunlit_fraction

    print(pStreatV, pStreat_sunlit)
    print( pRoofV,pRoof_sunlit)
    print( pWallV, pWall_sunlit)

    ###############################################
    #### multiple scattering effect
    ###########################################

    if2 = 2.0
    n = 500
    kangle = np.linspace(0,89,90) # hemisphere space
    dangle = np.pi/2.0/90.0 # step in pi/2.0
    tantemp = np.tan(kangle*rd) #
    fweiplus = np.sin(kangle*rd)*dangle*np.cos(kangle*rd) #cos and sin
    tempS = np.sum(shapes[:,2]*shapes[:,0]*shapes[:,3]) * tantv
    fstreatv = np.exp(-tempS)  # gap frequency of trunk in a direction
    i0v = 1-fstreatv
    eu = 0.0
    ed = 0.0
    au = 0.5
    ad = 0.5
    for kshape1 in range(n_shape):
        shape1 = shapes[kshape1]
        length1 = shape1[0]
        width1 = shape1[1]
        height1 = shape1[2]
        alpha1 = shape1[3]
        dheight = height1 / n_part
        height1p = height1
        for kh in range(n_part):
            heighttemp = dheight * (kh+0.5)

            area = (shapes[:,2]-heighttemp)*shapes[:,0]*shapes[:,3]
            ind = area <0
            area[ind] = 0
            projs = np.sum(area) * tantemp
            ftemp = np.exp(-projs)
            tempdS = length1 * dheight * tantv * alpha1
            tempSv = np.sum((shapes[:,2]-heighttemp)*shapes[:,0]*shapes[:,3]) * tantv
            ftempv = np.exp(-tempSv)
            eu = eu + if2 *np.sum(ftemp*fweiplus)*tempdS*au*ftempv
            # a hemisphere space
            area = shapes[:, 0] * heighttemp * shapes[:, 3]
            ind = shapes[:,2] < heighttemp
            area[ind] = shapes[ind,2]
            tempS = np.sum(area) * tantemp   # temp trunk projection in a hemi
            ftemp = np.exp(-tempS)  # temp gap frequency in a hemi
            ed = ed + if2 * np.sum(ftemp * fweiplus) * tempdS * ad * ftempv
    eu = eu / i0v
    ed = ed / i0v
    p = 1 - eu - ed


    wr = 0.0
    for kshape1 in range(n_shape):
        shape1 = shapes[kshape1]
        length1 = shape1[0]
        width1 = shape1[1]
        height1 = shape1[2]
        alpha1 = shape1[3]
        heighttemp = height1
        area = (shapes[:,2]-heighttemp)*shapes[:,0]*shapes[:,3]
        ind = area < 0
        area[ind] = 0
        projs = np.sum(area) * tantemp
        ftemp =  np.exp(-projs)
        tempdS = length1 * length1 * alpha1
        tempSv = np.sum((shapes[:, 2] - heighttemp) * shapes[:, 0] * shapes[:,3]) * tantv
        ftempv = np.exp(-tempSv)
        wr =  wr + if2 * np.sum((1-ftemp) * fweiplus) * tempdS * ftempv
    eww = Ewall * (1 - Ewall) * i0v * p
    ews = Ewall * i0v * ed * (1-i0v) *(1-Estreat)
    ewr = Ewall *(1-Eroof) * wr
    esw = Estreat *(1-Ewall) * i0v * ed


    print(eww,ews,ewr,esw)
