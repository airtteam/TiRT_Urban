import numpy as np





class Urban:


    shapes = np.asarray([[10,10,10,0.001],
                         [10,10,30,0.0005]])
    n_shape,n_dim = np.shape(shapes)

    ### liqing 0.90-0.85
    ### shimian 0.90-0.95
    ### zhuan 0.930
    ### boli 0.94
    ### suliao 0.7-0.9
    ### shuini 0.95
    ### wuding 0.91
    ### hunningtu 0.94


    Eroof = 0.92
    Ewall = 0.90
    Estreat = 0.90


    Troof_sunlit = 310
    Troof_shaded = 300
    Twall_sunlit = 310
    Twall_shaded = 300
    Tstreat_sunlit = 310
    Tstreat_shaded = 300
    vza_ = np.asarray([40,40])
    vaa_ = np.asarray([90,90])
    sza = 20
    saa = 5
    n_part = 20
    n_angle = 2
    canopy_effective_emissivity = []
    component_effective_emissivity = []
    canopy_brightness_temperature = []
    pStreatV = 0
    pRoofV = 0
    pWallV = 0

    ###############################################
    #### geometrical optical theory
    ###########################################
    def calculate_component_effective_emissivity(self,kangle):
        shapes = self.shapes
        Eroof = self.Eroof
        Ewall = self.Ewall
        Estreat = self.Estreat
        n_shape = self.n_shape
        n_dem = self.n_dim
        n_part = self.n_part
        vza = self.vza_[kangle]
        vaa = self.vaa_[kangle]
        sza = self.sza
        saa = self.saa
        raa = vaa - saa



        rd = np.pi / 180.0
        ui = np.cos(sza * rd)
        uv = np.cos(vza * rd)
        si = np.sin(sza * rd)
        sv = np.sin(vza * rd)
        up = np.cos(raa * rd)
        tantv = np.tan(vza * rd)
        tants = np.tan(sza * rd)

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
            projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa) * rd)) + height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa) * rd)))
            projv_temp = projv_wall + proj_roof
            projv = projv + projv_temp
            projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(
                np.cos((saa - laa) * rd)) + height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
            projs_temp = projv_wall + proj_roof
            projs = projs + projs_temp
        Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
        projvs = projv + projs * Overlapping

        pStreatV = np.exp(-projv)
        pStreatS = np.exp(-projs)
        pStreatV_sunlit = np.exp(-projvs)
        pStreatV_shaded = pStreatV - pStreatV_sunlit

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
                projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(
                    np.cos((vaa - laa) * rd)) + height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa) * rd)))
                projv_temp = projv_wall + proj_roof
                projv = projv + projv_temp

                projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(
                    np.cos((saa - laa) * rd)) + height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
                projs_temp = projs_wall + proj_roof
                projs = projs + projs_temp
            Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
            projvs = projv + projs * Overlapping
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
            dheight = height1 / n_part
            projv = 0
            projs = 0
            for kh in range(n_part):
                height_temp = dheight * (kh + 0.5)

                proj = (shapes[:, 2] - height_temp) * shapes[:, 3] * tantv
                ind = proj < 0
                proj[ind] = 0

                proj_roof = np.sum(shapes[:, 3] * shapes[:, 2] * shapes[:, 0])
                projv_wall = np.sum(proj * shapes[:, 0]) * np.abs(np.cos((vaa - laa) * rd)) + np.sum(
                    proj * shapes[:, 1]) * np.abs(np.cos((vaa - waa) * rd))
                projv_temp = projv_wall + proj_roof
                projv = projv + projv_temp

                proj = (shapes[:, 2] - height_temp) * shapes[:, 3] * tants
                ind = proj < 0
                proj[ind] = 0
                projs_wall = np.sum(proj * shapes[:, 0]) * np.abs(np.cos((saa - laa) * rd)) + np.sum(
                    proj * shapes[:, 1]) * np.abs(np.cos((saa - waa) * rd))
                projs_temp = projs_wall + proj_roof
                projs = projs + projs_temp
            projv = projv / n_part
            projs = projs / n_part
            Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
            projvs = projv + projs * Overlapping
            gapv_wall = np.exp(-projv)
            gapvs_wall = np.exp(-projvs)

            lsunlit = 0
            part1 = np.abs(laa - saa)
            if part1 > 180: part1 = 360 - part1
            part2 = np.abs(laa - vaa)
            if part2 > 180: part2 = 360 - part2
            if (part1 < 90) * (part2 < 90) > 0: lsunlit = 1
            wsunlit = 0
            part1 = np.abs(waa - saa)
            if part1 > 180: part1 = 360 - part1
            part2 = np.abs(waa - vaa)
            if part2 > 180: part2 = 360 - part2
            if (part1 < 90) * (part2 < 90) > 0: wsunlit = 1

            pWall_sunlit_fraction = pWall_sunlit_fraction + alpha1 * gapvs_wall * (
                        (height1 * length1 * tantv * np.abs(np.cos((vaa - laa)) * rd)) * lsunlit + (
                            height1 * width1 * tantv * np.cos((vaa - waa) * rd) * wsunlit))
            pWall_sunlit_fraction_Weight = pWall_sunlit_fraction_Weight + alpha1 * gapv_wall * (
                        (height1 * length1 * tantv * np.abs(np.cos((vaa - laa) * rd))) + (
                            height1 * width1 * tantv * np.abs(np.cos((vaa - waa) * rd))))

        pWall_sunlit_fraction = pWall_sunlit_fraction / pWall_sunlit_fraction_Weight

        pWallV = 1 - pStreatV - pRoofV
        pWallV_sunlit = pWallV * pWall_sunlit_fraction
        pWallV_shaded = pWallV - pWallV_sunlit


    def calculate_direct_emissivity(self,kangle,ifP = 0):
        shapes = self.shapes
        Eroof = self.Eroof
        Ewall = self.Ewall
        Estreat = self.Estreat
        n_shape = self.n_shape
        n_dem = self.n_dim
        n_part = self.n_part
        vza = self.vza_[kangle]
        vaa = self.vaa_[kangle]
        sza = self.sza
        saa = self.saa
        raa = vaa - saa


        rd = np.pi / 180.0
        ui = np.cos(sza * rd)
        uv = np.cos(vza * rd)
        si = np.sin(sza * rd)
        sv = np.sin(vza * rd)
        up = np.cos(raa * rd)
        tantv = np.tan(vza * rd)
        tants = np.tan(sza * rd)

        height1r = 0
        projv = 0
        projs = 0
        projvs = 0
        laa = 0
        waa = laa + 90
        proof = 0
        for kshape2 in range(n_shape):
            shape2 = shapes[kshape2]
            length2 = shape2[0]
            width2 = shape2[1]
            height2 = shape2[2]
            alpha2 = shape2[3]

            height2m1 = height2 - height1r
            if height2m1 <= 0: continue
            proj_roof = alpha2 * length2 * width2
            projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                                   height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa) * rd)))
            projv_temp = projv_wall
            projv = projv + projv_temp
            projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs( np.cos((saa - laa) * rd)) +
                                   height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
            projs_temp = projs_wall
            projs = projs + projs_temp
            proof = proof+ proj_roof
        Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
        projvs = projv + projs * Overlapping

        pStreatV = np.exp(-projv)*(1-proof)
        pStreatS = np.exp(-projs)*(1-proof)
        pStreatV_sunlit = np.exp(-projvs)*(1-proof)
        pStreatV_shaded = pStreatV - pStreatV_sunlit
        pStreatV_sunlit_fraction = pStreatV_sunlit/pStreatV

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
                projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                                       height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa) * rd)))
                projv_temp = projv_wall
                projv = projv + projv_temp

                projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(np.cos((saa - laa) * rd)) +
                                       height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
                projs_temp = projs_wall
                projs = projs + projs_temp
            Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
            projvs = projv + projs * Overlapping
            gapv_roof = np.exp(-projv)
            gaps_roof = np.exp(-projs)
            gapvs_roof = np.exp(-projvs)
            pRoofV_temp = alpha1 * gapv_roof * (length1) * (width1)
            pRoofV = pRoofV + pRoofV_temp
            pRoofS_temp = alpha1 * gaps_roof * (length1) * (width1)
            pRoofS = pRoofS + pRoofS_temp
            pRoof_sunlit_temp = alpha1 * gapvs_roof * (length1) * (width1)
            pRoof_sunlit = pRoof_sunlit + pRoof_sunlit_temp
        pRoof_shaded = pRoofV - pRoof_sunlit

        pWall_sunlit_fraction = 0
        pWall_sunlit_fraction_Weight = 0
        for kshape1 in range(n_shape):
            shape1 = shapes[kshape1]
            length1 = shape1[0]
            width1 = shape1[1]
            height1 = shape1[2]
            alpha1 = shape1[3]
            height1r = height1
            dheight = height1 / n_part
            projv = 0
            projs = 0
            for kh in range(n_part):
                height_temp = dheight * (kh + 0.5)
                #### 计算在观测方向投影，分别有墙壁和屋顶的投影
                proj = (shapes[:, 2] - height_temp) * shapes[:, 3] * tantv
                ind = proj < 0
                proj[ind] = 0
                area_roof = shapes[:, 3] * shapes[:, 2] * shapes[:, 0]
                area_roof[ind] = 0
                proj_roof = np.sum(area_roof)
                projv_wall = np.sum(proj * shapes[:, 0]) * np.abs(np.cos((vaa - laa) * rd)) +\
                             np.sum(proj * shapes[:, 1]) * np.abs(np.cos((vaa - waa) * rd))
                projv_temp = projv_wall + proj_roof
                projv = projv + projv_temp

                ### 计算在太阳角度投影
                proj = (shapes[:, 2] - height_temp) * shapes[:, 3] * tants
                ind = proj < 0
                proj[ind] = 0
                projs_wall = np.sum(proj * shapes[:, 0]) * np.abs(np.cos((saa - laa) * rd)) + \
                             np.sum(proj * shapes[:, 1]) * np.abs(np.cos((saa - waa) * rd))
                area_roof = shapes[:, 3] * shapes[:, 2] * shapes[:, 0]
                area_roof[ind] = 0
                proj_roof = np.sum(area_roof)
                projs_temp = projs_wall + proj_roof
                projs = projs + projs_temp
            projv = projv / n_part
            projs = projs / n_part
            Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
            if Overlapping==0:
                a = 1
            projvs = projv + projs * Overlapping
            gapv_wall = np.exp(-projv)
            gapvs_wall = np.exp(-projvs)

            lsunlit = 0
            part1 = np.abs(laa - saa)
            if part1 > 180: part1 = 360 - part1
            part2 = np.abs(laa - vaa)
            if part2 > 180: part2 = 360 - part2
            if (part1 < 90) * (part2 < 90) > 0: lsunlit = 1
            wsunlit = 0
            part1 = np.abs(waa - saa)
            if part1 > 180: part1 = 360 - part1
            part2 = np.abs(waa - vaa)
            if part2 > 180: part2 = 360 - part2
            if (part1 < 90) * (part2 < 90) > 0: wsunlit = 1

            pWall_sunlit_fraction = pWall_sunlit_fraction + alpha1 * gapvs_wall * (
                        (height1 * length1 * tantv * np.abs(np.cos((vaa - laa) * rd))) * lsunlit +
                        (height1 * width1 * tantv * np.abs(np.cos((vaa - waa) * rd)) * wsunlit))
            pWall_sunlit_fraction_Weight = pWall_sunlit_fraction_Weight + alpha1 * gapv_wall * (
                        (height1 * length1 * tantv * np.abs(np.cos((vaa - laa) * rd))) +
                        (height1 * width1 * tantv * np.abs(np.cos((vaa - waa) * rd))))
        if pWall_sunlit_fraction_Weight !=0:
            pWall_sunlit_fraction = pWall_sunlit_fraction / pWall_sunlit_fraction_Weight
        else:
            pWall_sunlit_fraction = 0

        pWallV = 1 - pStreatV - pRoofV
        # pWallV = pWall_sunlit_fraction_Weight
        pWallV_sunlit = pWallV * pWall_sunlit_fraction
        pWallV_shaded = pWallV - pWallV_sunlit


        # pStreatV = 1-pWallV-pRoofV
        # pStreatV_sunlit = pStreatV * pStreatV_sunlit_fraction
        # pStreatV_shaded = pStreatV - pStreatV_sunlit

        if ifP ==1:
            return pWallV*Ewall, pStreatV*Estreat, pRoofV*Eroof
            # return 0, 0, pRoofV
        elif ifP ==2:
            return pWallV_sunlit * Ewall, pWallV_shaded * Ewall,\
                   pStreatV_sunlit * Estreat, pStreatV_shaded * Estreat,\
                   pRoof_sunlit*Eroof,pRoof_shaded*Eroof
        elif ifP ==3:
            return pWallV_sunlit, pWallV_shaded , \
                   pStreatV_sunlit , pStreatV_shaded , \
                   pRoof_sunlit , pRoof_shaded
        else:
            return pWallV*Ewall + pStreatV *Estreat + pRoofV*Eroof



    ###############################################
    #### spectral invariant theory
    ###########################################
    def calculate_scattering_emissivity(self, hza0,ifP = 0):

        shapes = self.shapes
        Eroof = self.Eroof
        Ewall = self.Ewall
        Estreat = self.Estreat
        n_shape = self.n_shape
        n_dem = self.n_dim
        n_part = self.n_part
        vza = self.vza_[hza0]
        vaa = self.vaa_[hza0]
        sza = self.sza
        saa = self.saa
        raa = np.asarray(vaa - saa)

        ind = raa>180
        if np.sum(ind)>0:
            raa[ind] = 360-raa[ind]


        rd = np.pi / 180.0
        ui = np.cos(sza * rd)
        uv = np.cos(vza * rd)
        si = np.sin(sza * rd)
        sv = np.sin(vza * rd)
        up = np.cos(raa * rd)
        tantv = np.tan(vza * rd)
        tants = np.tan(sza * rd)
        laa = 0   ### the relatvie anzimuth angle of length
        waa = laa + 90  #### the relative azimuth angle of width

        ###-------------------------------
        ### 街道的计算
        ###-------------------------------
        height1r = 0
        projv = 0
        projs = 0
        projvs = 0
        for kshape2 in range(n_shape):
            #### information of building will be projected
            shape2 = shapes[kshape2]
            length2 = shape2[0]
            width2 = shape2[1]
            height2 = shape2[2]
            alpha2 = shape2[3]
            ### pass the lower building against a reference height
            height2m1 = height2 - height1r
            if height2m1 <= 0: continue
            ### projection of roof, this is direct to its area with density
            proj_roof = alpha2 * length2 * width2
            ### projection of wall from length and width for view;    h * l * tanL * cosl + h* w*tanW*cosW
            projv_wall = alpha2 * (height2m1 * length2 * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                                   height2m1 * width2 * tantv * np.abs(np.cos((vaa - waa) * rd)))
            ### for kshape2 the projection
            projv_temp = projv_wall + proj_roof
            ### all projection accumulate
            projv = projv + projv_temp
            ### projection of wall from length and width for solar;    h * l * tanL * cosl + h* w*tanW*cosW
            projs_wall = alpha2 * (height2m1 * length2 * tants * np.abs(np.cos((saa - laa) * rd)) +
                                   height2m1 * width2 * tants * np.abs(np.cos((saa - waa) * rd)))
            #### for solar
            projs_temp = projs_wall + proj_roof
            ### accumate
            projs = projs + projs_temp

        ### calculate the sunlitfraction, this is the overlappling increatsing part, hotspot O = 0
        Overlapping = np.sqrt(tantv * tantv + tants * tants - 2 * tantv * tants * up) / (tantv + tants)
        projvs = projv + projs * Overlapping

        ### gap frequency of streat in solar and viewing directions
        pStreatV = np.exp(-projv)
        pStreatS = np.exp(-projs)
        ### sunit and shaded part
        pStreatV_sunlit = np.exp(-projvs)
        pStreatV_shaded = pStreatV - pStreatV_sunlit

        ###-------------------------------
        ### 墙壁的计算
        ###-------------------------------
        ### 半球投影的归一化
        if2 = 1.0
        ### 上半球的方向
        n_hza0 = 90
        n_haa0 = 60
        hza0 = np.linspace(0, 89, n_hza0) # hemisphere space
        haa0 = np.linspace(0, 360, n_haa0)
        hza,haa = np.meshgrid(hza0,haa0)
        hza = np.reshape(hza,-1)  ### to linear
        haa = np.reshape(haa,-1)  ### to linear

        ### step of hza in rad unit rather zhan degree unit
        dangle = np.pi/2.0/n_hza0 # 角度的步长
        tantemp = np.tan(hza * rd)   ### 半球的正切
        fweiplus = np.sin(hza*rd)*dangle *np.cos(hza*rd) # 角度的归一化方式, 一般是0.5
        # fweiplus = np.sin(hza * rd) * dangle  ##### 一般是1.0
        fweiplusSum = np.sum(if2*fweiplus)  ### 权重累计，如果没有meshgrid，应该是0.5，现在总和是0.5*n_haa0

        ### 墙壁在观测方向的投影面积
        tempSv = (np.sum(shapes[:, 3] * shapes[:, 2]  * shapes[:, 0]) * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                  np.sum(shapes[:, 3] * shapes[:, 2]  * shapes[:, 1]) * tantv * np.abs(np.cos((vaa - waa) * rd)))
        ### 屋顶的投影面积
        proj_rooff =  np.sum(shapes[:,1]*shapes[:,0]*shapes[:,3])
        ### 透过屋顶又通过墙壁的到达街道的概率
        f = np.exp(-(tempSv+proj_rooff))
        ff =  np.exp(-(proj_rooff))
        f0 = np.exp(-tempSv)
        fw = ff- f
        i0v = 1-f0
        i0v = ff-f
        fs = f
        fr = 1-ff
        # print(fr+fs+fw)
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
                tempdS = (length1 * dheight * alpha1)* tantv * np.abs(np.cos((vaa - laa) * rd)) +\
                         (width1 * dheight * alpha1)* tantv * np.abs(np.cos((vaa - waa) * rd))
                ftempv = 1.0
                ind = (shapes[:,2]-heighttemp) > 0
                if(np.sum(ind)>0):
                    proj_roof = np.sum(shapes[ind,1]*shapes[ind,0]*shapes[ind,3])
                    projStemp = np.sum(shapes[ind,3] * (shapes[ind,2]-heighttemp) *shapes[ind,0])* \
                                 tantemp * np.abs(np.cos((haa - laa) * rd)) + \
                                 np.sum(shapes[ind,3] * (shapes[ind,2]-heighttemp) *shapes[ind,1]) * \
                                 tantemp * np.abs(np.cos((haa - waa) * rd))
                    ftemp = np.exp(-(projStemp))
                    tempSv =  (np.sum(shapes[ind,3] * (shapes[ind,2]-heighttemp) *shapes[ind,0])* tantv * np.abs(np.cos((vaa - laa) * rd)) +
                            np.sum(shapes[ind,3] * (shapes[ind,2]-heighttemp) *shapes[ind,1]) * tantv * np.abs(np.cos((vaa - waa) * rd)))
                    ftempv = np.exp(-(tempSv))
                    eu = eu + np.sum(if2 *ftemp*fweiplus/ fweiplusSum) *tempdS*ftempv

                projStemp = 0
                proj_roof = 0
                ind = shapes[:, 2] < heighttemp
                if (np.sum(ind) > 0):
                    proj_roof = proj_roof + np.sum(shapes[ind, 1] * shapes[ind, 0] * shapes[ind, 3])
                    projStemp = projStemp + (np.sum(shapes[ind,3] * shapes[ind,2] *shapes[ind,0])* tantemp * np.abs(np.cos((haa - laa) * rd)) +
                        np.sum(shapes[ind,3] * shapes[ind,2] *shapes[ind,1]) * tantemp * np.abs(np.cos((haa - waa) * rd)))
                ind = shapes[:, 2] > heighttemp
                if (np.sum(ind) > 0):
                    proj_roof = proj_roof + np.sum(shapes[ind, 1] * shapes[ind, 0] * shapes[ind, 3])
                    projStemp = projStemp + (np.sum(shapes[ind, 3] * heighttemp * shapes[ind, 0]) * tantemp * np.abs(np.cos((haa - laa) * rd)) +
                                 np.sum(shapes[ind, 3] * heighttemp * shapes[ind, 1]) * tantemp * np.abs(np.cos((haa - waa) * rd)))
                ftemp = np.exp(-(projStemp+proj_roof))
                ed = ed + np.sum(if2 * ftemp * fweiplus/ fweiplusSum) * tempdS * ftempv

        eu = eu / i0v*0.5
        ed = ed / i0v*0.5
        p = 1 - eu - ed
        if i0v == 0:
            p = 0
            eu = 0
            ed = 0




        heighttemp = 0
        ws = 0
        C = 1.05
        ind = (shapes[:, 2] - heighttemp) > 0
        if (np.sum(ind) > 0):
            proj_roof = np.sum(shapes[ind, 1] * shapes[ind, 0] * shapes[ind, 3])
            projStemp = np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 0]) * tantemp * np.abs(np.cos((haa - laa) * rd)) + \
                         np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 1]) * tantemp * np.abs(np.cos((haa - waa) * rd))
            ftemp = np.exp(-(projStemp))
            tempSv = (np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 0]) * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                      np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 1]) * tantv * np.abs(np.cos((vaa - waa) * rd)))
            ftempv = np.exp(-(tempSv)) ### 到达该体元的概率
            tempds = 1-proj_roof
            ws = (1-np.sum(if2*(ftemp) * fweiplus/ fweiplusSum)) * ftempv * tempds


        wr = 0.0
        for kshape1 in range(n_shape):
            shape1 = shapes[kshape1]
            length1 = shape1[0]
            width1 = shape1[1]
            height1 = shape1[2]
            alpha1 = shape1[3]
            heighttemp = height1
            ind = (shapes[:, 2] - heighttemp) > 0
            if (np.sum(ind) > 0):
                proj_roof = np.sum(shapes[ind, 1] * shapes[ind, 0] * shapes[ind, 3])
                projStemp = (np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 0]) * tantemp * np.abs(np.cos((haa - laa) * rd)) +
                             np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 1]) * tantemp * np.abs(np.cos((haa - waa) * rd)))
                ftemp = np.exp(-(projStemp))
                tempdS = length1 * width1 * alpha1
                tempSv = (np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 0]) * tantv * np.abs(np.cos((vaa - laa) * rd)) +
                          np.sum(shapes[ind, 3] * (shapes[ind, 2] - heighttemp) * shapes[ind, 1]) * tantv * np.abs(np.cos((vaa - waa) * rd)))
                ftempv = np.exp(-(tempSv))
                wr = wr + (1-np.sum(if2 * (ftemp) * fweiplus/fweiplusSum)) * tempdS * ftempv


        eww = Ewall * (1 - Ewall) * p * i0v
        ews = Ewall * (1-Estreat) * ws
        esw = Estreat *(1-Ewall) * ed * i0v
        ewr = Ewall *(1-Eroof) * wr
        # ewr = 0
        emw = eww + ewr+ ews
        ems = esw

        ew = fw * Ewall
        es = f * Estreat
        er = (1-ff)*Eroof

        vsraa = (180.0 - np.abs(raa)) / 180.0  # sunlit part becasue of raa
        vhraa = 1 - vsraa

        ## 场景中有多少光照组分
        fts = 0
        if (sza !=0):
            area =  np.sum((shapes[:,3]*shapes[:,0] * shapes[:,2] * tants * np.abs(np.cos((saa - laa) * rd))) +
                           (shapes[:,3]*shapes[:,1] * shapes[:,2] * tants * np.abs(np.cos((saa - waa) * rd))))
            if area > 0:
                fts = (1 - np.exp(-area)) / (area)

        # eww = 0
        # ews = 0
        emw = ews + eww
        emws = emw * fts*vhraa
        emwh = emw - emws
        ems = esw
        emss = ems * pStreatS
        emsh = ems - emss

        # print(vza,vaa,wr)
        if ifP == 1:
            return emw,ems,0
        elif ifP == 2:
            return emws, emwh, emss, emsh, 0, 0
        elif ifP == 3:
            return 0, 0, 0, 0, 0, 0
        else:
            return esw + ews + eww + ewr


    def set_strcutural_input(self,shapes):
        self.shapes = shapes
        self.n_shape, self.n_dim = np.shape(shapes)
    def set_thermal_input(self,Trs,Trh,Tws,Twh,Tss,Tsh):
        self.Troof_sunlit = Trs
        self.Troof_shaded = Trh
        self.Twall_sunlit = Tws
        self.Twall_shaded = Twh
        self.Tstreat_sunlit = Tss
        self.Tstreat_shaded = Tsh

    def set_angular_input(self,vza_,vaa_,sza,saa):
        self.vaa_ = vaa_
        self.vza_ = vza_
        self.sza = sza
        self.saa = saa
        self.n_angle = np.size(vza_)

    def set_spectral_input(self,estreat,ewall,eroof):
        self.Estreat = estreat
        self.Ewall = ewall
        self.Eroof = eroof

    def calculate_effective_emissivity(self):
        emissivity_ = []
        for kangle in range(self.n_angle):
            direct = self.calculate_direct_emissivity(kangle,1)
            scatter = self.calculate_scattering_emissivity(kangle,1)
            emissivity_.append(direct + scatter)

        self.canopy_effective_emissivity = np.asarray(emissivity_ )
        return np.asarray(emissivity_ )

    def calculate_effective_component_emissivity(self,ifP = 0):
        emissivity_ = []
        for kangle in range(self.n_angle):
            direct = self.calculate_direct_emissivity(kangle,ifP)
            scatter = self.calculate_scattering_emissivity(kangle,ifP)
            direct = np.asarray(direct)
            scatter = np.asarray(scatter)
            # scatter[:] = 0
            direct[:] = 0
            emissivity_.append(direct + scatter)

        self.canopy_effective_emissivity = np.asarray(emissivity_ )
        return np.asarray(emissivity_ )