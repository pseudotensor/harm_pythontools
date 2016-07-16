def roe1():
        #
        # determine outlfow equilibrium radius, based upon integration of -v_r
        numoutflowtimes=1.0
        Tcumcut=fti
        #
        startr=20
        starti=iofr(startr)
        startj=int(ny/3.0)
        startk=0
        #
        starth=h[starti,startj,startk]
        print("starth=%g" % (starth))
        #
        # dr(r) = dr/dx1 * dx1
        drvsr=dxdxp[1,1][starti:nx,startj,startk]*_dx1
        #
        # vr = u^1/u^0 * sqrt(|g_{11}|)
        vr=uu[1]*np.sqrt(np.fabs(gv3[1,1]))/uu0
        myvr=vr[starti:nx,startj,startk]
        #
        # integrand = dr/vr
        dtgrand=drvsr/myvr
        Tcum = dtgrand.cumsum()
        iouttestlist=ti[starti:nx,startj,startk][Tcum*numoutflowtimes-Tcumcut>0]
        myi=iouttestlist[0]
        myroe=r[myi,startj,startk]
        print("myroe=%g" % (myroe))
        #
def roe2():
        #
        # determine outlfow equilibrium radius, based upon integration of -v_r
        startr=20
        starti=iofr(startr)
        startj=int(ny/3.0)
        startk=0
        #
        starth=h[starti,startj,startk]
        print("starth=%g" % (starth))
        print("starth=%g deg" % (starth*180.0/np.pi))
        #
        # dr(r) = dr/dx1 * dx1
        #dr=dxdxp[1,1]*_dx1
        #dh=dxdxp[2,2]*_dx2
        #
        # v1 = dx1/dt -> dx1 = v1*dt
        v1=uu[1]/uu0
        # v2 = dx2/dt -> dx2 = v2*dt
        v2=uu[2]/uu0
        #
        myi=starti
        myj=startj
        myk=startk
        intmyi = int(np.round(myi))
        intmyj = int(np.round(myj))
        intmyk = int(np.round(myk))
        #
        # x1a = startx1 + $dx1 * ia
        # x1b = startx1 + $dx1 * ib
        # x1b-x1a=dx1 = $dx1 * (ib-ia) = $dx1 *di
        # -> di = dx1/$dx1
        #
        # x2 = startx2 + $dx2 * j
        #
        myt=0
        #
        mydt = 0.1
        iters=t/mydt
        for ii in np.arange(0,iters):
            dx1 = mydt * v1[intmyi,intmyj,intmyk]
            dx2 = mydt * v2[intmyi,intmyj,intmyk]
            #
            di = dx1/_dx1
            dj = dx2/_dx2
            #
            myi = myi + di
            myj = myj + dj
            #
            intmyi = int(np.round(myi))
            intmyj = int(np.round(myj))
            #
            myt = myt + mydt
            #
            if(np.mod(ii,int(iters/10.0))==0):
                roe = r[intmyi,intmyj,intmyk]
                hoe = h[intmyi,intmyj,intmyk]
                print("c: roe = %g hoe = %g" % (roe,hoe))
        #
        roe = r[intmyi,intmyj,intmyk]
        hoe = h[intmyi,intmyj,intmyk]
        print("rst = %g hst = %g" % (startr,starth))
        print("roe = %g hoe = %g" % (roe,hoe))
        #
