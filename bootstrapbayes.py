import sys, os.path
import argparse
import datetime,time
import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
#import scipy.interpolate
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.api as sm
import emcee,corner

"""
Feb 25 2017: bootstrplin:
    Program to tak 2 columns one is actual other is predicted.
    Fit a line, than use bootstrap to compute intercepts/gradients
    Iput a grid as x y z list then transform list to new data as many times as  modeled with
    bootstrap. The output is the percentiles, e.g. .1, .5, and .9


~output bootstrapped linear coefs and plots
F:\sekData\Baronia\velocities>python bootstraplin.py --datafilename l3zmodbi.csv --
csv --datacols 2 9
~sample file l31zmodbi.csv
X	Y	TVD	MD	Top	Well	Residuals	Zero	ZMODDIFF	ZMODBI
1478380	1719590.05	5310.35	5421.35	atop_L3.1	BN-001	150.8430377879	0	-7.5662834788545	5317.9162834789
1478852	1724018.05	5312.13	5353.13	atop_L3.1	BN-004	196.15429145701	0	29.283861762021	5282.846138238
1479806	1718138.05	5343.2	5384.2	atop_L3.1	BN-005	134.25966254769	0	-14.541360615878	5357.7413606159
1480551.91	1721749.33	5298.51	5633.06	atop_L3.1	BN-006	149.04751480741	0	-11.314138269385	5309.8241382694
1479497.64	1720843.89	5293.58	5446.24	atop_L3.1	BN-007	152.68854879274	0	-9.3390440785834	5302.9190440786
1479108.39	1721300.79	5291.54	5400.93	atop_L3.1	BN-007S1	153.85900025372	0	-8.7926045346194	5300.3326045346
1482987.75	1720720.93	5370.14	6943.53	atop_L3.1	BN-008	126.45651839101	0	-15.591529182442	5385.7315291824
1482161.29	1720553.16	5340.75	6431.32	atop_L3.1	BN-008S2	137.67891433032	0	-12.26290485673	5353.0129048567
1479708.68	1722638.13	5296.52	5653.2	atop_L3.1	BN-009	168.28334772719	0	3.7960579971368	5292.7239420029
1479713.01	1722256.95	5299.97	5531.65	atop_L3.1	BN-009S1	167.47960512359	0	3.8191100678268	5296.1508899322
1477694.45	1721313.14	5286.09	5788.43	atop_L3.1	BN-010	135.80429484884	0	-24.397349634006	5310.487349634
1481758.89	1723048.24	5324.57	6500.3	atop_L3.1	BN-011	163.34002830081	0	5.2656057952754	5319.3043942047

~output pvalues p10 p50 and p90 using lin func from original data and agrid listing of one horizon
    F:\sekData\Baronia\velocities>python bootstraplin.py --datafilename l3zbi.csv --
csv --datacols 2 9 --gridfilename l31zmod.dat  >l31zp.txt
~output probability at z value
F:\sekData\Baronia\velocities>python bootstraplin.py --datafilename l3zmodbi.csv --
csv --datacols 2 9 --gridfilename l31zmod.dat  --zvalopt --zval 5500


~March 01, 2017 added multiplier shift functionality

F:\sekData\Baronia\velocities>python bootstraplin.py --datafilename l31zmodbi.cs
v --csv --datacols 2 9 --gridfilename l31zmod.dat  >l31_svelzp.txt

~May06 2017:
        added onepoint to plot CDF and pvals for only one point
        added xplots for predicted vs actual and actual vs error
~May07 2017:
    added creating a df from input data and saving it to csv
    added plotting of all actual vs P 10 50 90

F:\sekData\Baram\Horizons\Mar2417>python bootstraplin.py --datafilename l7wz_gtm
50a.csv --csv --datacols 4 7
Slope    Intercept
      1.11     -153.37

  F:\sekData\Baram\Horizons\I L N Horizon>python bootstraplin.py --datafilename N3
_0.csv --csv --datacols 7 8 --filter 0 2100 --gridfilename N3_0_grid --decimalpl
aces 4 >n30.txt

F:\sekData\Baram\Horizons\I L N Horizon>python bootstraplin.py --datafilename N3
_0.csv --csv --datacols 7 8 --filter 0 2100 --onepoint 1650 --decimalplaces 4


~changed name to bootstraplings for geostatistics to output eas file
D:\sekData\OriginalData\sekdata\Baram\Horizons\May1717>python bootstraplings1.py
 --datafilename i4_0.csv --csv --datacolsgs 2 3 7 8 0 --gstat

F:\sekData\Baram\Horizons\May1717>python bootstraplings.py --datafilename i4_0.c
sv --csv --datacolsgs 2 3 7 8 0 --gstat --variogram 10123 6950 0 --hide_plots

F:\sekData\Baram\Horizons\May1717>python bootstraplings.py --datafilename i4_0.c
sv --csv --datacolsgs 2 3 7 8 0 --gstat --variogram 10123 6950 0 --hide_plots  >

~May 20 2017: changed name to bootstrapgstat.py
    to output eas cmd file and generate extra pseudo points for ked

F:\sekData\Baram\Horizons\May1717>python bootstrapgstat.py --datafilename i40.cs
v --csv --gstat --datacolsgs 1 2 3 4 0 --gridfilename I4.0_pick --gridcols 2 3 4

F:\sekData\Baram\Horizons\May1717>python bootstrapgstat.py --datafilename i40.cs
v --csv --datacolsgs 1 2 3 4 0  --gridfilename I4.0_pick --gridcols 2 3 4 --gsta
t --pseudopoints --hideplots

~ use minmax scaler to adjust range of data then re scale back to original actual range
    D:\sekData\OriginalData\sekdata\Baram\Horizons\May1717>python bootstrapgstat.py
    --datafilename I2_0.csv --csv --datacols 3 4 --flipsign  --gridfilename I2_0_gri
    d --gridheader 20 --hideplots

Workflow:
    AAA. input actual & predicted columns to analyse errors and compute straight
    line transformation along with relevant statistics
    BBB. re run with a grid file to convert the grid and compute a P10/P50/P90
    columns
    CCC. re run with without grid bu t with --gstat option to generate an eas file
    to use for variogram calculation using gstatw program
    DDD. re run with --gstat option and --gridfilename to apply KED to generated
    dat file. The result will be P10/P50/p90 columns and  cmd file.
    Use cmd file to run gstat program  to tie to wells using KED.
    Usually KED will have artifacats in areas away from wells.
    EEE. re run CCC above with P50 grid if you want the other grids to tie the wells
    FFF. from gstat we get an ASCII surfer grid which needs to be converted back to xyz
    points using dat2dgs.py or grd2xyz.py
    GGG. If input actual vs predicted columns are of different units then:
        GGA. run AAA with --minmaxscaler option
        GGB. run BBB with --minmaxscaler option.
        Minmaxscaler option does not work with gstat option

~changed name to bootstrapct for bootstrap clound transform
        It basically does everything that bootstrapgstats does plus cloud transform option

    python bootstrapct.py I4_Error_PSDM.csv --csv --datacols 4 5 --flipsign --minmaxscaler --optype applyp50 --gridfilename I4_Grid_PSDM --gridheader 20 --gridminmaxscaler
"""

def qhull(sample):
    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
    	axis = sample[:,0]
    	base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
    	return link(dome(sample, base),dome(sample, base[::-1]))
    else:
        return sample


    """
    Flat file data with xy and variable (vr)
    Input polygon to list data filtered only inside polygon
    Input horizon flat file with xy only to back interpolate on data set
    Jan 23 2012

    """


def pip(x,y,poly):
   # check if point is a vertex
   """
   if (x,y) in poly:
        return True
    """
   # check if point is on a boundary
   for i in range(len(poly)):
      p1 = None
      p2 = None
      if i==0:
         p1 = poly[0]
         p2 = poly[1]
      else:
         p1 = poly[i-1]
         p2 = poly[i]
      if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
         return True

   n = len(poly)
   inside = False

   p1x,p1y = poly[0]
   for i in range(n+1):
      p2x,p2y = poly[i % n]
      if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
            if x <= max(p1x,p2x):
               if p1y != p2y:
                  xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
               if p1x == p2x or x <= xints:
                  inside = not inside
      p1x,p1y = p2x,p2y

   if inside: return True
   else: return False


#...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


def idw(xy,vr,xyi):

    N = vr.size
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 2  # weights ~ 1 / distance**p
    invdisttree = Invdisttree( xy, vr, leafsize=leafsize, stat=1 )
    interpol = invdisttree( xyi, nnear=Nnear, eps=eps, p=p )
    return interpol

def surfergrid2xyz(fname,nullvalue,mltp,nskip,lst2xyz):
    z=[]
    xy=[]
    fgrid=open(fname,'r')
    headercode= fgrid.readline()
    header1=fgrid.readline()
    xnodes,ynodes=list(map(int,header1.split()))
    #ynodes,xnodes=map(int,header1.split())
    header2=fgrid.readline()
    xmin,xmax=list(map(float,header2.split()))
    header3=fgrid.readline()
    ymin,ymax=list(map(float,header3.split()))
    header4=fgrid.readline()
    zmin,zmax=list(map(float,header4.split()))
    xinc=(xmax-xmin)/(xnodes-1)
    yinc=(ymax -ymin)/(ynodes-1)
    #print xnodes, ynodes, xmin, xmax, ymin,ymax, zmin, zmax
    #print xinc, yinc

    yc = ymin
    for j in range(ynodes):
        xc = xmin
        k= 0
        while k < xnodes:
            oneline=fgrid.readline()
            flds=oneline.split()
            nflds=len(flds)
            k +=nflds
            if nflds >0:
                for m in range(nflds):
                    if flds[m] != nullvalue:
                        zc=float(flds[m])
                        z.append(zc)
                        xy.append((xc,yc))
                        if lst2xyz == 1:
                            print("%12.2f  %12.2f  %10.3f" %(xc,yc,zc))
                    if lst2xyz == 2:
                        print("%12.2f  %12.2f  %15s" %(xc,yc,flds[m]))
                    xc += xinc
        yc+=yinc


    fgrid.close()
    xyarray=np.array(xy[::nskip][:])
    zarray=np.array(z[::nskip])
    #xyarray=np.array(xy)
    #zarray=np.array(z)
    zarray *=mltp
    return xyarray,zarray,xnodes,ynodes,xmin,xmax,ymin,ymax

def zmapgridin(fname,mltp,nskip):
    fgrid=open(fname,'r')
    header=True
    firstat=True
    while header==True:
        line=fgrid.readline()
#        print(line[:-1])
        flds=line.split()
        if (flds[0][0]=='@' and firstat):
            firstat=False
            line1=fgrid.readline()
            line1flds=line1.split(',')
            line2=fgrid.readline()
            line2flds=line2.split(',')
            nullval=line1flds[1].strip()
            nrows=int(line2flds[0])
            ncols=int(line2flds[1])
            minx=float(line2flds[2])
            maxx=float(line2flds[3])
            miny=float(line2flds[4])
            maxy=float(line2flds[5])
            dx=(maxx-minx)/ncols
            dy=(maxy-miny)/nrows
#            print( dx,dy)
        elif (flds[0][0]=='@' and firstat==False):#add check for + in zmap grid
                                                #just delete line with + from grid
            header=False
    xstart=minx
    ystart=maxy

    zval=[]
    nlines=0
#    print ("Null value %s ", nullval)
    for line in fgrid:
#        if line[0][0] != ' ':
#            continue
#9/Jul/2012 13:06 removed it and prog works OK
        lineflds=line.split()
        nlines= nlines+1
#        print(nlines,lineflds)


        for i in range(len(lineflds)):
            zval.append(lineflds[i])
    xy=[]
    z=[]
    k=-1
    for i in range(ncols):
        for j in range(nrows):
            k=k+1
#            print "k: %5d" % k
            if str(zval[k])== nullval:
                pass
#                print(zval[i+j],nullval)
            else:
                xc= xstart + i * dx
                yc = ystart - j * dy
#                print("%10.2f %10.2f %10.0f " % (float(xc),float(yc), float(zval[k])))
                xy.append((xc,yc))
                z.append(float(zval[k]))
#    print("len of xyz:",len(xyz))
    fgrid.close()
    #xyarray=np.array(xy[::nskip][::nskip])
    xyarray=np.array(xy[::nskip][:])
    zarray=np.array(z[::nskip])
    zarray *=mltp
    return xyarray,zarray,ncols,nrows,minx,maxx,miny,maxy

def create_surfergrid(limits,xy,vr):
    minx=limits[0]
    maxx=limits[1]
    dx=limits[2]
    miny=limits[3]
    maxy=limits[4]
    dy=limits[5]
    #print "limits",limits
    nx = int((maxx - minx)/dx)
    ny= int((maxy - miny )/dy)
    xc,xstep= np.linspace(minx,maxx,nx,endpoint=True,retstep=True)
    yc,ystep=np.linspace(miny,maxy,ny,endpoint=True,retstep=True)
    #print "xstep: %10.3f, ystep: %10.3f  nx= %10d   ny=%10d" %(xstep,ystep,nx,ny)
    #xyi= np.arange(xc.size*yc.size).reshape(yc.size,xc.size)
    #print xyi.size
    xyi=[]
    for i in range(yc.size):
        for j in range(xc.size):
            xyi.append((xc[j],yc[i]))
    zc=idw(xy,vr,xyi)
    return nx,ny,zc

def list_surfergrid(limits,nx,ny,zc):
    minx=limits[0]
    maxx=limits[1]
    miny=limits[3]
    maxy=limits[4]
    print("DSAA")
    print("%-10d  %-10d" %(nx,ny))
    print("%-12.0f  %-12.0f" %(minx,maxx))
    print("%-12.0f  %-12.0f" %(miny,maxy))
    print("%-15.5f  %-15.5f" %(zc.min(),zc.max()))

    for i in range(zc.size):
            print("% 15s" % zc[i])



def to_surfergrid(limits,nx,ny,zc,varname):
    f0=open(varname +'.grd',"w")
    minx=limits[0]
    maxx=limits[1]
    miny=limits[3]
    maxy=limits[4]
    print("DSAA", file=f0)
    print("%-10d  %-10d" %(nx,ny), file=f0)
    print("%-12.0f  %-12.0f" %(minx,maxx), file=f0)
    print("%-12.0f  %-12.0f" %(miny,maxy), file=f0)
    print("%-15.5f  %-15.5f" %(zc.min(),zc.max()), file=f0)

    for i in range(zc.size):
            print("% 15s" % zc[i], file=f0)
    f0.close()



def togstat_eas(xc,yc,zc,zr,wname,fname,dp=2):
    #f0=open(fname + 'bi.eas','w')
    f0=open(fname ,'w')
    now = datetime.datetime.now()
    print("# %s" %(fname), end=' ', file=f0)
    print(now.strftime("%Y-%b-%d  %H:%M"), file=f0)
    print("6", file=f0)
    print("XC", file=f0)
    print("YC", file=f0)
    print("VR", file=f0)
    print("VRBI", file=f0)
    print("DIFF", file=f0)
    print("WELL", file=f0)
    for i in range (len(xc)):
        dz= zc[i] -zr[i]
        print("%12.2f  %12.2f  %10.*f  %10.*f  %10.*f  %20s" % \
        (xc[i],yc[i],dp,zc[i],dp,zr[i],dp,dz,wname[i]), file=f0)
    f0.close()


def filterhullpolygon(x,y,polygon):
    xf=[]
    yf=[]
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
    return np.array(xf),np.array(yf)



def distxy(x1,y1,x2,y2):
    r = m.sqrt(((x1-x2)**2) + ((y1-y2)**2))
    return r



def distcomp3(x1,y1,x2,y2,ninc1=10,filtdist=1000.0):
    x1g = np.linspace(x1.min(),x1.max(),ninc1)
    y1g = np.linspace(y1.min(),y1.max(),ninc1)
    dx = (x1.max()-x1.min()) /ninc1
    dy = (y1.max() - y1.min()) / ninc1
    print('xmin',x1.min(),'xmax',x1.max(),'ymin',y1.min(),'ymax',y1.max())
    print('dx ',dx,'dy',dy)
    pgn = []
    for j in range(x2.size):
        pgn.append((x2[j],y2[j]) )
    #pgn = np.transpose(np.vstack((x2,y2)))
    #pgn =list(zip(x2.tolist(),y2.tolist()))
    #pth = mplpath.Path(pgn)
    print('pgn',len(pgn))
    #print (pgn)
    #filter based on point in polygon
    xc=[]
    yc = []
    for i in range(x1g.size):
        xi = x1g.min() + i * dx
        for j in range(len(y1g)):
           yi = y1g.min() + j * dy
           xc.append(xi)
           yc.append(yi)
           """
           if pip(xi,yi,pgn):
           #print('Point:',xi,yi)
           #if pth.contains_point([xi,yi]):
               xc.append(xi)
               yc.append(yi)
               cindx+=1
    print('inside len: ',len(xc),'Total points:',cindx)
    """
    xca = np.array(xc)
    yca = np.array(yc)

    #x1 y1 are for pseudo points to compute
    #x2 y2 are well coords or control points
    #filter based on how far point is from control points

    xf =[]
    yf =[]
    ok = True
    for i in range(xca.size):
        for j in range(x2.size):
            ok = True
            r =distxy(xca[i],yca[i],x2[j],y2[j])
            #print('r:',r)
            if r <= filtdist:
                ok = False
                break
        if ok:
            xf.append(xca[i])
            yf.append(yca[i])
        else:
            ok = True

    print('len of dist filtered points ',len(xf))
    return np.array(xf),np.array(yf)

    #return xca, yca

def dataavgmap(xy,vr,xyi,radius):
    tree=KDTree(xy)
    vravg= np.arange(xyi[:,0].size)
    for i in range(xyi[:,0].size):
        npts=tree.query_ball_point([xyi[i,0],xyi[i,1]],radius)
        #print [xyi[j] for j in np]
        vrsum=0
        for j in npts:
            vrsum += vr[j]
        if len(npts) > 0:
            vravg[i]= vrsum/len(npts)
        else:
            vravg[i]=vrsum
        print("%-10d  xyi: %12.2f  %12.2f  Points: %5d %10.3f " % (i,xyi[i,0],xyi[i,1],len(npts),vravg[i]))
    return vravg




def map2ddata(xy,vr,xyi,radius,maptype):
    stats=sts.describe(vr)
    # statsstd=sts.tstd(vr)
    if maptype == 'idw':
        vri=idw(xy,vr,xyi)
    elif maptype =='nearest':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='nearest')
    elif maptype == 'linear':
        #                vri=griddata(xy,vr,(xyifhull[:,0],xyifhull[:,1]),method='linear')
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='linear')
    elif maptype == 'cubic':
        vri=griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='cubic')
    elif maptype =='rbf':
        rbf=Rbf(xy[:,0],xy[:,1],vr)
        vri= rbf(xyi[:,0],xyi[:,1])
    elif maptype =='avgmap':
        vri=dataavgmap(xy,vr,xyi,radius)
    elif maptype =='triang':
        linearnd=LinearNDInterpolator(xy,vr,stats[2])
        vri= linearnd(xyi)
    elif maptype == 'ct':
        ct=CloughTocher2DInterpolator(xy,vr,stats[2])
        vri=ct(xyi)
    return vri


def ked_gstatcmd(sill,range,easfname,maskname):
    dirsplit,fextsplit= os.path.split(easfname)
    fname,fextn= os.path.splitext(fextsplit)
    #easfname = os.path.join(dirsplit,fname)+'bi.eas'



    f0=open(fname +'.cmd','w')
    now = datetime.datetime.now()
    #prname=[]
    #fname=[]
    #fname=varname +'bi.lst'
    #prname=maskname +'ked'
    prname=fname +'ked'
    vrname = fname +'vr'
    print("# ", file=f0)
    print("# ", end=' ', file=f0)
    print(now.strftime("%Y-%b-%d  %H:%M"), file=f0)
    print("# ", file=f0)
    print("data(%s):'%s',x=1, y=2, v=3, X=4, average;" % (fname,easfname), file=f0)
    print("variogram(%s): %10.0f  Sph(%10.0f);" %(fname,sill,range), file=f0)
    print("mask: '%s';" % (maskname), file=f0)
    print("predictions(%s): '%s';" %(fname,prname), file=f0)
    print("variances(%s): '%s';" %(fname,vrname), file=f0)
    f0.close()




def errprint(*args,**kwargs):
    print(*args,file=sys.stderr,**kwargs)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: n
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y



def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

def rsqr(x,xi):
    mean_sum_sqrerr = np.sum((x -np.mean(x))**2)
    regr_sum_sqrerr = np.sum((x - xi)**2)
    r2 = (mean_sum_sqrerr - regr_sum_sqrerr)/ mean_sum_sqrerr
    return (mean_sum_sqrerr,regr_sum_sqrerr,r2)



def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    # logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)

# Define our posterior using Python functions
# for clarity, I've separated-out the prior and likelihood
# but this is not necessary. Note that emcee requires log-posterior

def log_prior(theta):
    alpha, beta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2) - np.log(sigma)

def log_likelihood(theta, x, y):
    alpha, beta, sigma = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

#**************************
def bayesonly(x,y,dp=2,ndim=3,nwalkers=50,nburn=1000,nsteps=2000):
    # Here's the function call where all the work happens:
    # we'll time it using IPython's %time magic
    print('Starting MCMC sampling.......')
    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[y, x])
    sampler.run_mcmc(starting_guesses, nsteps)
    emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

    bayesicpt = emcee_trace[0].mean()
    slope = emcee_trace[1].mean()
    # sigma = emcee_trace[2].mean()
    # samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    #emcee_trace[0] is intercept
    #emcee_trace[1] is slope
    print('Bayesian: Mean Intercept  {:.{prec}f} Mean Slope {:.{prec}f}'.format(bayesicpt,slope,prec=dp))
    print('emcee_trace shape: {}'.format(emcee_trace.shape))
    return (emcee_trace)



def bsonly(x,y,n=1000,dp=2,pvals=[10,50,90]):


    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(x, y, size=n)

    #x is predicted, y is actual
    # print('Bootstrap: Intercept  {:.{prec}f} Slope {:.{prec}f}'.format(icpt,s,prec=dp))
    bsicpt=np.percentile(bs_intercept_reps, 50)
    bsslope=np.percentile(bs_slope_reps, 50)
    print('Bootstrap: Mean Intercept  {:.{prec}f} Mean Slope {:.{prec}f}'.format(bsicpt,bsslope,prec=dp))
    return  (bsslope,bsicpt)





#******************************

def apply_bootstrap(x,y,n=1000,confidence_int=[2.5,97.5],pvals=[10,50,90],dp=2,pdffname=None,hide_plots=False,
    ndim=3,nwalkers=50,nburn=1000,nsteps=2000):



    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(x, y, size=n)

    #x is predicted, y is actual

    # set theta near the maximum likelihood, with
    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim))





    with PdfPages(pdffname) as pdf:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        _ = ax1.hist(y, bins=20, normed=True)
        _ = plt.xlabel('Actual Variable')
        _ = plt.ylabel('PDF')
        _ = plt.title('Actual Variable Distribution')

        ax2 = fig.add_subplot(122)
        _ = ax2.hist(x, bins=20, normed=True)
        _ = plt.xlabel('Predicted Variable')
        _ = plt.ylabel('PDF')
        _ = plt.title('Predicted Variable Distribution')



        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        plt.boxplot([y,x],labels=['Actual','Predicted'],showmeans=True,notch=True)
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()



        # Compute and print 95% CI for slope
        #print(np.percentile(bs_slope_reps, confidence_int))
        #print(np.percentile(bs_intercept_reps, confidence_int))

        # Plot the histogram : Bootstrap Slope
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        _ = ax1.hist(bs_slope_reps, bins=50, normed=True)
        _ = plt.xlabel('slope')
        _ = plt.ylabel('PDF')
        _ = plt.title('Bootstrap of Slope')
        #plt.show()
        xecdf,yecdf = ecdf(bs_slope_reps)
        ax2 = fig.add_subplot(122)
        _ = ax2.hist(bs_slope_reps, bins=50, normed=True,cumulative=True)
        _ = ax2.plot(xecdf,yecdf,lw=3,color='m')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        # Plot the histogram Bootstrap intercept
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        _ = ax1.hist(bs_intercept_reps, bins=50, normed=True)
        _ = plt.xlabel('intercept')
        _ = plt.ylabel('PDF')
        _ = plt.title('Bootstrap of Intercept')
        xecdf,yecdf = ecdf(bs_intercept_reps)
        ax2 = fig.add_subplot(122)
        _ = ax2.hist(bs_intercept_reps, bins=50, normed=True,cumulative=True)
        _ = ax2.plot(xecdf,yecdf,lw=3,color='m')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        #Plot predicted vs actual
        plt.scatter(x,y,c='red',alpha=.3)
        xi = np.linspace(x.min(),x.max())
        s,icpt = np.polyfit(x,y,1)
        xactual = np.polyval((s,icpt),xi)
        for i in range(len(bs_slope_reps)):
            plt.plot(xi,bs_slope_reps[i] *xi + bs_intercept_reps[i],lw=0.5,alpha=.2,c='g')
        plt.plot(xi,xactual,c='red',lw=3)

        plt.annotate('Vactual = %-.*f   + %-.*f * Vpredicted' % (dp,icpt,dp,s),\
                     xy =(xi[4],xactual[4]),xytext=(0.25,0.80),textcoords='figure fraction')
        plt.xlabel('Predicted Variable')
        plt.ylabel('Actual Variable')
        plt.title('Predicted vs Actual w/Bootstrapped Lines')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        #errors (Actual - st line fitting predicted)
        vi = y - np.polyval((s,icpt),x)

        #Plot Actual vs st line fitting prediction errors
        plt.scatter(y,vi,c='red',alpha=.3)
        plt.xlabel('Actual Variable')
        plt.ylabel('Predicted linear fn Variable Error')
        plt.title('Actual vs Predicted error')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()


        xpipred, ypipred = ecdf(x)
        xpiwell, ypiwell = ecdf(y)
        # Generate plot
        _ = plt.plot(xpipred, ypipred, marker='.', linestyle='none',label='Predicted')
        _ = plt.plot(xpiwell, ypiwell, marker='.', linestyle='none',color='red',label='Actual')

        # Make the margins nice
        plt.margins(0.02)

        # Label the axes
        plt.xlabel('Actual & Predicted ')
        plt.ylabel('ECDF')
        plt.title('ECDF ACTUAL VS PREDICTED COMPARISON')
        plt.legend()
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        fig = sm.qqplot(x,fit=True,line='45')
        plt.title('Q-Q Plot of Actual ')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        fig = sm.qqplot(y,fit=True,line='45')
        plt.title('Q-Q Plot of Predicted ')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        pp_x = sm.ProbPlot(x, fit=True)
        pp_y = sm.ProbPlot(y, fit=True)
        pp_x.qqplot(line='45', other=pp_y)
        plt.title('P-P Plot ACTUAL vs PREDICTED PROBABILITIES')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()


        stderrors = sts.zscore(vi)
        actual_range = [y.min(),y.max()]
        #err_range = [stderrors.min(),stderrors.max()]
        plt.scatter(y,stderrors,c='red',alpha=.3)
        plt.plot([int(actual_range[0]),round(actual_range[1],0)],[0,0],'-',c='b',lw=2,label='Mean Residual')
        plt.plot([int(actual_range[0]),round(actual_range[1],0)],[3,3],'--',c='b',lw=2,label='Upper Bound')
        plt.plot([int(actual_range[0]),round(actual_range[1],0)],[-3,-3],'--',c='b',lw=2,label='Lower Bound')
        plt.xlabel('Actual Values')
        plt.ylabel('Standardized Errors')
        plt.title('Z Scored Errors vs Actual Values')
        #plt.legend(loc='lower left',bbox_to_anchor=(0.3,0.5))
        plt.legend(loc='lower right')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()


        vout=np.empty(n)
        vpval = np.empty([x.size,3])
        vpvaldiff = np.empty([x.size,3])
    #    print('X   Y   VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
    #          (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]))
        for i in range(x.size):
            for j in range(n):
                vout[j]= x[i] * bs_slope_reps[j] + bs_intercept_reps[j]
            vpval[i]= np.percentile(vout,pvals)
            #subtract from actual values, predicted is x
            vpvaldiff[i]= vpval[i] -y[i]
    #        print('%10.0f  %10.0f  %10.0f  %10.0f  %10.1f  %10.1f  %10.1f' %\
    #              (x[i],vpval[i,0],vpval[i,1],vpval[i,2],vpvaldiff[i,0],vpvaldiff[i,1],vpvaldiff[i,2]))

        bscols = ['VIN','VPRED','VP%-d'% pvals[0],'VP%-d'% pvals[1],'VP%-d' % pvals[2],\
                  'VP%-dDIFF'% pvals[0],'VP%-dDIFF'% pvals[1],'VP%-dDIFF'% pvals[2]]
        bsdata = pd.DataFrame({'VIN':y,'VPRED':x,'VP%-d'% pvals[0]:vpval[:,0],\
                               'VP%-d'%pvals[1]:vpval[:,1],'VP%-d'% pvals[2]:vpval[:,2],\
                               'VP%-dDIFF' % pvals[0]:vpvaldiff[:,0],\
                               'VP%-dDIFF'% pvals[1]:vpvaldiff[:,1],\
                               'VP%-dDIFF'% pvals[2]:vpvaldiff[:,2]})
        bsdata = bsdata[bscols].copy()
    #    print(bsdata.describe())

        #Plot Actual vs Predicted probabilities P10, P50 and P90
        plt.scatter(vpval[:,0],y,c='red',alpha=.6,s=5,label='P%-d' % pvals[0])
        plt.scatter(vpval[:,1],y,c='g',alpha=.6,s=5,label='P%-d'% pvals[1])
        plt.scatter(vpval[:,2],y,c='b',alpha=.6,s=5,label='P%-d'% pvals[2])
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values : P:%-d/%-d/%-d' % (pvals[0],pvals[1],pvals[2]))
        plt.title('Bootstrapped Probabilities')
        plt.legend(loc='lower right')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()
        d =pdf.infodict()
        d['Title'] = 'Bootstrap Analysis '
        d['Author'] = 'Sami Elkurdy'
        d['CreationDate'] = datetime.datetime.today()

        # Here's the function call where all the work happens:
        # we'll time it using IPython's %time magic
        print('Starting MCMC sampling.......')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[y, x])
        sampler.run_mcmc(starting_guesses, nsteps)
        emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

        bayesicpt = emcee_trace[0].mean()
        # print('Intercept Mean {:.{prec}f}'.format(bayesicpt,prec=dp))
        plt.plot((0,emcee_trace.shape[1]),(bayesicpt,bayesicpt),lw=3,c='g')
        plt.plot(emcee_trace[0,:],alpha=0.5)
        plt.title('MCMC Trace for Intercept')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        slope = emcee_trace[1].mean()
        # print('Slope Mean {:.{prec}f}'.format(slope,prec=dp))
        plt.plot((0,emcee_trace.shape[1]),(slope,slope),lw=3,c='g')
        plt.plot(emcee_trace[1,:],alpha=0.5)
        plt.title('MCMC Trace for Slope')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        sigma = emcee_trace[2].mean()
        # print('Sigma Mean {:.{prec}f}'.format(sigma,prec=dp))
        plt.plot((0,emcee_trace.shape[1]),(sigma,sigma),lw=3,c='g')
        plt.plot(emcee_trace[2,:],alpha=0.5)
        plt.title('MCMC Trace for Sigma')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        # Plot the histogram : Bayesian Intercept
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.hist(emcee_trace[0,:], bins=50, normed=True)
        plt.xlabel('Intercept')
        plt.ylabel('PDF')
        plt.title('Bayesian Intercept')
        #plt.show()
        xecdf,yecdf = ecdf(emcee_trace[0,:])
        ax2 = fig.add_subplot(122)
        ax2.hist(emcee_trace[0,:], bins=50, normed=True,cumulative=True)
        ax2.plot(xecdf,yecdf,lw=3,color='m')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        # Plot the histogram Bayesian Slope
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.hist(emcee_trace[1,:], bins=50, normed=True)
        plt.xlabel('Slope')
        plt.ylabel('PDF')
        plt.title('Bayesian Slope')
        xecdf,yecdf = ecdf(emcee_trace[1,:])
        ax2 = fig.add_subplot(122)
        ax2.hist(emcee_trace[1,:], bins=50, normed=True,cumulative=True)
        ax2.plot(xecdf,yecdf,lw=3,color='m')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()



        # emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
        plot_MCMC_results(y, x, emcee_trace)
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)
        fig = corner.corner(samples, labels=["$Intercept$", "$Slope$", "$\sigma\,f$"],
                              truths=[slope,icpt,sigma])
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()


        bs_out= np.zeros(10)
        bs_out[0]=np.percentile(bs_intercept_reps, pvals[0])
        bs_out[1]=np.percentile(bs_intercept_reps, pvals[1])
        bs_out[2]=np.percentile(bs_intercept_reps, pvals[2])
        bs_out[3]=np.percentile(bs_slope_reps, pvals[0])
        bs_out[4]=np.percentile(bs_slope_reps, pvals[1])
        bs_out[5]=np.percentile(bs_slope_reps, pvals[2])
        bs_out[6]=np.percentile(bs_intercept_reps, confidence_int[0])
        bs_out[7]=np.percentile(bs_intercept_reps, confidence_int[1])
        bs_out[8]=np.percentile(bs_slope_reps, confidence_int[0])
        bs_out[9]=np.percentile(bs_slope_reps, confidence_int[1])

        by_out= np.zeros(6)
        by_out[0]=np.percentile(emcee_trace[0], pvals[0])#intercept
        by_out[1]=np.percentile(emcee_trace[0], pvals[1])
        by_out[2]=np.percentile(emcee_trace[0], pvals[2])
        by_out[3]=np.percentile(emcee_trace[1], pvals[0])#slope
        by_out[4]=np.percentile(emcee_trace[1], pvals[1])
        by_out[5]=np.percentile(emcee_trace[1], pvals[2])



        byvpval = np.empty([x.size,3])
        byvpvaldiff = np.empty([x.size,3])
    #    print('X   Y   VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
    #          (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]))
        for i in range(x.size):
            byvpval[i,0]= by_out[3] * x[i] + by_out[0]  #p10 of slope and intercept
            byvpval[i,1]= by_out[4] * x[i] + by_out[1]  #p50 of slope and intercept
            byvpval[i,2]= by_out[5] * x[i] + by_out[2]  #p90 of slope and intercept
            #subtract from actual values, predicted is x
            byvpvaldiff[i]= byvpval[i] -y[i]
    #        print('%10.0f  %10.0f  %10.0f  %10.0f  %10.1f  %10.1f  %10.1f' %\
    #              (x[i],vpval[i,0],vpval[i,1],vpval[i,2],vpvaldiff[i,0],vpvaldiff[i,1],vpvaldiff[i,2]))

        bycols = ['VIN','VPRED','BAYESVP%-d'% pvals[0],'BAYESVP%-d'% pvals[1],'BAYESVP%-d' % pvals[2],\
                  'BAYESVP%-dDIFF'% pvals[0],'BAYESVP%-dDIFF'% pvals[1],'BAYESVP%-dDIFF'% pvals[2]]
        bydata = pd.DataFrame({'VIN':y,'VPRED':x,'BAYESVP%-d'% pvals[0]:byvpval[:,0],\
                               'BAYESVP%-d'%pvals[1]:byvpval[:,1],'BAYESVP%-d'% pvals[2]:byvpval[:,2],\
                               'BAYESVP%-dDIFF' % pvals[0]:byvpvaldiff[:,0],\
                               'BAYESVP%-dDIFF'% pvals[1]:byvpvaldiff[:,1],\
                               'BAYESVP%-dDIFF'% pvals[2]:byvpvaldiff[:,2]})
        bydata = bydata[bycols].copy()





        #Plot predicted vs actual
        plt.scatter(x,y,c='blue',alpha=.3,label='Input')
        xi = np.linspace(x.min(),x.max())
        yols = np.polyval((s,icpt),xi) #OLS fitting
        ybs = np.polyval((bs_out[4],bs_out[1]),xi) #bootstrap fitting
        yby = np.polyval((slope,bayesicpt),xi)   #bayesian fitting
        plt.plot(xi,yols,c='red',lw=3,label='OLS')
        plt.plot(xi,ybs,c='green',lw=3,label='Bootstrap')
        plt.plot(xi,yby,c='magenta',lw=3,label='Bayesian')
        plt.title('Bayesian vs OLS Line fitting Comparison')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.legend(loc='lower right')
        plt.annotate('Vactual = %-.*f   + %-.*f * Vpredicted' % (dp,bayesicpt,dp,slope),\
                     xy =(xi[4],xactual[4]),xytext=(0.25,0.80),textcoords='figure fraction')
        pdf.savefig()
        if not hide_plots:
            plt.show()
        plt.close()

        #return p50 for the bootstrap slope and intercept
    # return  bs_out,(bs_out[4],bs_out[1]),bsdata,(slope,bayesicpt,sigma)
    return  bs_out,(bs_out[4],bs_out[1]),bsdata,(by_out[4],by_out[1],sigma),bydata


def listsim(xsim):
    for i in range(xsim.size):
        print("%10.4f " % xsim[i])

#list x y kriged values, and kriged variance
def listxyzv(x,y,z,v,dp=2):
    for i in range(len(z)):
        print('%12.2f  %12.2f  %10.*f  %10.*f' %(x[i],y[i],dp,z[i],dp,v[i]))

def listpseudop(x,y,fn,dp=2):
    for i in range(len(x)):
        print('%12.*f  %12.*f  100 ' %(dp,x[i],dp,y[i]),file=fn)

# This function is based on an old version to calculate cdf and inverse cdf
# Need to change it to np.percentile
def ecompzval(xin,zval,exceedence):
    # N= xin.size
    # xs= np.sort(xin)
    # f , inv = ecdf(xs)

    # h = fit(f)
    # p=[ (h(xs[i])) for i in range(N)]
    # #print len(p)
    # #for n in range(len(p)):
    #     #print "%10.3f  %10.3f" %(xs[n],p[n])
    # xinv = fit(inv)


    # pval = h(zval)
    pval= sts.percentileofscore(xin,zval)

    if exceedence:
        pvalatz = (1.0 -pval) *100.0
    else:
        pvalatz=pval *100.0
    return pvalatz

def gridlistin(fname,xyvcols=[0,1,2],nheader=0): #used for single coef per file
    xyv=np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    #filter surfer null values by taking all less than 10000, arbitrary!!
    xyv = xyv[xyv[:,2]<10000.0]
    #xya = xya[~xya[:,2]==  missing]
    return xyv[:,0],xyv[:,1],xyv[:,2]

#vin0 is zpredicted, vin1 is z original
def process_pval_grid(x,y,v,pvals,vin0,vin1,bsgridfname,n=1000,dp=2,usebayes=False,bayesdownsample=10,
        hideplots=False):


    vpval = np.empty([x.size,3])
    vpvaldiff = np.empty([x.size,3])
    if usebayes:
        emcee_trace = bayesonly(vin0,vin1,dp=dp)
        emcee_tracex = emcee_trace[:,::bayesdownsample]
        vout=np.empty(emcee_tracex.shape[1])

        if not hideplots:

            # Plot the histogram : Bayesian Intercept
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.hist(emcee_tracex[0,:], bins=50, normed=True)
            plt.xlabel('Intercept')
            plt.ylabel('PDF')
            plt.title('Bayesian Intercept')
            #plt.show()
            xecdf,yecdf = ecdf(emcee_tracex[0,:])
            ax2 = fig.add_subplot(122)
            ax2.hist(emcee_tracex[0,:], bins=50, normed=True,cumulative=True)
            ax2.plot(xecdf,yecdf,lw=3,color='m')
            plt.show()

            # Plot the histogram Bayesian Slope
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.hist(emcee_tracex[1,:], bins=50, normed=True)
            plt.xlabel('Slope')
            plt.ylabel('PDF')
            plt.title('Bayesian Slope')
            xecdf,yecdf = ecdf(emcee_tracex[1,:])
            ax2 = fig.add_subplot(122)
            ax2.hist(emcee_tracex[1,:], bins=50, normed=True,cumulative=True)
            ax2.plot(xecdf,yecdf,lw=3,color='m')
            plt.show()



        print('X   Y   VIN   BAYESV_P%-d   BAYESV_P%-d   BAYESV_P%-d  BAYESVDIF_P%-d   BAYESVDIF_P%-d   BAYESVDIF_P%-d' %\
              (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]),file=bsgridfname)
        print('# of samples in trace {}'.format(emcee_tracex.shape[1]))

        for i in range(x.size):
            for j in range(emcee_tracex.shape[1]):
                vout[j]= v[i] * emcee_tracex[1,j] + emcee_tracex[0,j]
                # print('In :',j,vout[j],emcee_tracex[1,j],emcee_tracex[0,j])
            vpval= np.percentile(vout,pvals)

            vpvaldiff= vpval -v[i]
            print('%12.2f  %12.2f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f' %\
                  (x[i],y[i],dp,v[i],dp,vpval[0],dp,vpval[1],dp,vpval[2],dp,vpvaldiff[0],\
                   dp,vpvaldiff[1],dp,vpvaldiff[2]),file=bsgridfname)

    else:
        bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(vin0, vin1, size=n)
        vout=np.empty(n)
        print('X   Y   VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
              (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]),file=bsgridfname)
        for i in range(x.size):
            for j in range(n):
                vout[j]= v[i] * bs_slope_reps[j] + bs_intercept_reps[j]
            vpval= np.percentile(vout,pvals)
            vpvaldiff= vpval -v[i]
            #vpval= sts.percentileatscore(vout,pvals)
            #to test scipy function instead of numpy
            print('%12.2f  %12.2f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f' %\
                  (x[i],y[i],dp,v[i],dp,vpval[0],dp,vpval[1],dp,vpval[2],dp,vpvaldiff[0],\
                   dp,vpvaldiff[1],dp,vpvaldiff[2]),file=bsgridfname)



def process_scaled_pval_grid(x,y,v,pvals,vin0,vin1,bsgridfname,*minmax,n=1000,dp=2,scaleout=False,
        usebayes=False):
    """
    Process minmax scaler
    minmax should have data min & data max
    """
    if usebayes:
        bayespval = bayesonly(vin0,vin1,dp=dp,pvals=pvals)
        vpval = np.empty([x.size,3])
        vscaled = np.empty(x.size)
        v1 = v.reshape(-1,1)
        vscaled = MinMaxScaler(feature_range=(minmax)).fit_transform(v1)
        print('X   Y   VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
              (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]),file=bsgridfname)
        vpvaldiff = np.array([x.size,3])
        for i in range(x.size):
            vpval[i,0]= v[i] * bayespval[3] + bayespval[0]
            vpval[i,1]= v[i] * bayespval[4] + bayespval[1]
            vpval[i,2]= v[i] * bayespval[5] + bayespval[2]
            # vpvaldiff= vscaled - vpval
            vpvaldiff[i,0]= vpval[i,0] -vscaled[i]
            vpvaldiff[i,1]= vpval[i,1] -vscaled[i]
            vpvaldiff[i,2]= vpval[i,2] -vscaled[i]

        for i in range(x.size):
            print('%12.2f  %12.2f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f' %
                    (x[i],y[i],dp,vscaled[i],dp,vpval[i,0],dp,vpval[i,1],dp,vpval[i,2],
                    dp,vpvaldiff[i,0],dp,vpvaldiff[i,1],dp,vpvaldiff[i,2]),file=bsgridfname)
    else:
        bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(vin0, vin1, size=n)
        bsicpt=np.percentile(bs_intercept_reps, 50)
        bsslope=np.percentile(bs_slope_reps, 50)
        print('X   Y   VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
              (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]),file=bsgridfname)
        vout=np.empty(n)
        v1 = v.reshape(-1,1)
        vscaled = MinMaxScaler(feature_range=(minmax)).fit_transform(v1)
        vpval = np.empty([x.size,3])
        vscaled = np.empty(x.size)
        for i in range(x.size):
            for j in range(n):
                vout[j]= v[i] * bs_slope_reps[j] + bs_intercept_reps[j]
            vpval[i,:]= np.percentile(vout,pvals)
            if scaleout:
                vpval = MinMaxScaler(feature_range=(minmax)).fit_transform(vpval)
        vpvaldiff= vscaled - vpval

        for i in range(x.size):
            print('%12.2f  %12.2f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f' %
                    (x[i],y[i],dp,vscaled[i],dp,vpval[i,0],dp,vpval[i,1],dp,vpval[i,2],
                    dp,vpvaldiff[i,0],dp,vpvaldiff[i,1],dp,vpvaldiff[i,2]),file=bsgridfname)




def process_zval_grid(x,y,v,zval,vin0,vin1,outfname,dp=2, n=1000,exceedence=False):
    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(vin0, vin1, size=n)
    vout=np.empty(n)
    vpval = np.empty(x.size)
    print('X   Y   VIN   Prob of Z <%-d   ' % (zval))
    for i in range(x.size):
        for j in range(n):
            vout[j]= v[i] * bs_slope_reps[j] + bs_intercept_reps[j]
        vpval[i] = sts.percentileofscore(vout,zval)
        # checking for exceedence one value at a time
        if exceedence:
            print('%12.2f  %12.2f  %10.*f  %10.2f' %(x[i],y[i],dp,v[i],dp,1.0 - vpval[i]), file=outfname )
        else:
            print('%12.2f  %12.2f  %10.*f  %10.2f' %(x[i],y[i],dp,v[i],dp,vpval[i]), file=outfname )
    # checking for exceedence after the whole array is generated. It is then saved as a dataframe
    # if exceedence:
    #     vpval = (1.0 -vpval) *100.0
    # else:
    #     vpval = vpval * 100.0
    # return vpval


def process_onepoint_pval(v,pvals,vin0,vin1,n=1000,dp=2,pdffname=None):
    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(vin0, vin1, size=n)
    vout=np.empty(n)
    for j in range(n):
        vout[j]= v * bs_slope_reps[j] + bs_intercept_reps[j]
    vpval= np.percentile(vout,pvals)
    vpvaldiff= vpval -v
        #vpval= sts.percentileatscore(vout,pvals)
        #to test scipy function instead of numpy
    with PdfPages(pdffname) as pdf:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.hist(vout, bins=50, normed=True)
        plt.xlabel('Predicted Value')
        plt.ylabel('PDF')
        plt.title('Bootstrap of Predicted Value')
        #plt.show()
        xecdf,yecdf = ecdf(vout)
        ax2 = fig.add_subplot(122)
        ax2.hist(vout, bins=50, normed=True,cumulative=True,alpha=.4)
        ax2.plot(xecdf,yecdf,lw=3,color='m')
        ax2.scatter(vpval,np.array(pvals)/100,c='red',s=50)
        pvalsa = np.array(pvals)/100
        for i in range(len(pvals)):
            ax2.plot([xecdf.min(),vpval[i],vpval[i]],[pvalsa[i],pvalsa[i],yecdf.min()],lw=3,c='red')

        pdf.savefig()
        plt.show()
        plt.close()
        d =pdf.infodict()
        d['Title'] = 'One Point POrediction Bootstrap Analysis '
        d['Author'] = 'Sami Elkurdy'
        d['CreationDate'] = datetime.datetime.today()


    print('VIN   V_P%-d   V_P%-d   V_P%-d  VDIF_P%-d   VDIF_P%-d   VDIF_P%-d' %\
          (pvals[0],pvals[1],pvals[2],pvals[0],pvals[1],pvals[2]))

    print('%10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f  %10.*f' %\
          (dp,v,dp,vpval[0],dp,vpval[1],dp,vpval[2],dp,vpvaldiff[0],dp,vpvaldiff[1],dp,vpvaldiff[2]))


def process_onepoint_zval(v,zval,vin0,vin1,n=1000,dp=2):
    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(vin0, vin1, size=n)
    vout=np.empty(n)
    print('VIN   Prob of Z <%-d   ' % (zval))
    for j in range(n):
        vout[j]= v * bs_slope_reps[j] + bs_intercept_reps[j]
    vpval= sts.percentileofscore(vout,zval)
    print('%10.*f  %10.*f ' %(dp,v,dp,vpval))




def getcommandline():
    parser= argparse.ArgumentParser(description='Bootstrap with clound transform, Dec 2017')
    parser.add_argument('datafilename', help='file to compute linear fit from')
    parser.add_argument('--csv',default= False,action='store_true',help='data is csv file x y z t wellname columns')
    parser.add_argument('--optype',choices=['analysis','applyp50','gstat','gstatpp'],default='analysis',
                        help = 'type of operation: ')
    parser.add_argument('--usebayes',default=False,action='store_true',help='Use slope intercept from bayesian. dfv= use bootstrap')
    parser.add_argument('--nheader',default=0,type=int,help='header lines to skip. dfv=0')
    parser.add_argument('--datacols',type=int,nargs=2,default=[0,1],help='column numberof actual & Predicted variable.dfv = 0 1')
    parser.add_argument('--datacolsgs',type=int, nargs=5,default=[0,1,2,3,4],\
                        help='x y actual predicted wellname column numbers. Used to output eas file.dfv 0 1 2 3 4')

    parser.add_argument('--gstatvariogram',type=float,nargs=2,default = [25000,8000],\
                        help='Variogram  sill range,dfv= 25000 8000' )

    parser.add_argument('--gridnodes',type=int,nargs=2,default=[25,25],\
                        help='Grid nodes dx dy for Surfer grid computation. dfv= 25 25')

    parser.add_argument('--gridfilename', help='grid file to transform. 3 column ASCII space delimited')
    parser.add_argument('--gridheader',default=0,type=int,help='header lines to skip. dfv=0')
    parser.add_argument('--gridcols',type=int,nargs=3,default=[0,1,2],help='column numbers. dfv = 0 1 2')
    parser.add_argument('--interpolate',default='idw',choices=['idw','linear','cubic','nearest','rbf','avgmap','triang','ct'],\
                        help='Interpolation type. dfv= idw. Other types uses griddata only for no polygon input')
    parser.add_argument('--radius',default=5000.00, type=float,help='search radius for zmap interpolation. use w/ -n avgmap.dfv=5000m')


    # parser.add_argument('--pseudopoints',action='store_true',default=False,help='Generate pseudo points for KED')
    parser.add_argument('--ppointsinc',type=int,default=10,\
                        help='pseudo points regular increment.dfv= 20 ')
    parser.add_argument('--ppointsdistance',type=float,default=1000.0,\
                        help='Pseudo points filter distance from control points,dfv=1000')
    parser.add_argument('--multipliershift',type=float,nargs=2,default=[1.0,0.0] ,\
        help='Multiplier shift for z  , dfv= 1.0 0,0')


    parser.add_argument('--flipsign',action='store_true',default=False,help='Flip sign of both columns. dfv= leave as is')
    parser.add_argument('--minmaxscaler', action='store_true',default=False,help='Scale Actual & Predicted to same range')
    # parser.add_argument('--gridminmaxscaler',default=False,action='store_true',
                        # help='Scale output grids the P10, P50 and P90 to input min max, default= no scaling')
    """
    scale input grid (vin) to actual data set -> minmaxscaler only
    can also scale input grid and output 3 grids of P10 P50 and P90 -> minmaxscaler & gridminmaxscaler
    can only scale output grids but leave input unscaled -> gridminmaxscaler only
    can apply not scaling at all -> none
    """
    parser.add_argument('--onepoint',type=float,default=0,help='One Point to compute not a grid file')

    parser.add_argument('--decimalplaces',type=int,default=2,help='decimal places. dfv=2')
    parser.add_argument('--bs_iter',type=int,default=1000,help='bootstrap iteration number.dfv=1000')
    parser.add_argument('--bayes',nargs=3,type=int,default=[50,1000,2000],
                        help='nwalkers nburn nsteps, default= 50 1000 2000')
    parser.add_argument('--bayesdownsample',type=int,default=10,help='downsample bayesian trace by factor. default= 10')
    parser.add_argument('--exceedence',action='store_true',default = False,help='1-CDF probabilities, dfv= CDF probabilities')
    parser.add_argument('--pvals',type=float,nargs=3,default=[10,50,90],\
    help='3 values of P to compute corresponding Z, dfv= 10 50 90')
    parser.add_argument('--zvalopt',action='store_true',default=False,help='Compute probability at zvalue. Use w/ --zval. dfv=false')
    parser.add_argument('--zval',type=float,\
    help='Probability of z to be less than or equal to this value. Use --exceedence for greater than this z')
    parser.add_argument('--hideplots',action='store_true',default=False,help='Hide all plots. dfv =show all plots')
    parser.add_argument('--listall',action='store_true',default=False,help='List all grids for QC. dfv=0')

    result=parser.parse_args()
    if not result.datafilename:
        parser.print_help()
        exit()
    else:
        return result



def main():
    cmdl=getcommandline()
    sns.set()
    colnames =['VIN','VPRED']
    colnamesgs =['WNAME','X','Y','VIN','VPRED']
    # if cmdl.optype == ('gstat' or 'gstatpp' ):
    if cmdl.optype == 'gstat' or cmdl.optype =='gstatpp':
        if cmdl.csv:
            vio1 = pd.read_csv(cmdl.datafilename,header=None,skiprows=1)

        else:
            vio1 = pd.read_csv(cmdl.datafilename,header=None,\
            skiprows=cmdl.nheader,delim_whitespace= True,comment='#')

        vincol = vio1.columns[cmdl.datacolsgs[2]]
        vpredcol = vio1.columns[cmdl.datacolsgs[3]]
        xincol = vio1.columns[cmdl.datacolsgs[0]]
        yincol = vio1.columns[cmdl.datacolsgs[1]]
        wincol = vio1.columns[cmdl.datacolsgs[4]]
        # print(vincol,vpredcol,xincol,yincol,wincol)
        vio = vio1[[wincol,xincol,yincol,vincol,vpredcol]].copy()
        vio.columns=colnamesgs
    else:
        if cmdl.csv:
            # vio = pd.read_csv(cmdl.datafilename,usecols=cmdl.datacols,header=None,skiprows=1)
            vio1 = pd.read_csv(cmdl.datafilename,header=None,skiprows=1)

        else:
            vio1 = pd.read_csv(cmdl.datafilename,header=None,\
            skiprows=cmdl.nheader,delim_whitespace= True,comment='#')
        vincol = vio1.columns[cmdl.datacols[0]]
        vpredcol = vio1.columns[cmdl.datacols[1]]


        vio = vio1[[vincol,vpredcol]].copy()
        vio.columns=colnames

    #print(vio.describe())
    print(vio.head())
    vio['VPRED'] = vio['VPRED'].apply(lambda x: x * cmdl.multipliershift[0])
    vio['VPRED'] = vio['VPRED'].apply(lambda x: x + cmdl.multipliershift[1])

    vio['VIN'] = vio['VIN'].apply(lambda x: x * cmdl.multipliershift[0])
    vio['VIN'] = vio['VIN'].apply(lambda x: x + cmdl.multipliershift[1])
    if cmdl.flipsign:
        vio['VPRED'] = vio['VPRED'].apply(lambda x: np.negative(x))
        vio['VIN'] = vio['VIN'].apply(lambda x: np.negative(x))
    #errprint(cmdl.filter[0],cmdl.filter[1])
    # vio = vio[(vio['VIN'] > cmdl.filter[0]) & (vio['VIN'] <= cmdl.filter[1])]
    # #always reset index after filtering so bootstrap will work********
    # vio.index = range(len(vio))

    if cmdl.minmaxscaler:
        zactual = vio['VIN'].values
        zpred = vio['VPRED'].values
        zalimits = (zactual.min(),zactual.max())
        zactual1 = zactual.reshape(-1,1)
        zpred1 = zpred.reshape(-1,1)
        vio['VIN'] = MinMaxScaler(feature_range=zalimits).fit_transform(zactual1)
        vio['VPRED'] = MinMaxScaler(feature_range = zalimits).fit_transform(zpred1)
        #previous version did not have zalimits inside MinMaxScaler, i.e. scaled to 0 to 1
        print('After minmaxscaler')
        print(vio.head())


    timestr = time.strftime("%Y%m%d-%H%M%S")
    dirsplit,fextsplit= os.path.split(cmdl.datafilename)
    fname,fextn= os.path.splitext(fextsplit)
    if cmdl.optype == 'gstat'  or cmdl.optype == 'gstatpp':
        pdffname = os.path.join(dirsplit,fname)+'_cols%-d_%-d'%(cmdl.datacolsgs[2],cmdl.datacolsgs[3])+".pdf"
        easfname = os.path.join(dirsplit,fname)+'_bi.eas'
    else:
        pdffname = os.path.join(dirsplit,fname)+'_cols%-d_%-d'%(cmdl.datacols[0],cmdl.datacols[1])+".pdf"

    if cmdl.optype == 'analysis':
        bspoint,lincf,bstrapdf,bayescf,bayesdf=apply_bootstrap(vio['VPRED'],vio['VIN'],n=cmdl.bs_iter,
                    pvals=cmdl.pvals,dp=cmdl.decimalplaces,pdffname = pdffname,hide_plots=cmdl.hideplots,
                    nwalkers=cmdl.bayes[0],nburn=cmdl.bayes[1],nsteps=cmdl.bayes[2])
        #print(bspoint[8],bspoint[9],bspoint[6],bspoint[7])
        statoutfname = os.path.join(dirsplit,fname) +"_stats.csv"
        bootstrapfname = os.path.join(dirsplit,fname) +"_bootstrap.csv"
        bayesfname = os.path.join(dirsplit,fname) +"_bayes.csv"

    if cmdl.optype == 'gstat' :
    # if cmdl.gstat:
        togstat_eas(vio['X'],vio['Y'],vio['VIN'],vio['VPRED'],vio['WNAME'],easfname,cmdl.decimalplaces)
        #read grid file that has been corrected by bootstraplin.

        xc,yc,vc=gridlistin(cmdl.gridfilename,cmdl.gridcols,cmdl.gridheader)
        xyc = np.transpose(np.vstack((xc,yc)))
        xyhull = qhull(xyc)

        #saving qhull data
        hullfname = os.path.join(dirsplit,fname) +"_xyhull.dat"
        f5 = open(hullfname,'w')
        listpseudop(xyhull[:,0],xyhull[:,1],f5)
        f5.close()
    elif cmdl.optype == 'gstatpp':
        # if cmdl.pseudopoints:
        xc,yc,vc=gridlistin(cmdl.gridfilename,cmdl.gridcols,cmdl.gridheader)
        """
        if cmdl.gridminmaxscaler:
            zlimits = (vio['VIN'].min(),vio['VIN'].max())
            vc = MinMaxScaler(feature_range=zlimits).fit_transform(vc)
        """
        xyc = np.transpose(np.vstack((xc,yc)))
        xyhull = qhull(xyc)
        xfpp,yfpp = distcomp3(xyhull[:,0],xyhull[:,1],vio['X'],vio['Y'],
                ninc1=cmdl.ppointsinc,filtdist= cmdl.ppointsdistance)
        #print(xfpp.size,yfpp.size)
        xppf,yppf =filterhullpolygon(xfpp,yfpp,xyhull)
        #print('xppf ',len(xppf),'yppf ',len(yppf))

        xyppf = np.transpose(np.vstack((xppf,yppf)))
        zppf = map2ddata(xyc,vc,xyppf,cmdl.radius,cmdl.interpolate)
        ppnames = ['PP-%-3d'% n for n in range(xppf.size)]
        allx = np.append(vio['X'].values,xppf,0)
        ally = np.append(vio['Y'].values,yppf,0)
        allin = np.append(vio['VIN'].values,zppf,0)
        allpred = np.append(vio['VPRED'].values,zppf,0)
        allw = vio['WNAME'].tolist()
        allw.extend(ppnames)

        togstat_eas(allx,ally,allin,allpred,allw,easfname,cmdl.decimalplaces)

        #saving pseudopoints computed and filtered
        pseudoprnfname = os.path.join(dirsplit,fname) +"_pseudopoints.dat"
        f6 = open(pseudoprnfname,'w')
        #listpseudop(xfpp,yfpp,f6)
        listpseudop(xppf,yppf,f6)
        f6.close()
        dirsplitgrd,fextsplitgrd= os.path.split(cmdl.gridfilename)
        fnamegrd,fextngrd= os.path.splitext(fextsplitgrd)
        #maskfname = os.path.join(dirsplitgrd,fnamegrd)+".grd"

    if cmdl.optype == 'gstat' or cmdl.optype =='gstatpp':
        dirsplitgrd,fextsplitgrd= os.path.split(cmdl.gridfilename)
        fnamegrd,fextngrd= os.path.splitext(fextsplitgrd)
        maskfname = os.path.join(dirsplitgrd,fnamegrd)
        limits=[]
        limits.append(xc.min())
        limits.append(xc.max())
        limits.append(cmdl.gridnodes[0])
        limits.append(yc.min())
        limits.append(yc.max())
        limits.append(cmdl.gridnodes[1])
        nc,nr,zsurfer=create_surfergrid(limits,xyc,vc)
        #list_surfergrid(limits,nc,nr,zsurfer)
        to_surfergrid(limits,nc,nr,zsurfer,maskfname)
        # sill and range
        ked_gstatcmd(cmdl.gstatvariogram[0],cmdl.gstatvariogram[1],easfname,maskfname)

    if cmdl.optype == 'analysis':
        bstrapdf.to_csv(bootstrapfname, index=False)
        bayesdf.to_csv(bayesfname, index=False)
        bstrapstats = bstrapdf.describe()
        bstrapstatsidx = list( bstrapstats.index)
        bstrapstats = bstrapstats.append(bstrapdf.skew(),ignore_index=True)
        bstrapstats = bstrapstats.append(bstrapdf.kurtosis(),ignore_index=True)
        bstrapstatsidx.extend(['skewness','kurtosis'])
        bstrapstats.index = bstrapstatsidx

        bstrapstats.to_csv(statoutfname)
        datastat = rsqr(vio['VIN'],vio['VPRED'])
        statscsvfname = os.path.join(dirsplit,fname) +"_modelstats.csv"

        if not os.path.isfile(statscsvfname):
            fo = open(statscsvfname,'a+')
            print('Data File Name,Actual Variable Column,Predicted Variable Column',\
            'Mean Sum of Squared Errors,Regression Sum of Squared Errors,Root Regression Mean Squared Error',\
            'Coefficient of Determination - Rsquared,Pearson Correlation Coefficient',\
            'Spearman Rank Order Correlation Coefficient',file=fo)
            print('%s,'% cmdl.datafilename,end=' ',file=fo)
            if cmdl.optype == 'gstat' or cmdl.optype =='gstatpp' :
                print('%-d, %-d,' %(cmdl.datacolsgs[2],cmdl.datacolsgs[3]),end=' ',file=fo)
            else:
                print('%-d,  %-d,' %(cmdl.datacols[0],cmdl.datacols[1]),end=' ',file=fo)
            print(' %10.4f,' % datastat[0],end=' ',file=fo)
            print(' %10.4f,'% datastat[1],end=' ',file=fo)
            print(' %10.4f,' % np.sqrt(datastat[1]),end='',file=fo)
            print(' %10.4f,'% datastat[2],end=' ',file=fo)
            print(' %10.4f,'% sts.pearsonr(vio['VIN'],vio['VPRED'])[0],end=' ',file=fo)
            print(' %10.4f'%sts.spearmanr(vio['VIN'],vio['VPRED'])[0],file=fo )
            #print('',file=fo)
            fo.close()
            print('Successfully created %s '  % statscsvfname)
        else:
            fo = open(statscsvfname,'a+')
            print('%s,'% cmdl.datafilename,end=' ',file=fo)
            if cmdl.optype == 'gstat' or  cmdl.optype == 'gstatpp' :
                print('%-d, %-d,' %(cmdl.datacolsgs[2],cmdl.datacolsgs[3]),end=' ',file=fo)
            else:
                print('%-d,  %-d,' %(cmdl.datacols[0],cmdl.datacols[1]),end=' ',file=fo)
            print(' %10.4f,' % datastat[0],end=' ',file=fo)
            print(' %10.4f,'% datastat[1],end=' ',file=fo)
            print(' %10.4f,' % np.sqrt(datastat[1]),end='',file=fo)
            print(' %10.4f,'% datastat[2],end=' ',file=fo)
            print(' %10.4f,'% sts.pearsonr(vio['VIN'],vio['VPRED'])[0],end=' ',file=fo)
            print(' %10.4f'%sts.spearmanr(vio['VIN'],vio['VPRED'])[0],file=fo )
            #print('',file=fo)
            fo.close()
            print('Successfully created %s '  % statscsvfname)
        errprint('Slope    Intercept')
        errprint('%10.*f  %10.*f' %(cmdl.decimalplaces,lincf[0],cmdl.decimalplaces,lincf[1]))
        errprint('Bayes Slope    Bayes Intercept    Bayes Sigma')
        errprint('%10.*f    %10.*f    %10.*f' %(cmdl.decimalplaces,bayescf[0],cmdl.decimalplaces,bayescf[1],
                cmdl.decimalplaces,bayescf[2]))

    if cmdl.optype == 'applyp50':
    #if cmdl.gridfilename and not cmdl.gstat:
        xc,yc,vc=gridlistin(cmdl.gridfilename,cmdl.gridcols,cmdl.gridheader)
        xyhull = qhull(np.transpose(np.vstack((xc,yc))))
        hullfname = os.path.join(dirsplit,fname) +"_xyhull.dat"

        if cmdl.zvalopt:
            probouttxtfname = os.path.join(dirsplit,fname) +"_patzp50.txt"
            f3 = open(probouttxtfname,'w')
            # patzval = process_zval_grid(xc,yc,vc,cmdl.zval,vio['VPRED'],vio['VIN'],f3,
            #     cmdl.decimalplaces,cmdl.bs_iter,exceedence= cmdl.exceedence)
            process_zval_grid(xc,yc,vc,cmdl.zval,vio['VPRED'],vio['VIN'],f3,
                cmdl.decimalplaces,cmdl.bs_iter,exceedence= cmdl.exceedence)
            f3.close()
            print('Successfully created {} '.format(probouttxtfname) )
            # redundancy: data fram is generated. Data is already saved as txt
            # patzdf = pd.DataFrame({'X':xc,'Y':yc,'GRDZ':vc,'PatZ':patzval})
            # patzcols = ['X','Y','GRDZ','PatZ']
            # patzdf = patzdf[patzcols].copy()
            # patzdf.to_csv(probouttxtfname,index=False,sep=' ')
        else:
            if cmdl.usebayes:
                bsgridfname = os.path.join(dirsplit,fname) +"_bygrid.dat"
            else:
                bsgridfname = os.path.join(dirsplit,fname) +"_bsgrid.dat"
            f2 = open(bsgridfname,'w')

            process_pval_grid(xc,yc,vc,cmdl.pvals,vio['VPRED'],vio['VIN'],
                f2,cmdl.bs_iter,dp=cmdl.decimalplaces,usebayes=cmdl.usebayes,
                bayesdownsample = cmdl.bayesdownsample,hideplots = cmdl.hideplots)

            f2.close()


    elif cmdl.onepoint:
        if cmdl.zvalopt:
            process_zval_grid(cmdl.onepoint,cmdl.zval,vio['VPRED'],vio['VIN'],cmdl.bs_iter)
        else:
            onepdffname = os.path.join(dirsplit,fname)+timestr+"_one.pdf"
            process_onepoint_pval(cmdl.onepoint,cmdl.pvals,vio['VPRED'],vio['VIN'],cmdl.bs_iter,pdffname=onepdffname)

    # if cmdl.zvalopt:
    #     proboutfname = os.path.join(dirsplit,fname) +"_patzval.csv"
    #     proboutprnfname = os.path.join(dirsplit,fname) +"_patzval.prn"
    #     vio['PatZ'] = ecompzval(vio['VIN'].values,cmdl.zval)
    #     vio.to_csv(proboutfname,index=False)
    #     vio.to_csv(proboutprnfname,index=False,sep=' ')
    #     print('Successfully created {} and  {} '  .format(proboutfname,proboutprnfname) )
    # else:
    #     proboutfname = os.path.join(dirsplit,fname) +"_zatpvals.csv"
    #     proboutprnfname = os.path.join(dirsplit,fname) +"_zatpvals.prn"



if __name__=='__main__':
	main()
