"""
Dec 2017: backinterpolate:
    with 2 files: either flat and/or csv
    one with x y z columns -> datafile or grid
    the other with x y to backinterpolate its z value from datafile -> horfile

    python backinterpolate.py i4_sus_ss.csv i4_sus_ss_i4well_ntg_swa.csv --datacsv --xyfcsv --xyfcols 1 2 --xyfdiff --xyfzcol 3

python backinterpolate.py UL30.dat UL_30_C1.csv --dataheader 20  --xyfcols 2 3  --xyfzcol 4 --xyfdiff --xyfcsv --flipsign --decimalplaces 3

python backinterpolate1.py UL30.dat  UL_30_C1.csv --dataheader 20 --datacols 0 1 2 --xyfcsv  --xyfcols 2 3  --flipsign

python backinterpolate1.py UL30.dat  UL_30_C1.csv --dataheader 20 --datacols 0 1 2 --xyfcsv  --xyfcols 2 3 4 0 --flipsign
"""

import sys, os.path
import argparse
import numpy as np
import math as m
import pandas as pd
import scipy.stats as sts
#import scipy.interpolate
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from sklearn.preprocessing import MinMaxScaler

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
    # cindx=0
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
    # elif maptype =='avgmap':
    #     vri=dataavgmap(xy,vr,xyi,cmdl.radius)
    elif maptype =='triang':
        linearnd=LinearNDInterpolator(xy,vr,stats[2])
        vri= linearnd(xyi)
    elif maptype == 'ct':
        ct=CloughTocher2DInterpolator(xy,vr,stats[2])
        vri=ct(xyi)
    return vri





def errprint(*args,**kwargs):
    print(*args,file=sys.stderr,**kwargs)


#list x y kriged values, and kriged variance
def listxyzv(x,y,z,v,dp=2):
    for i in range(len(z)):
        print('%12.2f  %12.2f  %10.*f  %10.*f' %(x[i],y[i],dp,z[i],dp,v[i]))

def listpseudop(x,y,fn,dp=2):
    for i in range(len(x)):
        print('%12.*f  %12.*f  100 ' %(dp,x[i],dp,y[i]),file=fn)



def datain(fname,xyvcols=[0,1,2],nheader=0): #used for single coef per file
    xyv=np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    #filter surfer null values by taking all less than 10000, arbitrary!!
    xyv = xyv[xyv[:,2]<10000.0]
    #xya = xya[~xya[:,2]==  missing]
    return xyv[:,0],xyv[:,1],xyv[:,2]


def horin(fname,xyvcols=[0,1],nheader=0):
    xy=np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    #return xyv[:,0],xyv[:,1]
    return xy






def getcommandline():
    parser= argparse.ArgumentParser(description='Back Interpolate data. Dec 2017')
    parser.add_argument('datafilename', help='Flat file or csv to interpolate from - grid xyz ')
    parser.add_argument('--datacsv',default= False,action='store_true',help='data is csv file x y z wellname columns')
    parser.add_argument('--dataheader',default=0,type=int,help='header lines to skip. dfv=0')
    parser.add_argument('--datacols',type=int, nargs=3,default=[0,1,2],\
                        help='x y z  to interpolate from.dfv 0 1 2 ')

    parser.add_argument('xyfilename', help='xy file to find z for. well file x y z. 2 column ASCII space delimited')
    parser.add_argument('--xyfheader',default=0,type=int,help='horizon file header lines to skip. dfv=0')
    parser.add_argument('--xyfcols',type=int,nargs='+',default=[0,1],
        help='horizon file either enter x y, 2 cols only, or  x y z well 4 cols, column numbers. dfv = 0 1 ')
    parser.add_argument('--xyfcsv',default= False,action='store_true',help='horizon file  is csv file x y  wellname columns')
    # parser.add_argument('--xyfdiff',default=False,action='store_true',help='Compute difference between actual vs predicted')
    # parser.add_argument('--xyfzidcol',type=int,nargs=2,default=[2,2],help='Column of z and well for diff computation')

    parser.add_argument('--fliphorzsign',action='store_true',default=False,help='Flip sign of z data col. dfv= leave as is')


    parser.add_argument('--multipliershift',type=float,nargs=2,default=[1.0,0.0] ,\
        help='Multiplier shift for z  , dfv= 1.0 0,0')


    parser.add_argument('--flipsign',action='store_true',default=False,help='Flip sign of z data col. dfv= leave as is')
    parser.add_argument('--minmaxscaler', action='store_true',default=False,help='Scale Actual & Predicted to same range')
    parser.add_argument('--decimalplaces',type=int,default=2,help='decimal places. dfv=2')
    parser.add_argument('--filter',nargs=2, type=float,default=[-10000.0,10000.0],help='min max of actual variable only to filter. dfv= -10000 10000')
    parser.add_argument('--listdata',action='store_true',default=False,help='List data file  dfv=false')
    parser.add_argument('--listhor',action='store_true',default=False,help='List horizon file  dfv=false')

    result=parser.parse_args()
    if not (result.datafilename and result.xyfilename):
        parser.print_help()
        exit()
    else:
        return result



def main():
    cmdl=getcommandline()
    colnames =['X','Y','Z']

    if cmdl.datacsv:
        vio = pd.read_csv(cmdl.datafilename,usecols=cmdl.datacols,header=None,skiprows=1)

    else:
        vio = pd.read_csv(cmdl.datafilename,usecols=cmdl.datacols,header=None,\
        skiprows=cmdl.dataheader,delim_whitespace= True,comment='#')
    #vio = vio0.iloc[:,cmdl.datacols].copy()
    vio.columns =colnames
    #print(vio.describe())
    vio['Z'] = vio['Z'].apply(lambda x: x * cmdl.multipliershift[0])
    vio['Z'] = vio['Z'].apply(lambda x: x + cmdl.multipliershift[1])

    if cmdl.flipsign:
        vio['Z'] = vio['Z'].apply(lambda x: np.negative(x))
    #errprint(cmdl.filter[0],cmdl.filter[1])
    vio = vio[(vio['Z'] > cmdl.filter[0]) & (vio['Z'] <= cmdl.filter[1])]
    #always reset index after filtering so bootstrap will work********
    vio.index = range(len(vio))

    if cmdl.minmaxscaler:
        zactual = vio['Z'].values
        # zalimits = (zactual.min(),zactual.max())
        zactual1 = zactual.reshape(-1,1)
        vio['Z'] = MinMaxScaler().fit_transform(zactual1)



    dirsplit,fextsplit= os.path.split(cmdl.datafilename)
    fname1,fextn= os.path.splitext(fextsplit)

    dirsplit2,fextsplit2= os.path.split(cmdl.xyfilename)
    fname2,fextn2= os.path.splitext(fextsplit2)
    fname = fname1 +'_'+ fname2

    #xc,yc,vc=datain(cmdl.datafilename,cmdl.datacols,cmdl.dataheader)
    xc = vio['X'].values
    yc = vio['Y'].values
    zc = vio['Z'].values

    xyc = np.transpose(np.vstack((xc,yc)))
    xyhull = qhull(xyc)

    if cmdl.xyfcsv:
        hor = pd.read_csv(cmdl.xyfilename,usecols=cmdl.xyfcols,header=None,skiprows=1)

    else:
        hor = pd.read_csv(cmdl.xyfilename,usecols=cmdl.xyfcols,header=None,\
        skiprows=cmdl.xyfheader,delim_whitespace= True,comment='#')

    # print('cmdl xyfcols:',len(cmdl.xyfcols),cmdl.xyfcols)
    # print(hor.head())
    if len(cmdl.xyfcols) > 2 :
        xyfcolnames = ['X','Y','Z','WELL']
        wx = hor[cmdl.xyfcols[0]]
        wy = hor[cmdl.xyfcols[1]]
        wz = hor[cmdl.xyfcols[2]]
        wn = hor[cmdl.xyfcols[3]]
        hor1 = pd.DataFrame({'X':wx,'Y':wy,'Z':wz,'WELL':wn})
        hor1 = hor1[xyfcolnames].copy()
    else:
        xyfcolnames = ['X','Y']
        wx = hor.iloc[:,0]
        wy = hor.iloc[:,1]
        hor1 = pd.DataFrame({'X':wx,'Y':wy})
        hor1 = hor1[xyfcolnames].copy()
    print(hor1.head())
    xh = hor1['X'].values
    yh = hor1['Y'].values
    xhf,yhf = filterhullpolygon(xh,yh,xyhull)
    xyhf = np.transpose(np.vstack((xhf,yhf)))
    zi = idw(xyc,zc,xyhf)
    print('len of zh %d, zc %d' % (len(zi),len(zc)))
    #print(zh)
    if len(cmdl.xyfcols) > 2 :
        # wname = hor['WELL']
        zh = hor1['Z'].values
        if cmdl.fliphorzsign:
            zh *= (-1.0)
        zhf = zh[xhf == xh]
        # print('len of zh %d, zhf %d' % (len(zh),len(zhf)))
        dz = zhf - zi
        horf = pd.DataFrame(data = {'X':xhf,'Y':yhf,'Actual':zhf,'BI':zi,'DIFF':dz,'WELL':hor1['WELL']})
        horfordered = horf[['X','Y','Actual','BI','DIFF','WELL']].copy()
    else:
        horf = pd.DataFrame(data = {'X':xhf,'Y':yhf,'BI':zi})
        horfordered = horf[['X','Y','BI']].copy()
    horbi = os.path.join(dirsplit,fname) + "_bi.csv"
    horbitxt = os.path.join(dirsplit,fname) + "_bi.txt"
    horfordered.to_csv(horbi,index=False)
    horfordered.to_csv(horbitxt,index=False,sep=' ')
    print('Successfully generated {}'.format(horbi))
    print('Successfully generated {}'.format(horbitxt))


if __name__=='__main__':
	main()
