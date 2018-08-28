# -*- coding: utf-8 -*-
"""
Created on Sat May 27 23:33:12 2017

@author: Lenovo
"""
import sys, os.path
import datetime
import argparse
import numpy as np


def errprint(*args,**kwargs):
    print(*args,file=sys.stderr,**kwargs)



def pip(x,y,poly):  #point_in_poly

   # check if point is a vertex
   if (x,y) in poly: return True

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

def filterhullpolygon(xy,polygon):
    xyf=[]
    for i in range(xy[:,0].size):
        if pip(xy[i,0],xy[i,1],polygon):
            xyf.append((xy[i,0],xy[i,1]))
    return np.array(xyf)








def read_vals(fname):
    zval=[]
    xy =[]
    
    linecount =0
    fgrid=open(fname,'r')
    headercode= fgrid.readline()
    linecount+=1
    header1=fgrid.readline()
    linecount+=1
    xnodes,ynodes=list(map(int,header1.split()))
    #ynodes,xnodes=map(int,header1.split())
    header2=fgrid.readline()
    linecount+=1
    xmin,xmax=list(map(float,header2.split()))
    header3=fgrid.readline()
    linecount+=1
    ymin,ymax=list(map(float,header3.split()))
    header4=fgrid.readline()
    linecount+=1
    zmin,zmax=list(map(float,header4.split()))
    xinc=(xmax-xmin)/(xnodes-1)
    yinc=(ymax -ymin)/(ynodes-1)
    errprint (xnodes, ynodes, xmin, xmax, ymin,ymax, zmin, zmax)
    errprint( xinc, yinc)
    errprint('total number of nodes:',xnodes * ynodes)

    for line in fgrid:
        #print(line)
        [zval.append(float(z)) for z in line.split()]
    fgrid.close()


    yc = ymin
    for j in range(ynodes):
        xc = xmin
        k= 0
        for k in range(xnodes):
                xy.append((xc,yc))
                xc += xinc
                
        yc+=yinc

    

    return np.array(xy),np.array(zval)

def listxyz(fname,xy,z,pgnxy,dp=2):
    dirsplit,fextsplit= os.path.split(fname)
    surfergname,fextn= os.path.splitext(fextsplit)
    gridlistfname = os.path.join(dirsplit,surfergname)+'_gridlist.dat'
    glist = open(gridlistfname,'w')
    for i in range(z.size):
        if pip(xy[i,0],xy[i,1],pgnxy):
            print('%12.2f  %12.2f  %10.*f' %(xy[i,0],xy[i,1],dp,z[i]),file=glist)
    glist.close()

def commandlineparse():
    parser = argparse.ArgumentParser(description='Read surfer grid and polygon output xyz listing. May 27 2017 ')
    parser.add_argument('--gridfilename',help='Surfer grd file')
    parser.add_argument('--polygonfilename',help='Polygon file to filter grid file')
    parser.add_argument('--decimalplaces',type=int,default=2,help='decimal places. dfv=2')
    parser.add_argument('--nullvalue',default='1.70141e+038',help='Default Null value is for Surfer grid')

    results=parser.parse_args()
    if not (results.gridfilename) or not( results.polygonfilename):
    #if not (results.gridfilename) :
        print("\n\nNeed BOTH a grid file  and a polygon file \n\n")
        parser.print_help()
        exit()
    else:
        return results


def main():
    cmdl=commandlineparse()
    xypgn=np.genfromtxt(cmdl.polygonfilename,usecols=[0,1])
    xy,zall = read_vals(cmdl.gridfilename)
    errprint('len of zall ',zall.size,'len of xy',xy[:,0].size)
    listxyz(cmdl.gridfilename,xy,zall,xypgn,cmdl.decimalplaces)
    #xyifhull=filterhullpolygon(xyi,xyhull)
    #zif = zi[np.where(xyi[:,0])]


if __name__ == '__main__':
    main()

