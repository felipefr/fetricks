"""

This file is part of fetricks:  useful tricks and some extensions for FEniCs and other FEM-related utilities
Obs: (fe + tricks: where "fe" stands for FEM, FEniCs and me :) ).

Copyright (c) 2022-2023, Felipe Rocha.
See file LICENSE.txt for license information.
Please report all bugs and problems to <felipe.figueredo-rocha@ec-nantes.fr>, or
<f.rocha.felipe@gmail.com>
"""

def writeDict(d):
    f = open(d['files']['net_settings'],'w')
    
    for keys, value in zip(d.keys(),d.values()):
        f.write("{0}: {1}\n".format(keys,value))
        
    f.close()


def getDatasetsXY(nX, nY, XYdatafile, scalerX = None, scalerY = None, Ylabel = 'Y', scalerType = 'MinMax'):
    Xlist = []
    Ylist = []
    
    if(type(XYdatafile) != type([])):
        XYdatafile = [XYdatafile]
    
    for XYdatafile_i in XYdatafile:    
        Xlist.append(myhd.loadhd5(XYdatafile_i, 'X')[:,:nX])
        Ylist.append(myhd.loadhd5(XYdatafile_i, Ylabel)[:,:nY])
    
    X = np.concatenate(tuple(Xlist),axis = 0)
    Y = np.concatenate(tuple(Ylist),axis = 0)
    
    if(type(scalerX) == type(None)):
        scalerX = constructor_scaler(scalerType)            
        scalerX.fit(X)
    
    if(type(scalerY) == type(None)):
        scalerY = constructor_scaler(scalerType)            
        scalerY.fit(Y)
            
    return scalerX.transform(X), scalerY.transform(Y), scalerX, scalerY

