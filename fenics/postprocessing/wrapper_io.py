from __future__ import print_function
from dolfin import *

def readXDMF_with_markers(meshFile, mesh, comm = MPI.comm_world):

    with XDMFFile(comm,meshFile) as infile:
        infile.read(mesh)
    
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(comm, "{0}_faces.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "faces")
                
    mf  = MeshFunction("size_t", mesh, mvc)
  
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(comm, "{0}_regions.xdmf".format(meshFile[:-5])) as infile:
        infile.read(mvc, "regions")
    
    mt  = MeshFunction("size_t", mesh, mvc)
    
    return mt, mf
    
def exportXDMF_gen(filename, fields, k = -1):
    with XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
    
        
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write(field, k) 
        
        if('cell' in fields.keys()):
            for field in fields['cell']:
                ofile.write(field, k) 

        if('cell_vector' in fields.keys()):
            for field in fields['cell_vector']:
                for field_i in field.split():
                    ofile.write(field_i, k) 


def exportXDMF_gen_append(ofile, fields, k = -1):
    # if(k==0):
    ofile.parameters["flush_output"] = True
    ofile.parameters["functions_share_mesh"] = True
        
        
    if('vertex' in fields.keys()):
        for field in fields['vertex']:
            ofile.write_checkpoint(field, field.name(), float(k), XDMFFile.Encoding.HDF5, True) 
    

    if('cell' in fields.keys()):
        for field in fields['cell']:
            for field_i in field.split():
                ofile.write_checkpoint(field_i,field_i.name(), float(k), XDMFFile.Encoding.HDF5, True) 
           
            
               
def exportXDMF_checkpoint_gen(filename, fields):
    with XDMFFile(filename) as ofile: 
        ofile.parameters["flush_output"] = True
        ofile.parameters["functions_share_mesh"] = True
            
        count = 0
        if('vertex' in fields.keys()):
            for field in fields['vertex']:
                ofile.write_checkpoint(field, field.name(), count, append = True)
                # count = count + 1
                
        
        if('cell' in fields.keys()):
            for field in fields['cell']:
                ofile.write_checkpoint(field, field.name(), count, XDMFFile.Encoding.HDF5, append = True) 
                # count = count + 1

        if('cell_vector' in fields.keys()):
            for field in fields['cell_vector']:
                for field_i in field.split():
                    ofile.write_checkpoint(field_i, field_i.name(), count, XDMFFile.Encoding.HDF5, append = True) 
                    # count = count + 1
