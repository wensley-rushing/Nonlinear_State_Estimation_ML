# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:52:23 2022

@author: lucag
"""


import openseespy.opensees as ops


Pa = 1; MPa = Pa*1e6; GPa = Pa*1e9
m = 1; mm = 0.001*m



def define_SteelSections():


    
    # ##############################################
    # Definition of materials IDs
    # ##############################################
    
    
    # Basic parameters for S355
    
    E = 210*GPa
    fy = 355*MPa #       yield stress    #S355
    b = 0.02;  # strain-hardening ratio
    R0 = 18 ;cR1 = 0.925;cR2 = 0.15    # transition from elastic to plastic branches
    # Recommended values: $R0=between 10 and 20, $cR1=0.925, $cR2=0.15
    
    
    matSteelTag = 1
    
    ops.uniaxialMaterial('Steel02', matSteelTag, fy, E, b,*[R0,cR1,cR2])
    



    # ##############################################
    # Definition of sections IDs
    # ##############################################
    
    
    HEB200tag = 1; #columns
    IPE200tag = 2; #beams
    
    # Define HEB200 dimensions
    h_col  = 200 * mm;       #    nominal depth
    b_col  = 200 * mm;     #     flange width
    tf_col = 15  * mm;     #     flange thickness
    tw_col = 9   * mm;     #     web thickness
    
    # Define IPE200 dimensions
    h_beam   = 200 * mm;     #    nominal depth
    b_beam   = 100 * mm;     #     web thickness
    tf_beam  = 8.5 * mm;     #     flange width
    tw_beam  = 5.6 * mm;     #     flange thickness
    
    #   Wsection { secID matID d bf tf tw nfdw nftw nfbf nftf}
    	# secID - section ID number
    	# matID - material ID number 
    	# d  = nominal depth
    	# tw = web thickness
    	# bf = flange width
    	# tf = flange thickness
    	# nfdw = number of fibers along web depth      
    	# nftw = number of fibers along web thickness     #not relevant for 2D
    	# nfbf = number of fibers along flange width      #not relevant for 2D
    	# nftf = number of fibers along flange thickness
        
   #Wsection (secID, matID,       d,      bf,     tf,      tw,        nfdw, nftw, nfbf, nftf)
    Wsection(HEB200tag, matSteelTag, h_col,  b_col,  tf_col,  tw_col,    8,    1,    1,    2)
    Wsection(IPE200tag, matSteelTag, h_beam, b_beam, tf_beam, tw_beam,   8,    1,    1,    2)
    
    
    print("----------------------sections defined-------------------------")
    
    return HEB200tag, IPE200tag





def Wsection (secID, matID, d,bf,tf,tw,nfdw,nftw,nfbf,nftf):
    
	# input parameters
	# secID - section ID number
	# matID - material ID number 
	# d  = nominal depth
	# tw = web thickness
	# bf = flange width
	# tf = flange thickness
	# nfdw = number of fibers along web depth 
	# nftw = number of fibers along web thickness
	# nfbf = number of fibers along flange width
	# nftf = number of fibers along flange thickness
    
    dw = d-2*tf
    y = [0., -d/2, -dw/2, dw/2, d/2]
    z = [0., -bf/2, -tw/2, tw/2, bf/2]
    

    
    # Define Sections
    #  section('Fiber', secTag)
    ops.section('Fiber', secID)
    #patch('quad', matTag, numSubdivIJ, numSubdivJK, *crdsI, *crdsJ, *crdsK, *crdsL)
    ops.patch('quad',1,nfbf, nftf,        y[1],z[4],  y[1],z[1],  y[2],z[1],  y[2],z[4] )
    ops.patch('quad',1,nftw, nfdw,        y[2],z[3],  y[2],z[2],  y[3],z[2],  y[3],z[3] )
    ops.patch('quad',1,nfbf, nftf,        y[3],z[4],  y[3],z[1],  y[4],z[1],  y[4],z[4] )

#end SteelSections