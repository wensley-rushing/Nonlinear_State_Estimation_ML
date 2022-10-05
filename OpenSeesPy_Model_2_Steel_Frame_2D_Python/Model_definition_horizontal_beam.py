# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:35:00 2022

@author: lucag
"""

import openseespy.opensees as ops

from define_SteelSections import define_SteelSections
from define_RCSections import define_RCSections




def createModel(L1, M):


#sketch of the structure  with node and element numbering

#
# 20		2021     	  21	

# | |					 | |	
# | |					 | |	
# 1020 				     1121
# | |					 | |	
# | |					 | |	

# 10				      11	
#
    nodes = [10, 11, 12]
    elements = [1012]
    
    
    ops.wipe()                                   # deletes everything that was defined before
    
    ops.model('basic', '-ndm', 2, '-ndf', 3)  	# starts the model. Model 2 dimensions, 3 DOFs per node
    
    
    
    
    # definition of nodes
    # ----------------
    # node(nodeTag, *crds)  
    
    #e.g. 
    #ops.node(10, 0.0(x), 0.0(y))   
    
    ops.node(10,    0,  0)
    ops.node(11,    L1/2, 0)
    ops.node(12,    L1, 0)
    
    
    
    
    
    # Single Point constraints  (support conditions)
    # -------------------------------
    
    #fix 	nodeTag	    Dx	    Dy	     Rz
    
    ops.fix(10,      	1,	    1,	     1)       #fixed supports
    ops.fix(12,      	1,	    1,	     1)       #fixed supports
    
    
    
    # Define nodal masses
    # -------------------------------
    
    #Mass 	Nodetag 	mx 		my 		mIz
    #ops.mass(11,      	M,	    M,	     0)       #fixed supports
    ops.mass(10,      	0,	    0,	     0)       #fixed supports
    ops.mass(12,      	0,	    0,	     0)       #fixed supports
    
    
    
    # Define sections
    # ----------------------------------
    #HEB200tag, IPE200tag = define_SteelSections() #defines nonlinear steel sections IPE200 and HEB200
    HEB200tag, IPE200tag = define_RCSections()
    # column , beam
    
    
    
    # Define geometric transformations
    # ----------------------------------
    LinearGeomT = 1
    PDeltaGeomT = 2
    
    # GeoTran 	   type 	    tag
    ops.geomTransf('Linear', LinearGeomT)    #for beams
    ops.geomTransf('PDelta', PDeltaGeomT)    #for columns
    
    
    # Define beamIntegrations (Gauss-Lobatto integration rule)
    
    # beamIntegration('Lobatto',    tag,        secTag,  Np)   Np = number of integration points
    ops.beamIntegration('Lobatto', HEB200tag, HEB200tag, 5)
    ops.beamIntegration('Lobatto', IPE200tag, IPE200tag, 3)
    
    
    
    # Define element(s)
    
    #   element('forceBeamColumn', eleTag, *eleNodes, transfTag,   integrationTag)

    #beam
    ops.element('forceBeamColumn', 1012,   *[10,12],  LinearGeomT, HEB200tag)
    
    return nodes, elements
    
    print("--------------- Model SUCCESSFULLY defined ---------------")






















