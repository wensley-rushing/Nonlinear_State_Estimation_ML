# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:35:00 2022

@author: lucag
"""

import openseespy.opensees as ops

from define_SteelSections import define_SteelSections
from define_RCSections import define_RCSections




def createModel(H1, M):


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
    nodes = [10, 20]
    elements = [1020]
    
    
    ops.wipe()                                   # deletes everything that was defined before
    
    ops.model('basic', '-ndm', 2, '-ndf', 3)  	# starts the model. Model 2 dimensions, 3 DOFs per node
    
    
    
    
    # definition of nodes
    # ----------------
    # node(nodeTag, *crds)  
    
    #e.g. 
    #ops.node(10, 0.0(x), 0.0(y))   
    
    ops.node(10,    0,  0)
    ops.node(20,    0,  H1)
    
    
    
    
    
    
    # Single Point constraints  (support conditions)
    # -------------------------------
    
    #fix 	nodeTag	    Dx	    Dy	     Rz
    
    ops.fix(10,      	1,	    1,	     1)       #fixed supports
    # ops.fix(11,      	1,	    1,	     1)       #fixed supports
    
    
    
    # Define nodal masses
    # -------------------------------
    
    #Mass 	Nodetag 	mx 		my 		mIz
    ops.mass(20,      	M,	    M,	     0)       #fixed supports
    # ops.mass(21,      	M,	    M,	     0)       #fixed supports
    
    
    
    # Define sections
    # ----------------------------------
    #HEB200tag, IPE200tag = define_SteelSections() #defines nonlinear steel sections IPE200 and HEB200
    HEB200tag, IPE200tag = define_RCSections()
    
    
    
    
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
    ops.beamIntegration('Lobatto', IPE200tag, IPE200tag, 5)
    
    
    
    # Define element(s)
    
    #   element('forceBeamColumn', eleTag, *eleNodes, transfTag,   integrationTag)
    #columns
    ops.element('forceBeamColumn', 1020,   *[10,20],  LinearGeomT, IPE200tag)
    # ops.element('forceBeamColumn', 1121,   *[11,21],  PDeltaGeomT, HEB200tag)
    # #beam
    # ops.element('forceBeamColumn', 2021,   *[20,21],  LinearGeomT, IPE200tag)
    
    return nodes, elements
    
    print("--------------- Model SUCCESSFULLY defined ---------------")






















