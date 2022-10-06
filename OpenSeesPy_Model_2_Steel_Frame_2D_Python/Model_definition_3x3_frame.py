# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:35:00 2022

@author: lucag
"""

import openseespy.opensees as ops

from define_SteelSections import define_SteelSections
from define_RCSections import define_RCSections



def createModel(H1, L1, M):     
        
        #sketch of the structure  with node and element numbering
        
        #
        
        # 40		4041     	  41		4142     	  42		4243    	  43	
        
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | |	
        # 3040 				     3141		         	 3242		        	 3343
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | | 
        
        # 30		3031     	  31		3132     	  32		3233    	  33	
        
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | |	
        # 2030 				     2131		         	 2232		        	 2333
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | |
        
        # 20		2021     	  21		2122     	  22		2223    	  23	
        
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | |	
        # 1020 				     1121		         	 1222		        	 1323
        # | |					 | |					 | |					 | |	
        # | |					 | |					 | |					 | |	
        
        # 10				      11				      12				      13	
        
        nodes = [10, 11, 12, 13,
                 20, 21, 22, 23,
                 30, 31, 32, 33,
                 40, 41, 42, 43]
        
        elements = [1020, 1121, 1222, 1323, 2021, 2122, 2223,                        
                    2030, 2131, 2232, 2333, 3031, 3132, 3233,
                    3040, 3141, 3242, 3343, 4041, 4142, 4243]
    
        col =       [1020, 1121, 1222, 1323,                         
                    2030, 2131, 2232, 2333, 
                    3040, 3141, 3242, 3343]
        
        
        
        ops.wipe()                                   # deletes everything that was defined before
        
        ops.model('basic', '-ndm', 2, '-ndf', 3)  	# starts the model. Model 2 dimensions, 3 DOFs per node
        
        
        
        
        # definition of nodes
        # ----------------
        # node(nodeTag, *crds)  
        
        #e.g. 
        #ops.node(10, 0.0(x), 0.0(y))   
        
        ops.node(10,    0,  0)
        ops.node(11,    L1, 0)
        ops.node(12,    2*L1,  0)
        ops.node(13,    3*L1, 0)
        
        ops.node(20,    0,  H1)
        ops.node(21,    L1, H1)
        ops.node(22,    2*L1, H1)
        ops.node(23,    3*L1, H1)
        
        ops.node(30,    0,  2*H1)
        ops.node(31,    L1, 2*H1)
        ops.node(32,    2*L1, 2*H1)
        ops.node(33,    3*L1, 2*H1)
        
        ops.node(40,    0,  3*H1)
        ops.node(41,    L1, 3*H1)
        ops.node(42,    2*L1, 3*H1)
        ops.node(43,    3*L1, 3*H1)
        
        
     
    
    
    
    
        
        # Single Point constraints  (support conditions)
        # -------------------------------
        
        #fix 	nodeTag	    Dx	    Dy	     Rz
        
        ops.fix(10,      	1,	    1,	     1)       #fixed supports
        ops.fix(11,      	1,	    1,	     1)       #fixed supports
        ops.fix(12,      	1,	    1,	     1)       #fixed supports
        ops.fix(13,      	1,	    1,	     1)       #fixed supports
        
        
        
        # Define nodal masses
        # -------------------------------
        
        #Mass 	Nodetag 	mx 		my 		mIz
        ops.mass(20,      	M[3],	    M[3],	     0)       #fixed supports
        ops.mass(21,      	M[2],	    M[2],	     0)       #fixed supports
        ops.mass(22,      	M[2],	    M[2],	     0)       #fixed supports
        ops.mass(23,      	M[3],	    M[3],	     0)       #fixed supports
        
        ops.mass(30,      	M[3],	    M[3],	     0)       #fixed supports
        ops.mass(31,      	M[2],	    M[2],	     0)       #fixed supports
        ops.mass(32,      	M[2],	    M[2],	     0)       #fixed supports
        ops.mass(33,      	M[3],	    M[3],	     0)       #fixed supports
        
        ops.mass(40,      	M[1],	    M[1],	     0)       #fixed supports
        ops.mass(41,      	M[0],	    M[0],	     0)       #fixed supports
        ops.mass(42,      	M[0],	    M[0],	     0)       #fixed supports
        ops.mass(43,      	M[1],	    M[1],	     0)       #fixed supports
        
    
    
    
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
        #columns # floor 1
        ops.element('forceBeamColumn', 1020,   *[10,20],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 1121,   *[11,21],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 1222,   *[12,22],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 1323,   *[13,23],  PDeltaGeomT, HEB200tag)
        
        #columns # floor 2
        ops.element('forceBeamColumn', 2030,   *[20,30],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 2131,   *[21,31],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 2232,   *[22,32],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 2333,   *[23,33],  PDeltaGeomT, HEB200tag)
        
        #columns # floor 3
        ops.element('forceBeamColumn', 3040,   *[30,40],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 3141,   *[31,41],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 3242,   *[32,42],  PDeltaGeomT, HEB200tag)
        ops.element('forceBeamColumn', 3343,   *[33,43],  PDeltaGeomT, HEB200tag)
        
        #beams # floor 1
        ops.element('forceBeamColumn', 2021,   *[20,21],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 2122,   *[21,22],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 2223,   *[22,23],  LinearGeomT, IPE200tag)
        
        #beams # floor 2
        ops.element('forceBeamColumn', 3031,   *[30,31],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 3132,   *[31,32],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 3233,   *[32,33],  LinearGeomT, IPE200tag)
        
        #beams # floor 3
        ops.element('forceBeamColumn', 4041,   *[40,41],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 4142,   *[41,42],  LinearGeomT, IPE200tag)
        ops.element('forceBeamColumn', 4243,   *[42,43],  LinearGeomT, IPE200tag)
        
        
        return nodes, elements, col
        
        
        print("--------------- Model SUCCESSFULLY defined - 2 storeys ---------------")





















