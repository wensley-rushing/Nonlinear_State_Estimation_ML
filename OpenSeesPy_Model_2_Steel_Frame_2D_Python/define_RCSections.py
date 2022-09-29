# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:52:23 2022

@author: Lard Fogh - s163761
"""


import openseespy.opensees as ops
import numpy as np


Pa = 1;  MPa = Pa*1e6;  GPa = Pa*1e9
m = 1;  mm = 0.001*m



def define_RCSections():
    
    # ##############################################
    # Definition of materials IDs
    # ##############################################
    
    # Material tags
    matUConTag = 1 	# material tag -- unconfined cover concrete
    matCConTag = 2 	# material tag -- confined core concrete
    matReinf =  3 	# material tag -- reinforcement 
    MatTags = [matUConTag, matCConTag, matReinf]
    
    
    # nominal concrete compressive strength
    fc = -20*MPa 	# CONCRETE Compressive Strength, MPa   (+Tension, -Compression)
    
    # Unconfined cover concrete
    fc1U = fc 			    # UNCONFINED concrete (todeschini parabolic model), maximum stress
    eps1U = -0.003 			# strain at maximum strength of unconfined concrete
    fc2U = 0.2*fc1U 		# ultimate stress
    eps2U = -0.01 			# strain at ultimate stress
    lambda0 =  0.1 			# ratio between unloading slope at $eps2 and initial slope $Ec
    Ec0 = 2*fc1U/eps1U      # Concrete Elastic Modulus
    
    # Confined core concrete
    Kfc = 1.3 			# ratio of confined to unconfined concrete strength
    
    fc1C = Kfc*fc		# CONFINED concrete (mander model), maximum stress
    eps1C = 2.*fc1C/Ec0 # strain at maximum stress 
    fc2C = 0.2*fc1C		# ultimate stress
    eps2C = 5*eps1C		# strain at ultimate stress 
      
    # tensile-strength properties
    ftC = -0.14*fc1C 		# tensile strength +tension
    ftU = -0.14*fc1U 		# tensile strength +tension
    Ets = ftU/0.002 		# tension softening stiffness
    
    
    
    # Steel - Basic parameters for S550
    Fy = 550*MPa 		# STEEL yield stress
    Es = 210*GPa 		# modulus of steel
    b = 0.02 			# strain-hardening ratio 
    R0 = 18 			# control the transition from elastic to plastic branches
    cR1 = 0.925 		# control the transition from elastic to plastic branches
    cR2 = 0.15 			# control the transition from elastic to plastic branches
    
    
    ops.uniaxialMaterial('Concrete02', MatTags[0], fc1U, eps1U, fc2U, eps2U, lambda0, ftC, Ets) # build cover concrete (unconfined)
    ops.uniaxialMaterial('Concrete02', MatTags[1], fc1C, eps1C, fc2C, eps2C, lambda0, ftC, Ets) # build cover concrete (confined)
    
    ops.uniaxialMaterial('Steel02', MatTags[2], Fy, Es, b, *[R0,cR1,cR2] ) # build reinforcement material
        
    
    
    # ##############################################
    # Definition of sections IDs
    # ##############################################
    
    RC_col_tag = 1  # Column section tag
    RC_beam_tag = 2 # Beam section tag
    
    
    # Define Column geometry
    H_col = 400*mm 		 # Height
    B_col = 400*mm	     # Width
    coverH_col = 25*mm	 # Cover to reinforcing steel NA, parallel to H
    coverB_col = 25*mm   # Cover to reinforcing steel NA, parallel to B
    
    numBarsTop_col = 3		# number of longitudinal-reinforcement bars in steel layer. -- top
    numBarsBot_col = 3		# number of longitudinal-reinforcement bars in steel layer. -- bot
    numBarsIntTot_col = 2	# number of longitudinal-reinforcement bars in steel layer. -- total intermediate skin reinforcement, symm about y-axis
    NumBars_col = [numBarsTop_col, numBarsBot_col, numBarsIntTot_col]
    
    
    diaBarsTop_col = 12*mm # Bar individual bar diameter top
    diaBarsBot_col = 12*mm # Bar individual bar diameter bottom
    diaBarsInt_col = 12*mm # Bar individual bar diameter intermediate
    
    barAreaTop_col = np.pi*diaBarsTop_col**2/4	# area of longitudinal-reinforcement bars -- top
    barAreaBot_col = np.pi*diaBarsBot_col**2/4	# area of longitudinal-reinforcement bars -- bot
    barAreaInt_col = np.pi*diaBarsInt_col**2/4	# area of longitudinal-reinforcement bars -- intermediate skin reinf
    AreaBars_col = [barAreaTop_col, barAreaBot_col, barAreaInt_col]
    
    # Define Beam geometry
    H_beam = 500*mm 		# Column Depth
    B_beam = 400*mm	# Column Width
    coverH_beam = 25*mm		# Column cover to reinforcing steel NA, parallel to H
    coverB_beam = 25*mm		# Column cover to reinforcing steel NA, parallel to B
    
    numBarsTop_beam = 3		# number of longitudinal-reinforcement bars in steel layer. -- top
    numBarsBot_beam = 6		# number of longitudinal-reinforcement bars in steel layer. -- bot
    numBarsIntTot_beam = 0			# number of longitudinal-reinforcement bars in steel layer. -- total intermediate skin reinforcement, symm about y-axis
    NumBars_beam = [numBarsTop_beam, numBarsBot_beam, numBarsIntTot_beam]
    
    
    diaBarsTop_beam = 12*mm # Bar individual bar diameter top
    diaBarsBot_beam = 12*mm # Bar individual bar diameter bottom
    diaBarsInt_beam = 0*mm # Bar individual bar diameter intermediate
    
    barAreaTop_beam = np.pi*diaBarsTop_beam**2/4	# area of longitudinal-reinforcement bars -- top
    barAreaBot_beam = np.pi*diaBarsBot_beam**2/4	# area of longitudinal-reinforcement bars -- bot
    barAreaInt_beam = np.pi*diaBarsInt_beam**2/4	# area of longitudinal-reinforcement bars -- intermediate skin reinf
    AreaBars_beam = [barAreaTop_beam, barAreaBot_beam, barAreaInt_beam]
    
    
    # Define rectangular beam
    RCSection(RC_col_tag , MatTags,  H_col,  B_col, coverH_col, coverB_col, NumBars_col, AreaBars_col)
    RCSection(RC_beam_tag , MatTags,  H_beam,  B_beam, coverH_beam, coverB_beam, NumBars_beam, AreaBars_beam)
              
    return RC_col_tag, RC_beam_tag



def RCSection(secID, matTags, HSec, BSec, coverH, coverB, numBars, areaBars):
    
    '''    
    # FIBER SECTION properties -------------------------------------------------------------
    #
    #                        y
    #                        ^
    #                        |     
    #             ---------------------     ---   --
    #             |   o     o     o    |     |    -- coverH
    #             |                    |     |
    #             |   o            o   |     |
    #    z <---   |          +         |     Hsec
    #             |   o            o   |     |
    #             |                    |     |
    #             |   o  o o  o o  o   |     |    -- coverH
    #             ---------------------     ---   --
    #             |-------Bsec--------|
    #             |----| coverB  |----|
    #
    #                       y
    #                       ^
    #                       |    
    #             ---------------------
    #             |\      cover       /|
    #             | \------Top------ / |
    #             |c|                |c|
    #             |o|                |o|
    #  z <-----   |v|       core     |v|  Hsec
    #             |e|                |e|
    #             |r|                |r|
    #             | /-------Bot------\ |
    #             |/      cover       \|
    #             ---------------------
    #                       Bsec
    #
    # Notes
    #    The core concrete ends at the NA of the reinforcement
    #    The center of the section is at (0,0) in the local axis system
    '''

    
    coverY = HSec/2	    # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
    coverZ = BSec/2	    # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
    coreY = coverY-coverH	# The distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
    coreZ = coverZ-coverB	# The distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concreteset nfY 16;			# number of fibers for concrete in y-direction
    nfY = 20			# number of fibers for concrete in y-direction
    nfZ = 1				# number of fibers for concrete in z-direction
    numBarsInt = numBars[2]/2	# number of intermediate bars per side  



    # Define Sections
    #  section('Fiber', secTag)
    ops.section('Fiber', secID)
    #patch   ('quad', matTag, numSubdivIJ, numSubdivJK, *crdsI, *crdsJ, *crdsK, *crdsL)
    ops.patch('quad', matTags[1], nfZ, nfY,        -coreY, coreZ, -coreY, -coreZ, coreY, -coreZ, coreY, coreZ ) # Define the core patch
    #
    ops.patch('quad', matTags[0], 1, nfY,          -coverY, coverZ,  -coreY,   coreZ,  coreY,   coreZ, coverY, coverZ ) # Define the four cover patches
    ops.patch('quad', matTags[0], 1, nfY,           -coreY, -coreZ, -coverY, -coverZ, coverY, -coverZ,  coreY, -coreZ )
    ops.patch('quad', matTags[0], nfZ, 1,          -coverY, coverZ, -coverY, -coverZ, -coreY,  -coreZ, -coreY,  coreZ )
    ops.patch('quad', matTags[0], nfZ, 1,            coreY,  coreZ,   coreY,  -coreZ, coverY, -coverZ, coverY, coverZ )
    # #
    if numBarsInt != 0 and areaBars[2] != 0:
        ops.layer('straight', matTags[2], numBarsInt, areaBars[2],  -coreY,  coreZ,   coreY,  coreZ )	# intermediate skin reinf. +z
        ops.layer('straight', matTags[2], numBarsInt, areaBars[2],  -coreY, -coreZ,   coreY, -coreZ )	# intermediate skin reinf. -z
    ops.layer('straight', matTags[2], numBars[0], areaBars[0],   coreY,  coreZ,   coreY, -coreZ )	# top layer reinfocement
    ops.layer('straight', matTags[2], numBars[1], areaBars[1],  -coreY,  coreZ,  -coreY, -coreZ )	# bottom layer reinforcement
        
    return