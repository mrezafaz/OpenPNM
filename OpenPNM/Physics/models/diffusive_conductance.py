r"""
===============================================================================
Submodule -- diffusive_conductance
===============================================================================

"""

import scipy as _sp
import OpenPNM.Utilities.misc as misc

def bulk_diffusion(physics,
                   phase,
                   network,
                   pore_molar_density='pore.molar_density',
                   pore_diffusivity='pore.diffusivity',
                   pore_area='pore.area',
                   pore_diameter='pore.diameter',
                   throat_area='throat.area',
                   throat_length='throat.length',
                   throat_diameter='throat.diameter',
                   calc_pore_len=True,
                   **kwargs):
    r"""
    Calculate the diffusive conductance of conduits in network, where a 
    conduit is ( 1/2 pore - full throat - 1/2 pore ) based on the areas

    Parameters
    ----------
    network : OpenPNM Network Object

    phase : OpenPNM Phase Object
        The phase of interest

    Notes
    -----
    (1) This function requires that all the necessary phase properties already 
    be calculated.
    
    (2) This function calculates the specified property for the *entire* 
    network then extracts the values for the appropriate throats at the end.
    
    """    
    #Get Nt-by-2 list of pores connected to each throat
    Ps = network['throat.conns']
    #Get properties in every pore in the network
    parea = network[pore_area]
    pdia = network[pore_diameter]
    #Get the properties of every throat
    tarea = network[throat_area]
    tlen = network[throat_length]
    #Interpolate pore phase property values to throats
    cp = phase[pore_molar_density]
    ct = phase.interpolate_data(data=cp)
    DABp = phase[pore_diffusivity]
    DABt = phase.interpolate_data(data=DABp)
    if calc_pore_len:
        lengths = misc.conduit_lengths(network,mode='centroid')
        plen1 = lengths[:,0]
        plen2 = lengths[:,2]
    else:        
        plen1 = (0.5*pdia[Ps[:,0]])
        plen2 = (0.5*pdia[Ps[:,1]])
    #remove any non-positive lengths
    plen1[plen1<=0]=1e-12
    plen2[plen2<=0]=1e-12
    #Find g for half of pore 1
    gp1 = ct*DABt*parea[Ps[:,0]]/plen1
    gp1[_sp.isnan(gp1)] = _sp.inf
    gp1[~(gp1>0)] = _sp.inf  # Set 0 conductance pores (boundaries) to inf
    #Find g for half of pore 2
    gp2 = ct*DABt*parea[Ps[:,1]]/plen2
    gp2[_sp.isnan(gp2)] = _sp.inf
    gp2[~(gp2>0)] = _sp.inf  # Set 0 conductance pores (boundaries) to inf
    #Find g for full throat
    #remove any non-positive lengths
    tlen[tlen<=0] = 1e-12
    gt = ct*DABt*tarea/tlen
    value = (1/gt + 1/gp1 + 1/gp2)**(-1)
    value = value[phase.throats(physics.name)]
    return value
    

def mpl_plus_bulk_diffusion(physics,
                   phase,
                   network,
                   pore_molar_density='pore.molar_density',
                   pore_diffusivity='pore.diffusivity',
                   pore_area='pore.area',
                   pore_diameter='pore.diameter',
                   throat_area='throat.area',
                   throat_length='throat.length',
                   throat_diameter='throat.diameter',
                   SD = 0.225,
                   calc_pore_len=True,
                   **kwargs):
    r"""
    Calculate the diffusive conductance of conduits in network, where a 
    conduit is ( 1/2 pore - full throat - 1/2 pore ) based on the areas

    Parameters
    ----------
    network : OpenPNM Network Object

    phase : OpenPNM Phase Object
        The phase of interest

    Notes
    -----
    (1) This function requires that all the necessary phase properties already 
    be calculated.
    
    (2) This function calculates the specified property for the *entire* 
    network then extracts the values for the appropriate throats at the end.
    
    """
    pore = _sp.ravel(network['pore.material']==0)
    mpl_element = _sp.ravel(network['pore.material']==1)
#    
    pore_pore = _sp.ravel(network['throat.material']==0)    
    #Get Nt-by-2 list of pores connected to each throat
    Ps = network['throat.conns']
    #Get properties in every pore in the network
    parea = network[pore_area]
    pdia = network[pore_diameter]
    #Get the properties of every throat
    tarea = network[throat_area]
    tlen = network[throat_length]
    #Interpolate pore phase property values to throats
    cp = phase[pore_molar_density]
    ct = phase.interpolate_data(data=cp)
    DABp = phase[pore_diffusivity]
    DABt = phase.interpolate_data(data=DABp)
    if calc_pore_len:
        lengths = misc.conduit_lengths(network,mode='centroid')
        plen1 = lengths[:,0]
        plen2 = lengths[:,2]
    else:        
        plen1 = (0.5*pdia[Ps[:,0]])
        plen2 = (0.5*pdia[Ps[:,1]])
    #remove any non-positive lengths
    plen1[plen1<=0]=1e-12
    plen2[plen2<=0]=1e-12
    #Find g for half of pore 1
    gp1 = ct*DABt*parea[Ps[:,0]]/plen1
    gp1[_sp.isnan(gp1)] = _sp.inf
    gp1[~(gp1>0)] = _sp.inf  # Set 0 conductance pores (boundaries) to inf
    #Find g for half of pore 2
    gp2 = ct*DABt*parea[Ps[:,1]]/plen2
    gp2[_sp.isnan(gp2)] = _sp.inf
    gp2[~(gp2>0)] = _sp.inf  # Set 0 conductance pores (boundaries) to inf
    #Find g for full throat
    #remove any non-positive lengths
    tlen[tlen<=0] = 1e-12
    gt = ct*DABt*tarea/tlen
    value = _sp.zeros(_sp.shape(Ps)[0])
    value[pore_pore] = (1/gt[pore_pore] + 1/gp1[pore_pore] + 1/gp2[pore_pore])**(-1)
    
    mple_mple = _sp.ravel(network['throat.material']==1)
    x = network['pore.coords'][:,0]
    y = network['pore.coords'][:,1]
    z = network['pore.coords'][:,2]
    d_centers = _sp.sqrt((x[Ps[:,1]]-x[Ps[:,0]])**2+(y[Ps[:,1]]-y[Ps[:,0]])**2+(z[Ps[:,1]]-z[Ps[:,0]])**2)
    mple_mple_area = network['throat.mplarea']
    value[mple_mple] = ct[mple_mple]*DABt[mple_mple]*mple_mple_area[mple_mple]*SD/d_centers[mple_mple]
    
    mple_pore = _sp.ravel(network['throat.material']==2)
    mlen = (0.5*pdia[Ps[:,1]][mple_pore])
    plen = d_centers[mple_pore] - mlen
    plen[plen<=0]=1e-12
    gp = ct[mple_pore]*DABp[Ps[:,0]][mple_pore]*parea[Ps[:,0]][mple_pore]/plen #Turns out the the first element is a pore and the second on is an mple
    gp[_sp.isnan(gp)] = _sp.inf
    gp[~(gp>0)] = _sp.inf 
    gm = ct[mple_pore]*DABp[Ps[:,1]][mple_pore]*mple_mple_area[mple_pore]*SD/mlen
    gm[_sp.isnan(gm)] = _sp.inf
    gm[~(gm>0)] = _sp.inf
    value[mple_pore] = (1/gp + 1/gm)**(-1)
    
    value = value[phase.throats(physics.name)]    
    return value