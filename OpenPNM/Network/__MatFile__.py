# -*- coding: utf-8 -*-
"""
===============================================================================
MatFile: Subclass to import Networks from a Matlab '.mat' file
===============================================================================

"""
import scipy as sp
import scipy.io as spio
import os
from OpenPNM.Network import GenericNetwork
import OpenPNM.Geometry
import OpenPNM.Utilities.misc as misc
from OpenPNM.Base import logging
logger = logging.getLogger(__name__)

class MatFile(GenericNetwork):
    r"""
    MatFile - constructs a pore network from a perfectly formatted .mat file (MATLAB)
    Create network from Matlab file. Returns OpenPNM.Network.GenericNetwork()
    object. The extra data of 'type' will trigger internal and boundary pores.

    Parameters
    ----------
    filename : string
        filename = 'standard_cubic_5x5x5.mat' (default)
        Name of mat file
    path : string
        path='' (default)
        the full path to the mat file on your computer
        leaving blank searches for the file in the local directory
    xtra_pore_data : list of strings
        xtra_pore_data = ['type','shape','material']
        any additional props to look for in the dictionary
    xtra_throat_data : list of strings
        xtra_throat_data = ['type','shape','material']
        any additional props to look for in the dictionary

    Examples
    ---------
    >>> import OpenPNM, os
    >>> fname = 'test_pn' # or 'test_pn.mat'
    >>> longpath = os.path.abspath(__file__) # line needed for auto-tests
    >>> path, file = os.path.split(longpath)  # unfortunately, auto-tests fail without this. Just type in the path of your own file, or leave it blank to search in your current directory.
    >>> pn = OpenPNM.Network.MatFile(filename=fname, path=path, xtra_pore_data='type', xtra_throat_data='type')

    Notes
    ------
    Matfiles should include the following variables

    +----------------+------------+----------------------------------+
    | Variable Name  | Value      | Description                      |
    +================+============+==================================+
    | pcoords        | <Npx3>     | physical coordinates, in meters, |
    |                | float      | of pores to be imported          |
    +----------------+------------+----------------------------------+
    | pdiameter      | <Npx1>     | pore diamters, in meters         |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | pvolume        | <Npx1>     | pore volumes, in cubic meters    |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | pnumbering     | <Npx1>     | = 0:1:Np-1                       |
    |                | int        |                                  |
    +----------------+------------+----------------------------------+
    | ptype          | <Npx1>     | (optional) designates surfaces   |
    |                | int        | of pores in network.             |
    |                |            | (more details below)             |
    +----------------+------------+----------------------------------+
    | tconnections   | <Ntx2>     | pore numbers of the two pores    |
    |                | int        | that each throat connects        |
    +----------------+------------+----------------------------------+
    | tdiameter      | <Ntx1>     | throat diameters, in meters      |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | tnumbering     | <Ntx1>     | = 0:1:Nt-1                       |
    |                | int        |                                  |
    +----------------+------------+----------------------------------+
    | ttype          | <Ntx1>     | (optional) designates surfaces   |
    |                | int        | of throats in network.           |
    |                |            | (more details below)             |
    +----------------+------------+----------------------------------+
    """

    def __init__(self,filename='', path='', xtra_pore_data=None, xtra_throat_data=None,**kwargs):

        r"""
        """
        super(MatFile,self).__init__(**kwargs)
        if filename == '':
            return
        if path == '':
            path = os.path.abspath('.')
        self._path = path
        filepath = os.path.join(self._path,filename)
        self._xtra_pore_data=xtra_pore_data
        self._xtra_throat_data=xtra_throat_data
        self._dictionary=spio.loadmat(filepath)

        self._Np=sp.size(self._dictionary['pnumbering'])
        self._Nt=sp.size(self._dictionary['tnumbering'])

        #Run through generation steps
        self._add_pores()
        self._add_throats()
        self._remove_disconnected_clusters()
        self._add_xtra_pore_data()
        self._add_xtra_throat_data()
        self._add_geometry()

    def _add_pores(self):
        Pind = sp.arange(0,self._Np)
        self['pore.all'] = sp.ones_like(Pind,dtype=bool)
        logger.info('Writing pore data')
        self['pore.coords']=sp.array(self._dictionary['pcoords'],float)

    def _add_throats(self):
        Tind = sp.arange(0,self._Nt)
        self['throat.all']=sp.ones_like(Tind,dtype=bool)
        logger.info('Writing throat data')
        self['throat.conns']=sp.array(self._dictionary['tconnections'],int)

    def _remove_disconnected_clusters(self):
        bad_pores = sp.array([],dtype=int)
        self._pore_map = self.pores()
        self._throat_map = self.throats()
        health = self.check_network_health()
        if health['disconnected_clusters'] == []:
            self._throat_map = self.throats()
            self._pore_map = self.pores()
        else:
            Np = self.num_pores()
            Nt = self.num_throats()
            cluster_sizes = [sp.shape(x)[0] for x in health['disconnected_clusters']]
            acceptable_size = min([min([50,Np/2]),max(cluster_sizes)]) # 50 or less, if it's a really small network.
            #step through each cluster of pores. If its a small cluster, add it to the list
            for cluster in health['disconnected_clusters']:
                if sp.shape(cluster)[0] < acceptable_size:
                    bad_pores = sp.append(bad_pores,sp.ravel(cluster))
            bad_throats = sp.unique(self.find_neighbor_throats(bad_pores))
            #Create map for pores
            if sp.shape(bad_pores)[0] > 0:
                i = 0
                self._pore_map = sp.zeros((Np-sp.shape(bad_pores)[0],),dtype=int)
                for pore in self.pores():
                    if pore not in bad_pores:
                        self._pore_map[i] = pore
                        i += 1
            #Create map for throats
            if sp.shape(bad_throats)[0] > 0:
                i = 0
                self._throat_map = sp.zeros((Nt-sp.shape(bad_throats)[0],),dtype=int)
                for throat in self.throats():
                    if throat not in bad_throats:
                        self._throat_map[i] = throat
                        i += 1
            #Fix the pore transformer
            try:
                if sp.shape(bad_pores)[0] > 0:
                    i = 0
                    old_transform = self._dictionary['pname_transform']
                    self._dictionary['pname_transform'] = sp.zeros((Np-sp.shape(bad_pores)[0],),dtype=int)
                    for pore in self.pores():
                        if pore not in bad_pores:
                            self._dictionary['pname_transform'][i] = old_transform[pore]
                            i += 1
            except:
                logger.info('Could not update pname_transform. Imported network may not have had it.')
                pass
            self.trim(pores=bad_pores)

    def _add_geometry(self):
        try:
            boundary_pores = sp.where(self['pore.type']!=0)[0]
            boundary_throats = sp.where(self['throat.type']!=0)[0]
            self['throat.top'] = sp.ravel(self['throat.type']==6)
            self['throat.bottom'] = sp.ravel(self['throat.type']==1)
            self['throat.left'] = sp.ravel(self['throat.type']==2)
            self['throat.right'] = sp.ravel(self['throat.type']==5)
            self['throat.front'] = sp.ravel(self['throat.type']==3)
            self['throat.back'] = sp.ravel(self['throat.type']==4)
            self['pore.top'] = self.tomask(pores=sp.ravel(self.find_connected_pores(self.throats('top'))))
            self['pore.bottom'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('bottom'))))
            self['pore.left'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('left'))))
            self['pore.right'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('right'))))
            self['pore.front'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('front'))))
            self['pore.back'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('back'))))
            add_boundaries = True
        except:
            boundary_pores = sp.array([])
            boundary_throats = sp.array([])
            logger.info('No boundary pores added.')
            add_boundaries = False
        Ps = sp.where([pore not in boundary_pores for pore in self.pores()])[0]
        Ts = sp.where([throat not in boundary_throats for throat in self.throats()])[0]
        geom = OpenPNM.Geometry.GenericGeometry(network=self,pores=Ps,throats=Ts,name='internal')
        geom['pore.volume'] = sp.ravel(sp.array(self._dictionary['pvolume'][self._pore_map[Ps]],float))
        geom['pore.diameter'] = sp.ravel(sp.array(self._dictionary['pdiameter'][self._pore_map[Ps]],float))
        geom['throat.diameter'] = sp.ravel(sp.array(self._dictionary['tdiameter'][self._throat_map[Ts]],float))
        geom.add_model(propname='pore.area',model=OpenPNM.Geometry.models.pore_area.spherical)
        geom.add_model(propname='throat.area',model=OpenPNM.Geometry.models.throat_area.cylinder)

        if add_boundaries:
            OpenPNM.Geometry.Boundary(network=self,pores=boundary_pores,throats=boundary_throats,name='boundary')
            self['pore.top_boundary']=self.tomask(pores=self.pores(['top','boundary'],mode='intersection'))
            self['pore.bottom_boundary']=self.tomask(pores=self.pores(['bottom','boundary'],mode='intersection'))
            self['pore.left_boundary']=self.tomask(pores=self.pores(['left','boundary'],mode='intersection'))
            self['pore.right_boundary']=self.tomask(pores=self.pores(['right','boundary'],mode='intersection'))
            self['pore.front_boundary']=self.tomask(pores=self.pores(['front','boundary'],mode='intersection'))
            self['pore.back_boundary']=self.tomask(pores=self.pores(['back','boundary'],mode='intersection'))

            self['throat.top_boundary']=self.tomask(throats=self.throats(['top','boundary'],mode='intersection'))
            self['throat.bottom_boundary']=self.tomask(throats=self.throats(['bottom','boundary'],mode='intersection'))
            self['throat.left_boundary']=self.tomask(throats=self.throats(['left','boundary'],mode='intersection'))
            self['throat.right_boundary']=self.tomask(throats=self.throats(['right','boundary'],mode='intersection'))
            self['throat.front_boundary']=self.tomask(throats=self.throats(['front','boundary'],mode='intersection'))
            self['throat.back_boundary']=self.tomask(throats=self.throats(['back','boundary'],mode='intersection'))

    def _add_xtra_pore_data(self):
        xpdata = self._xtra_pore_data
        if xpdata is not None:
            if type(xpdata) is type([]):
                for pdata in xpdata:
                    try:
                        self['pore.'+pdata]=self._dictionary['p'+pdata][self._pore_map]
                    except:
                        logger.warning('Could not add pore data: '+pdata+' to network')
                        pass
            else:
                try:
                    self['pore.'+xpdata]=self._dictionary['p'+xpdata][self._pore_map]
                except:
                    logger.warning('Could not add pore data: '+xpdata+' to network')
                    pass

    def _add_xtra_throat_data(self):
        xtdata = self._xtra_throat_data
        if xtdata is not None:
            if type(xtdata) is type([]):
                for tdata in xtdata:
                    try:
                        self['throat.'+tdata]=self._dictionary['t'+tdata][self._throat_map]
                    except:
                        logger.warning('Could not add throat data: '+tdata+' to network')
                        pass
            else:
                try:
                    self['throat.'+xtdata]=self._dictionary['t'+xtdata][self._throat_map]
                except:
                    logger.warning('Could not add throat data: '+xtdata+' to network')
                    pass

    def domain_length(self,face_1,face_2):
        r"""
        Calculate the distance between two faces of the network

        Parameters
        ----------
        face_1 and face_2 : array_like
            Lists of pores belonging to opposite faces of the network

        Returns
        -------
        The length of the domain in the specified direction

        Notes
        -----
        - Does not yet check if input faces are perpendicular to each other
        """
        #Ensure given points are coplanar before proceeding
        if misc.iscoplanar(self['pore.coords'][face_1]) and misc.iscoplanar(self['pore.coords'][face_2]):
            #Find distance between given faces
            x = self['pore.coords'][face_1]
            y = self['pore.coords'][face_2]
            Ds = misc.dist(x,y)
            L = sp.median(sp.amin(Ds,axis=0))
        else:
            logger.warning('The supplied pores are not coplanar. Length will be approximate.')
            f1 = self['pore.coords'][face_1]
            f2 = self['pore.coords'][face_2]
            distavg = [0,0,0]
            distavg[0] = sp.absolute(sp.average(f1[:,0]) - sp.average(f2[:,0]))
            distavg[1] = sp.absolute(sp.average(f1[:,1]) - sp.average(f2[:,1]))
            distavg[2] = sp.absolute(sp.average(f1[:,2]) - sp.average(f2[:,2]))
            L = max(distavg)
        return L


    def domain_area(self,face):
        r"""
        Calculate the area of a given network face

        Parameters
        ----------
        face : array_like
            List of pores of pore defining the face of interest

        Returns
        -------
        The area of the specified face
        """
        coords = self['pore.coords'][face]
        rads = self['pore.diameter'][face]/2.
        # calculate the area of the 3 principle faces of the bounding cuboid
        dx = max(coords[:,0]+rads) - min(coords[:,0]-rads)
        dy = max(coords[:,1]+rads) - min(coords[:,1]-rads)
        dz = max(coords[:,2]+rads) - min(coords[:,2]-rads)
        yz = dy*dz # x normal
        xz = dx*dz # y normal
        xy = dx*dy # z normal
        # find the directions parallel to the plane
        directions = sp.where([yz,xz,xy]!=max([yz,xz,xy]))[0]
        try:
            # now, use the whole network to do the area calculation
            coords = self['pore.coords']
            rads = self['pore.diameter']/2.
            d0 = (max(coords[:,directions[0]]+rads) - min(coords[:,directions[0]]-rads))
            d1 = (max(coords[:,directions[1]]+rads) - min(coords[:,directions[1]]-rads))
            A = d0*d1
        except:
            # if that fails, use the max face area of the bounding cuboid
            A = max([yz,xz,xy])
        if not misc.iscoplanar(self['pore.coords'][face]):
            logger.warning('The supplied pores are not coplanar. Area will be approximate')
            pass
        return A
        
class MPLMatFile(MatFile):
    r"""
    MPLMatFile

    Parameters
    ----------

    Examples
    ---------

    Notes
    ------
    Matfiles should include the following variables

    +----------------+------------+----------------------------------+
    | Variable Name  | Value      | Description                      |
    +================+============+==================================+
    | pcoords        | <Npx3>     | physical coordinates, in meters, |
    |                | float      | of pores to be imported          |
    +----------------+------------+----------------------------------+
    | pdiameter      | <Npx1>     | pore diamters, in meters         |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | pvolume        | <Npx1>     | pore volumes, in cubic meters    |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | pnumbering     | <Npx1>     | = 0:1:Np-1                       |
    |                | int        |                                  |
    +----------------+------------+----------------------------------+
    | ptype          | <Npx1>     | (optional) designates surfaces   |
    |                | int        | of pores in network.             |
    |                |            | (more details below)             |
    +----------------+------------+----------------------------------+
    | pmaterial      | <Npx1>     | binary array: 0 is GDL pore      |
    |                | int        | 1 is MPL element                 |
    +----------------+------------+----------------------------------+
    | tconnections   | <Ntx2>     | pore numbers of the two pores    |
    |                | int        | that each throat connects        |
    +----------------+------------+----------------------------------+
    | tdiameter      | <Ntx1>     | throat diameters, in meters      |
    |                | float      |                                  |
    +----------------+------------+----------------------------------+
    | tnumbering     | <Ntx1>     | = 0:1:Nt-1                       |
    |                | int        |                                  |
    +----------------+------------+----------------------------------+
    | ttype          | <Ntx1>     | (optional) designates surfaces   |
    |                | int        | of throats in network.           |
    |                |            | (more details below)             |
    +----------------+------------+----------------------------------+
    | tmaterial      | <Ntx1>     | indicates the type of connection |
    |                | int        | 0 is GDL pore - GDL pore         |
    |                |            | 1 is MPL element - MPL element   |
    |                |            | 2 is GDL pore - MPL element      |
    +----------------+------------+----------------------------------+
    | tmplarea       | <Ntx1>     | contact area between each MPL    |
    |                | int        | element and its neighbour MPL    |
    |                |            | element/GDL pore                  |
    +----------------+------------+----------------------------------+
    """

    def __init__(self,**kwargs):

        r"""
        """
        super(MPLMatFile,self).__init__(**kwargs)
        
    def _add_geometry(self):
        self._add_GDL_geometry()
        self._add_MPL_geometry()
        self._add_interface_geometry()
        
    def _add_GDL_geometry(self):
        try:
            self['throat.GDL_top'] = sp.ravel((self['throat.type']==6) & (self['throat.material']==0))
            self['throat.GDL_bottom'] = sp.ravel((self['throat.type']==1) & (self['throat.material']==0))
            self['throat.GDL_left'] = sp.ravel((self['throat.type']==2) & (self['throat.material']==0))
            self['throat.GDL_right'] = sp.ravel((self['throat.type']==5) & (self['throat.material']==0))
            self['throat.GDL_front'] = sp.ravel((self['throat.type']==3) & (self['throat.material']==0))
            self['throat.GDL_back'] = sp.ravel((self['throat.type']==4) & (self['throat.material']==0))
            self['pore.GDL_top'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_top'))))
            self['pore.GDL_bottom'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_bottom'))))
            self['pore.GDL_left'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_left'))))
            self['pore.GDL_right'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_right'))))
            self['pore.GDL_front'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_front'))))
            self['pore.GDL_back'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('GDL_back'))))
            add_GDL_boundaries = True
        except:
            logger.info('No boundary pores added.')
            add_GDL_boundaries = False
        GDL_internal_pores = sp.where((self['pore.type']==0) & (self['pore.material']==0))[0]
        GDL_internal_throats = sp.where((self['throat.type']==0) & (self['throat.material']==0))[0]
        GDL_internal_geom = OpenPNM.Geometry.GenericGeometry(network=self,pores=GDL_internal_pores,throats=GDL_internal_throats,name='GDL_internal')
        GDL_internal_geom['pore.volume'] = sp.ravel(sp.array(self._dictionary['pvolume'][self._pore_map[GDL_internal_pores]],float))
        GDL_internal_geom['pore.diameter'] = sp.ravel(sp.array(self._dictionary['pdiameter'][self._pore_map[GDL_internal_pores]],float))
        GDL_internal_geom['throat.diameter'] = sp.ravel(sp.array(self._dictionary['tdiameter'][self._throat_map[GDL_internal_throats]],float))
        GDL_internal_geom.add_model(propname='pore.area',model=OpenPNM.Geometry.models.pore_area.spherical)
        GDL_internal_geom.add_model(propname='throat.area',model=OpenPNM.Geometry.models.throat_area.cylinder)

        if add_GDL_boundaries:
            GDL_boundary_pores = sp.where((self['pore.type']!=0) & (self['pore.material']==0))[0]
            GDL_boundary_throats = sp.where((self['throat.type']!=0) & (self['throat.material']==0))[0]
            OpenPNM.Geometry.Boundary(network=self,pores=GDL_boundary_pores,throats=GDL_boundary_throats,name='GDL_boundary')
            self['pore.GDL_top_boundary']=self.tomask(pores=self.pores(['GDL_top','GDL_boundary'],mode='intersection'))
            self['pore.GDL_bottom_boundary']=self.tomask(pores=self.pores(['GDL_bottom','GDL_boundary'],mode='intersection'))
            self['pore.GDL_left_boundary']=self.tomask(pores=self.pores(['GDL_left','GDL_boundary'],mode='intersection'))
            self['pore.GDL_right_boundary']=self.tomask(pores=self.pores(['GDL_right','GDL_boundary'],mode='intersection'))
            self['pore.GDL_front_boundary']=self.tomask(pores=self.pores(['GDL_front','GDL_boundary'],mode='intersection'))
            self['pore.GDL_back_boundary']=self.tomask(pores=self.pores(['GDL_back','GDL_boundary'],mode='intersection'))
        
            self['throat.GDL_top_boundary']=self.tomask(throats=self.throats(['GDL_top','GDL_boundary'],mode='intersection'))
            self['throat.GDL_bottom_boundary']=self.tomask(throats=self.throats(['GDL_bottom','GDL_boundary'],mode='intersection'))
            self['throat.GDL_left_boundary']=self.tomask(throats=self.throats(['GDL_left','GDL_boundary'],mode='intersection'))
            self['throat.GDL_right_boundary']=self.tomask(throats=self.throats(['GDL_right','GDL_boundary'],mode='intersection'))
            self['throat.GDL_front_boundary']=self.tomask(throats=self.throats(['GDL_front','GDL_boundary'],mode='intersection'))
            self['throat.GDL_back_boundary']=self.tomask(throats=self.throats(['GDL_back','GDL_boundary'],mode='intersection'))
            
    def _add_MPL_geometry(self):
        Ps = self['throat.conns']
        pcoords = self['pore.coords']
        d_centers = sp.sqrt(sp.sum(sp.square(pcoords[Ps[:,0]]-pcoords[Ps[:,1]]),1))
        try:
            self['throat.MPL_top'] = sp.ravel((self['throat.type']==6) & (self['throat.material']==1))
            self['throat.MPL_bottom'] = sp.ravel((self['throat.type']==1) & (self['throat.material']==1))
            self['throat.MPL_left'] = sp.ravel((self['throat.type']==2) & (self['throat.material']==1))
            self['throat.MPL_right'] = sp.ravel((self['throat.type']==5) & (self['throat.material']==1))
            self['throat.MPL_front'] = sp.ravel((self['throat.type']==3) & (self['throat.material']==1))
            self['throat.MPL_back'] = sp.ravel((self['throat.type']==4) & (self['throat.material']==1))
            self['pore.MPL_top'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_top'))))
            self['pore.MPL_bottom'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_bottom'))))
            self['pore.MPL_left'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_left'))))
            self['pore.MPL_right'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_right'))))
            self['pore.MPL_front'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_front'))))
            self['pore.MPL_back'] = self.tomask(sp.ravel(self.find_connected_pores(self.throats('MPL_back'))))
            add_MPL_boundaries = True
        except:
            logger.info('No boundary pores added.')
            add_MPL_boundaries = False
        MPL_internal_pores = sp.where((self['pore.type']==0) & (self['pore.material']==1))[0]
        MPL_internal_throats = sp.where((self['throat.type']==0) & (self['throat.material']==1))[0]
        MPL_internal_geom = OpenPNM.Geometry.GenericGeometry(network=self,pores=MPL_internal_pores,throats=MPL_internal_throats,name='MPL_internal')
        MPL_internal_geom['pore.diameter'] = sp.ravel(sp.array(self._dictionary['pdiameter'][self._pore_map[MPL_internal_pores]],float))
        MPL_internal_geom['pore.area'] = MPL_internal_geom['pore.diameter']**2
        MPL_internal_geom['pore.volume'] = sp.ravel(sp.array(self._dictionary['pvolume'][self._pore_map[MPL_internal_pores]],float))
        MPL_internal_geom['throat.area'] = sp.ravel(sp.array(self._dictionary['tmplarea'][self._throat_map[MPL_internal_throats]],float))        
        MPL_internal_geom['throat.diameter'] = 1e-12
        MPL_internal_geom['throat.length'] = d_centers[MPL_internal_throats]/10

        if add_MPL_boundaries:
            MPL_boundary_pores = sp.where((self['pore.type']!=0) & (self['pore.material']==1))[0]
            MPL_boundary_throats = sp.where((self['throat.type']!=0) & (self['throat.material']==1))[0]
            MPL_boundary_geom = OpenPNM.Geometry.Boundary(network=self,pores=MPL_boundary_pores,throats=MPL_boundary_throats,name='MPL_boundary')
            MPL_boundary_geom['throat.area'] = sp.ravel(sp.array(self._dictionary['tmplarea'][self._throat_map[MPL_boundary_throats]],float))
            MPL_boundary_geom['throat.diameter'] = 1e-12
            MPL_boundary_geom['throat.length'] = d_centers[MPL_boundary_throats]/10

            self['pore.MPL_top_boundary']=self.tomask(pores=self.pores(['MPL_top','MPL_boundary'],mode='intersection'))
            self['pore.MPL_bottom_boundary']=self.tomask(pores=self.pores(['MPL_bottom','MPL_boundary'],mode='intersection'))
            self['pore.MPL_left_boundary']=self.tomask(pores=self.pores(['MPL_left','MPL_boundary'],mode='intersection'))
            self['pore.MPL_right_boundary']=self.tomask(pores=self.pores(['MPL_right','MPL_boundary'],mode='intersection'))
            self['pore.MPL_front_boundary']=self.tomask(pores=self.pores(['MPL_front','MPL_boundary'],mode='intersection'))
            self['pore.MPL_back_boundary']=self.tomask(pores=self.pores(['MPL_back','MPL_boundary'],mode='intersection'))
        
            self['throat.MPL_top_boundary']=self.tomask(throats=self.throats(['MPL_top','MPL_boundary'],mode='intersection'))
            self['throat.MPL_bottom_boundary']=self.tomask(throats=self.throats(['MPL_bottom','MPL_boundary'],mode='intersection'))
            self['throat.MPL_left_boundary']=self.tomask(throats=self.throats(['MPL_left','MPL_boundary'],mode='intersection'))
            self['throat.MPL_right_boundary']=self.tomask(throats=self.throats(['MPL_right','MPL_boundary'],mode='intersection'))
            self['throat.MPL_front_boundary']=self.tomask(throats=self.throats(['MPL_front','MPL_boundary'],mode='intersection'))
            self['throat.MPL_back_boundary']=self.tomask(throats=self.throats(['MPL_back','MPL_boundary'],mode='intersection'))
            
    def _add_interface_geometry(self):
        Ps = self['throat.conns']
        pcoords = self['pore.coords']
        d_centers = sp.sqrt(sp.sum(sp.square(pcoords[Ps[:,0]]-pcoords[Ps[:,1]]),1))
        self['throat.interface'] = sp.ravel(self['throat.material']==2)
        interface_throats = sp.where(self['throat.material']==2)[0]
        interface_geom = OpenPNM.Geometry.GenericGeometry(network=self,pores=[],throats=interface_throats,name='interface')
        interface_geom['throat.area'] = sp.ravel(sp.array(self._dictionary['tmplarea'][self._throat_map[interface_throats]],float))        
        interface_geom['throat.diameter'] = 1e-12
        interface_geom['throat.length'] = d_centers[interface_throats]/10
        interface_geom['throat.volume'] = 0.0        
    
if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
