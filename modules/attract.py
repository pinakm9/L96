# Implements class(es) for creating attractor database for a dynamical system

import numpy as np
import matplotlib.pyplot as plt
import tables
import utility as ut
import os
import geom
import scipy.spatial as ss

class AttractorDB:
    """
    Description:
        A class for creating/editing attractor database of a dynamical system

    Attrs:
        db_path: database file path
        func: dynamical function
        dim: dimension of the dynamical system
        params: dict of keyword arguments for the dynamical function
        num_paths: current number of trajectories in the database
        point_description: description of an attractor point as a hdf5 row

    Methods:
        gen_path: generates a new trajectory
        add_new_pts: addes new points to the database
        add_new_paths: adds new trajectories of equal length to the database
        add_to_path: extends an already existing trajectory in the database
        burn_in: moves a point forward in time for a long enough period for it to reach the attractor
        plot_path2D: plots an existing trajectory using only the first two coordinates
        collect_seeds: randomly collects seeds (indices of attractor points) for Voronoi tessellation and saves them in the seeds group
        tessellate: creates Voronoi tessellation from the seeds and saves it
        assign_pts_to_cells: assigns points to Voronoi cells
    """

    def __init__(self, db_path, func, dim, **params):
        """
        Args:
            db_path: database file path
            func: dynamical function
            dim: dimension of the dynamical system
            **params: dict of keyword arguments for the dynamical function
        """
        # initializes database to store attractor data
        self.db_path = db_path
        self.func = func
        self.dim = dim
        self.params = params
        self.point_description = {}
        for i in range(self.dim):
            self.point_description['x' + str(i)] = tables.Float64Col(pos = i)

        if not os.path.isfile(db_path):
            hdf5 = tables.open_file(db_path, 'w')
            hdf5.create_group('/', 'trajectories')
            points = hdf5.create_table(hdf5.root, 'points', self.point_description)
            points.flush()
            self.num_paths = 0
            hdf5.close()
        else:
            # figure out number of trajectories in a non-empty database
            hdf5 = tables.open_file(db_path, 'a')
            idx = [int(path_name.split('_')[-1]) for path_name in hdf5.root.trajectories._v_children]
            self.num_paths = max(idx) if len(idx) > 0 else 0
            hdf5.close()



    @ut.timer
    def gen_path(self, start, length):
        """
        Description:
            Generates a new trajectory

        Args:
            start: the point to start from
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectory

        Returns:
            the generated trajectory as a numpy array
        """
        path = np.zeros((self.dim, length))
        x = start
        for t in range(length):
            res = self.func(x, **self.params)
            path[:, t] = res
            x = res
        return path

    @ut.timer
    def burn_in(self, start='random', burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Description:
            Moves a point forward in time for a long enough period for it to reach the attractor

        Args:
            start: the point to start from, default = 'random' in which case a random point from a normal distribution will be selected
            burn_in_period: amount of time the point is to be moved according to the dynamics, default = 10,000
            mean: mean of the normal distribution from which the starting point is to be selected, default = None which means trajectory will start at zero vector
            cov: cov*Identity is the covarience of the normal distribution from which the starting point is to be selected, default = 0.01

        Returns:
            the final point on the attractor
        """
        if self.dim > 1:
            if mean is None:
                mean = np.zeros(self.dim)
        else:
            if mean is None:
                mean = 0.0

        def new_start():
            print('Invalid starting point, generatimng new random starting point ...')
            return np.random.multivariate_normal(mean, cov*np.eye(self.dim)) if self.dim > 1 else np.random.normal(mean, cov)
        if start is 'random':
            start = new_start()
        end_pt = self.gen_path(start, burn_in_period)[:, -1]
        while np.any(np.isinf(end_pt)):
            end_pt = self.gen_path(new_start(), burn_in_period)[:, -1]
        return end_pt


    @ut.timer
    def add_new_paths(self, num_paths=1, start='random', length=int(1e3), chunk_size=None, burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Description:
            Adds new trajectories of same length to the database

        Args:
            num_paths: number of new trajectories to be added
            start: the list of points to start from, deafult = 'random' in which case random starting points will be created via burn_in
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectories, default = int(1e3)
            chunk_size: portion of the trajectory to be wriiten to the database at a time, default = None, behaves as,
            if length < 1e4:
                chunk_size = length
            elif chunk_size is None:
                chunk_size = int(1e4)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        if length < 1e4:
            chunk_size = length
        elif chunk_size is None:
            chunk_size = int(1e4)
        if start is 'random':
            start = [self.burn_in(burn_in_period=burn_in_period, mean=mean, cov=cov) for i in range(num_paths)]
        for i in range(num_paths):
            self.num_paths += 1
            trajectory = hdf5.create_table(hdf5.root.trajectories, 'trajectory_' + str(self.num_paths), self.point_description)
            origin = start[i]
            for i in range(int(length/chunk_size)):
                path = self.gen_path(origin, chunk_size)
                trajectory.append(path.T)
                trajectory.flush()
                origin = path[:, -1]
                print('Chunk #{} has been written.'.format(i))
        hdf5.close()

    @ut.timer
    def add_to_path(self, path_index, length=int(1e3), chunk_size=None):
        """
        Description:
            Extends an already existing trajectory in the database

        Args:
            path_index: index of the path to be extended
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectory, default = int(1e3)
            chunk_size: portion of the trajectory to be wriiten to the database at a time, default = None, behaves as,
            if length < 1e4:
                chunk_size = length
            elif chunk_size is None:
                chunk_size = int(1e4)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        trajectory = getattr(hdf5.root.trajectories, 'trajectory_' + str(path_index))
        if length < 1e4:
            chunk_size = length
        elif chunk_size is None:
            chunk_size = int(1e4)
        start = np.array(list(trajectory[-1]), dtype = 'float64')
        for i in range(int(length/chunk_size)):
            path = self.gen_path(start, length)
            trajectory.append(path.T)
            trajectory.flush()
            start = path[:, -1]
            print('Chunk #{} has been written.'.format(i))
        hdf5.close()


    @ut.timer
    def add_new_pts(self, num_pts=int(1e3), reset=None, burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Args:
            num_pts: number of new points to be added, default=int(1e3)
            reset: number of points before the starting point resets, default = None, behaves as,
            if num_pts < 1e3:
                reset = num_pts
            elif reset is None:
                reset = int(1e3)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        points = hdf5.root.points
        if num_pts < 1e3:
            reset = num_pts
        elif reset is None:
            reset = int(1e3)
        for i in range(int(num_pts/reset)):
            start = self.burn_in(burn_in_period=burn_in_period, mean=mean, cov=cov)
            path = self.gen_path(start, reset)
            points.append(path.T)
            points.flush()
            print('Chunk #{} has been written.'.format(i))
        hdf5.close()

    @ut.timer
    def collect_seeds(self, num_seeds=int(1e3)):
        """
        Description:
            Randomly collects seeds (indices of attractor points) for Voronoi tessellation and saves them in the seeds group

        Args:
            num_seeds: number of seeds to be collected
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        description = {'index': tables.Int32Col(pos=0)}
        ints = np.random.choice(hdf5.root.points.shape[0], size=num_seeds, replace=False)
        try:
            hdf5.remove_node(hdf5.root.seeds)
        except:
            pass
        seeds = hdf5.create_table(hdf5.root, 'seeds', description)
        seeds.append(np.sort(ints))
        seeds.flush()
        hdf5.close()

    @ut.timer
    def tessellate(self, image_path=None):
        """
        Description:
            Creates Voronoi tessellation from the seeds and saves it

        Args:
            image_path: path of the image ending in the filename to save 2D Voronoi diagram
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        try:
            hdf5.remove_node(where=hdf5.root, name='Voronoi', recursive=True)
        except:
            pass
        seeds = hdf5.root.seeds
        points = hdf5.root.points
        idx = np.array(seeds.read().tolist(), dtype='int32').flatten()
        vt = geom.VoronoiTess(points[idx].tolist())
        hdf5.create_group(hdf5.root, 'Voronoi')
        for i, region in enumerate(vt.tess.point_region):
            if region > 0:
                ver_idx = vt.tess.regions[region]
                if len(ver_idx) > 0 and -1 not in ver_idx:
                    cell = hdf5.create_table(hdf5.root.Voronoi, 'cell_' + str(idx[i]), self.point_description)
                    cell.append(vt.tess.vertices[ver_idx])
                    cell.flush()
        hdf5.close()
        if self.dim == 2 and image_path is not None:
            ss.voronoi_plot_2d(vt.tess, show_vertices=False)
            plt.savefig(image_path)

    @ut.timer
    def assign_pts_to_cells(self):
        """
        Description:
            Assigns points to Voronoi cells
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        try:
            hdf5.remove_node(where=hdf5.root, name='allotments', recursive=True)
        except:
            pass
        hdf5.create_group(hdf5.root, 'allotments')
        for cell in hdf5.walk_nodes(hdf5.root.Voronoi, 'Table'):
            print('Assigning points to {}'.format(cell.name))
            cell_bdry = cell.read().tolist()
            polygon = geom.pts_to_poly(cell_bdry)
            cell_pt_idx = []
            for i, pt in enumerate(hdf5.root.points.read().tolist()):
                if pt in polygon:
                    cell_pt_idx.append(i)
            cell_ = hdf5.create_table(hdf5.root.allotments, cell.name, {'index': tables.Int32Col(pos=0)})
            cell_.append(np.array(cell_pt_idx, dtype='int32'))
            cell_.flush()
        hdf5.close()


    def plot_path2D(self, path_index, show=True, saveas=None):
        """
        Description:
            Plots a trajectory in the database using only the first two coordinates

        Args:
            path_index: index of the path to be plottted
            show: boolean flag to determine if the plot is to be shown, default = True
            saveas: file path for saving the plot, default = None in which case the plot won't be saved

        Returns:
            ax of the generated plot
        """
        plt.figure(figsize = (8,8))
        ax = plt.subplot(111)
        hdf5 = tables.open_file(self.db_path, 'a')
        trajectory = np.array(getattr(hdf5.root.trajectories, 'trajectory_' + str(path_index)).read().tolist(), dtype = 'float64')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], color = 'orange', s = 0.2)
        if show:
            plt.show()
        if saveas is not None:
            plt.savefig(saveas)
        return ax


class AttractorSampler:
    """
    Description:
        Implements attractor sampling from an existing attractor database

    Attrs:
        db_path: path to attractor database
        file: pytables file object representing the database
        db: pyables object representing the root node in the database
        seed_idx: indices of Voronoi seeds in the database
        seeds: coordinates of Voronoi seeds in the database
        dim: dimension of the points in the database
        points: points in the attractor database

    Methods:
        closest_seeds: finds seeds closest to given points
        sample_from_cells: randomly samples points from a list of Voronoi cells
        resample: replaces a list of points with points sampled from their closest Voronoi cells

    """
    def __init__(self, db_path):
        """
        Args:
            db_path: database file path
        """
        self.db_path = db_path
        self.file = tables.open_file(self.db_path, 'r')
        self.db = self.file.root
        self.seed_idx = [int(cell.name.split('_')[-1]) for cell in self.file.walk_nodes(self.db.Voronoi, 'Table')]
        self.seeds = np.array(self.db.points.read().tolist(), dtype='float64')[self.seed_idx]
        self.dim = self.seeds.shape[-1]
        self.seeds = self.seeds.reshape((len(self.seed_idx), self.dim))
        self.points = np.array(self.db.points.read().tolist(), dtype='float64')

    #@ut.timer
    def closest_seeds(self, pts):
        """
        Description:
            Finds seeds closest to given points

        Args:
            pts: list of points whose closest seeds are to be found

        Returns:
            indices of the closest seeds
        """
        cl_seeds = np.zeros(len(pts), dtype='int32')
        dist = np.zeros(len(self.seeds), dtype='float64')
        for j, pt in enumerate(pts):
            for i, seed in enumerate(self.seeds):
                diff = np.array(pt) - seed
                dist[i] = np.dot(diff, diff)
            cl_seeds[j] = self.seed_idx[np.argmin(dist)]
            #print('min dist = {}'.format(min(dist)))
        return cl_seeds

    #@ut.timer
    def sample_from_cells(self, cell_idx, num_pts=1):
        """
        Description:
            Randomly samples points from a list of Voronoi cells
        Args:
            cells: list of indices to Voronoi cells from which points are to be sampled
            num_pts: number of points to be sampled from each cell, default = 1

        Returns:
            list of randomly sampled points
        """
        ensemble = np.zeros((len(cell_idx) * num_pts, self.dim), dtype='float64')
        for i, cell_id in enumerate(cell_idx):
            allot = getattr(self.db.allotments, 'cell_' + str(cell_id)).read().tolist()
            allot = np.array(allot, dtype='int32').flatten()
            ensemble[i*num_pts: (i+1)*num_pts] = self.points[np.random.choice(allot, size=num_pts)]
        return ensemble


    def sample_from_cells2(self, pts, cell_idx, num_pts=1):
        """
        Description:
            Samples points from a list of Voronoi cells w.r.t minimum distance
        Args:
            pts: points to be replaced
            cells: list of indices to Voronoi cells from which points are to be sampled
            num_pts: number of points to be sampled from each cell, default = 1

        Returns:
            list of randomly sampled points
        """
        ensemble = np.zeros((len(cell_idx) * num_pts, self.dim), dtype='float64')
        for i, cell_id in enumerate(cell_idx):
            allot = getattr(self.db.allotments, 'cell_' + str(cell_id)).read().tolist()
            cell_pts = self.points[np.array(allot, dtype='int32').flatten()]
            dist = np.ones(len(cell_pts), dtype='float64')
            for j, cell_pt in enumerate(cell_pts):
                diff = cell_pt - pts[i]
                dist[j] = np.dot(diff, diff)
            ensemble[i] = cell_pts[np.argmin(dist)]
            print('min dist: {}, pt: {}, cell_pt: {}'.format(min(dist), pts[i], ensemble[i]))
        return ensemble


    def sample_from_cells3(self, pts, cell_idx, func, num_pts=1):
        """
        Description:
            Samples points from a list of Voronoi cells w.r.t minimum distance
        Args:
            pts: points to be replaced
            cells: list of indices to Voronoi cells from which points are to be sampled
            num_pts: number of points to be sampled from each cell, default = 1

        Returns:
            list of randomly sampled points
        """
        ensemble = np.zeros((len(cell_idx) * num_pts, self.dim), dtype='float64')
        for i, cell_id in enumerate(cell_idx):
            allot = getattr(self.db.allotments, 'cell_' + str(cell_id)).read().tolist()
            cell_pts = self.points[np.array(allot, dtype='int32').flatten()]
            weights = np.array([func(cell_pt) for cell_pt in cell_pts])
            ensemble[i] = cell_pts[np.argmax(weights)]
            print('max weight: {}, pt: {}, cell_pt: {}'.format(max(weights), pts[i], ensemble[i]))
        return ensemble


    @ut.timer
    def resample0(self, pts, weights):
        """
        Description:
            Replaces pts with attractor points. Weights decide number of offsprings, original point is kept if the number of offsprings is positive

        Args:
            pts: points to be replaced
            weights: weights for points ton decide the number of offsprings
        """
        # compute number of offsprings and sort by it
        ensemble = np.zeros((len(pts), self.dim), dtype='float64')
        offsprings = np.array([int(len(pts) * w) for w in weights], dtype=np.int32)
        off_idx = offsprings.argsort()
        offsprings = offsprings[off_idx[::-1]]
        pts = pts[off_idx[::-1]]
        discrepancy = len(pts) - sum(offsprings)
        for i in range(discrepancy):
            offsprings[i] += 1
        # find first zero in the offspring count
        fz = 0
        for i in range(len(pts)):
            if offsprings[i] == 0:
                break
            else:
                fz += 1
        # find the closest seeds for the necessary points
        cell_idx = self.closest_seeds(pts[:fz])
        print('offsprings: ', offsprings[:fz])
        # resample keeping the parent point
        start = 0
        end = 0
        for i, cell_id in enumerate(cell_idx):
            allot = getattr(self.db.allotments, 'cell_' + str(cell_id)).read().tolist()
            allot = np.array(allot, dtype='int32').flatten()
            start = end
            end += offsprings[i]
            ensemble[start] = pts[i]
            ensemble[start + 1: end] = self.points[np.random.choice(allot, size=offsprings[i] - 1)]
        return ensemble


    @ut.timer
    def resample(self, pts):
        """
        Description:
            Replaces a list of points with points sampled from their closest Voronoi cells

        Args:
            pts: the list of points to be replaced

        Returns:
            list of replacement/resampled points
        """
        cell_idx = self.closest_seeds(pts)
        return self.sample_from_cells(cell_idx)


    @ut.timer
    def resample2(self, pts):
        """
        Description:
            Replaces a list of points with points sampled from their closest Voronoi cells

        Args:
            pts: the list of points to be replaced

        Returns:
            list of replacement/resampled points
        """
        cell_idx = self.closest_seeds(pts)
        return self.sample_from_cells2(pts, cell_idx)

    @ut.timer
    def resample3(self, pts, func):
        """
        Description:
            Replaces a list of points with points sampled from their closest Voronoi cells

        Args:
            pts: the list of points to be replaced

        Returns:
            list of replacement/resampled points
        """
        cell_idx = self.closest_seeds(pts)
        return self.sample_from_cells3(pts, cell_idx, func)


    @ut.timer
    def resample4(self, num_pts, func):
        """
        Description:
            Replaces a list of points with points sampled from their closest Voronoi cells

        Args:
            pts: the list of points to be replaced

        Returns:
            list of replacement/resampled points
        """
        weights = np.array([func(p) for p in self.points])
        idx = np.argsort(weights)[-1: -num_pts-1: -1]
        #print('weights: {}'.format(weights[idx]))
        return self.points[idx], weights[idx]
