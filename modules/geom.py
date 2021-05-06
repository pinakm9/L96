import scipy.spatial as ss
import polytope as pc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import utility as ut

class VoronoiTess:

    def __init__(self, points):
        self.pts = points
        self.tess = ss.Voronoi(points)
        """1
        ss.voronoi_plot_2d(self.tess)
        plt.show()
        """
    def plot(self, ax=None, alpha=0.5, linewidth=0.7, saveas=None, show=True):
        # Configure plot
        if ax is None:
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.axis("equal")
        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        polygons = []
        """
        fig, ax1 = plt.subplots()
        ax1.scatter(*zip(*points))
        ax1.add_patch(dc.PolygonPatch(alpha_shape, alpha=2.0))
        plt.show()
        """
        # Add polygons
        for reg in self.tess.regions:
            polygon = [self.tess.vertices[i] for i in reg]
            if True:#np.all(polygon != -1.0):
                ax.fill(*zip(*polygon), alpha=0.4)

        #ax.scatter(self.pts[:, 0], self.pts[:, 1])
        if show:
            plt.show()
        if saveas is not None:
            plt.savefig(saveas)
        return ax

    def plot_polygons(self, ax=None, alpha=0.5, linewidth=0.7, saveas=None, show=True):
        # Configure plot
        if ax is None:
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.axis("equal")
        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        polygons = []
        """
        for reg in self.tess.point_region:
            polygon = self.tess.vertices[self.tess.regions[reg]]
            if -1 not in polygon:
                polygons.append(polygon)
            ax.fill(*zip(*polygon), alpha=0.4)
        print(len(polygons))
        """
        alpha_shape = al.alphashape(self.pts, 2.0)
        boundary = list(dc.PolygonPatch(alpha_shape).__dict__['_path'].__dict__['_vertices'])
        bdry_idx = [np.where(self.pts == vertex[0])[0][0] for vertex in boundary]
        """
        fig, ax1 = plt.subplots()
        ax1.scatter(*zip(*points))
        ax1.add_patch(dc.PolygonPatch(alpha_shape, alpha=2.0))
        plt.show()
        """
        # Add polygons
        regions, vertices = self.voronoi_finite_polygons_2d()
        for pt, reg in enumerate(regions):
            if pt in bdry_idx:
                continue
            poly = vertices[reg]
            colored_cell = pat.Polygon(poly, linewidth=linewidth, alpha=alpha, facecolor=ut.random_color(as_str=False, alpha=1), edgecolor="black") #
            ax.add_patch(colored_cell)

        ax.scatter(self.pts[:, 0], self.pts[:, 1])
        if saveas is not None:
            plt.savefig(saveas)
        if show:
            plt.show()
        return ax

    def voronoi_finite_polygons_2d(self, radius=None):
        """
        Reconstruct infinite self.tessonoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        self.tess : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if self.tess.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = self.tess.vertices.tolist()

        center = self.tess.points.mean(axis=0)
        if radius is None:
            radius = self.tess.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.tess.ridge_points, self.tess.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))




        # Reconstruct infinite regions
        for p1, region in enumerate(self.tess.point_region):
            vertices = self.tess.regions[region]
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = self.tess.points[p2] - self.tess.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = self.tess.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.tess.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)


"""
points = np.array([[0.24886105, 0.14452593],
       [0.6824991 , 0.70051654],
       [0.67251442, 0.66597723],
       [0.02171107, 0.4923205 ],
       [0.69987004, 0.28192133],
       [0.05316354, 0.6740963 ],
       [0.79515121, 0.31879103],
       [0.53554519, 0.4000468 ],
       [0.64824669, 0.26111343],
       [0.00733356, 0.80027481],
       [0.68405134, 0.00516987],
       [0.59998721, 0.47584221],
       [0.52414774, 0.68844192],
       [0.27724633, 0.57660563],
       [0.82377245, 0.87997439],
       [0.77131131, 0.89903237],
       [0.41370299, 0.70900939],
       [0.25161305, 0.82164982],
       [0.17503043, 0.38474403],
       [0.52406975, 0.88047603],
       [99999., -99999.],
       [-99999., 99999.],
       [99999.,99999.],
       [-99999., -99999.]
      ])


vt = VoronoiTess(points)
#vt.plot_polygons(saveas = '../images/color_vor.png')
print(vt.tess.regions)
print(len(vt.tess.regions))
print(vt.tess.point_region)
print(len(vt.tess.point_region))
"""
def pts_to_poly(points):
    hull = ss.ConvexHull(points)
    dim = np.array(points).shape[1]
    A = hull.equations[:, 0: dim]
    b = -hull.equations[:, dim]
    return pc.Polytope(A, b)
