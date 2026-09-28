'''
Geometry, lattice and box helpers used to build and inspect configurations.

Covers lattice/volume generation (``fcc_lattice``, ``partition_cuboid_volume``,
``partition_cubic_volume_oriented_rectangles``,
``make_centered_rand_orient_point_array``);
a from-scratch cell-list neighbour search (``get_neighbours``,
``get_neighbours_cross_lattice``, ``calculate_pair_distances``,
``get_cross_lattice_nonintersecting_volumes``);
vector math (``align_vectors``, ``get_perpendicular``, ``normalize_vectors``,
``generate_random_unit_vectors``, ``random_nested_3d_vectors_like``,
``get_orientation_vec``, ``fold_coords``, ``min_img_dist``);
and the box helpers (``add_box_constraints_func``/``remove_box_constraints_func``,
``check_free_cuboid``, ``require_min_global_cut``).
'''
import logging
import warnings
from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike

import espressomd.constraints
import espressomd.shapes

from pressomancy.infra import RoutineWithArgs

#: Ratio between the WCA/Lennard-Jones contact distance and sigma: the minimum
#: of the LJ well. Objects are parameterised by their contact
#: ("size") diameter, so the sigma that reproduces a wanted size is
#: ``size / WCA_CONTACT_FACTOR``.
WCA_CONTACT_FACTOR = 2 ** (1 / 6)

#: search, in bytes.
#: 8 MiB, tuned against dense elastomers where the previous unbounded buffers
#: exhausted memory. Raising it trades memory for fewer chunk iterations;
#: results are unaffected either way.
DEFAULT_MAX_CANDIDATE_BYTES = 8 << 20

#: How many times partition_cuboid_volume redraws a volume's contents before
#: giving up on placing it without overlapping its neighbours.
MAX_PLACEMENT_ATTEMPTS = 1000

def load_coord_file(file_path):
    '''
    load the comma-separated x,y,z coordinates of a text file, one point per line, as an (N, 3) array.
    '''
    return np.loadtxt(file_path, delimiter=',', ndmin=2)

def fold_coords(points, box_dim):
    """
    Wraps points back into the primary periodic box.

    :param points: array_like | coordinates, shape (..., ndim)
    :param box_dim: array_like or float | box lengths, shape (ndim,) or scalar
    :return: np.ndarray | points folded into [0, box_dim) along each axis
    """
    box_dim = _as_box(box_dim)
    return np.mod(points, box_dim)

def min_img_dist(source, target, box_dim):
    """
    Compute the minimum image displacement from source to target under periodic boundary conditions.

    Despite the name this returns *vectors*, not scalars: target - source reduced to the nearest
    periodic image. Take np.linalg.norm(..., axis=-1) for the distance.

    Parameters
    ----------
    source : iterable of float, shape (..., ndim) or float
        Source points.
    target : iterable of float, shape (..., ndim) or float
        Target points.
    box_dim : interable of float, shape (ndim,) or float
        e.g. Cuboid box dimensions [Lx, Ly, Lz].

    Returns
    -------
    np.ndarray
        Minimum image displacement vectors.
    """
    box_dim = np.asarray(box_dim)
    source = np.asarray(source); target = np.asarray(target)
    # Ensure consistent dimensions
    if box_dim.ndim > 0 and (source.shape[-1] != target.shape[-1] or source.shape[-1] != box_dim.shape[-1]):
        raise ValueError("Last dimension of source, target, and box_dim must match")
    distance = target - source
    box_half = box_dim*0.5
    return np.remainder(distance + box_half, box_dim) - box_half

def generate_random_unit_vectors(N_PART, rng=None):
    """
    Draws unit vectors uniformly distributed on the unit sphere.

    Samples z uniformly on [-1, 1] and the azimuth uniformly on [0, 2*pi).
    That is area-uniform on the sphere (and not merely uniform in the polar
    angle) because the sphere's projection onto the enclosing cylinder
    preserves area, so equal slabs in z carry equal area.

    :param N_PART: int | number of vectors to draw
    :param rng: np.random.Generator (=None) | source of randomness. Defaults to
        the legacy global np.random state, which is what Simulation.set_sys seeds.
        Pass a Generator for a reproducible draw independent of that global state.
    :return: np.ndarray | shape (N_PART, 3), unit vectors
    """
    source = np.random if rng is None else rng
    z = source.uniform(-1, 1, N_PART)
    r = np.sqrt(1 - z*z)
    phi = source.uniform(0, 2*np.pi, N_PART)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z))

def normalize_vectors(vectors, axis=-1):
    """
    Normalizes one or many vectors to unit length.

    Zero-norm entries are left unnormalized (their norm is treated as 1)
    rather than raising a division-by-zero error.

    :param vectors: array_like | a single vector or an array of vectors
    :param axis: int (=-1) | axis along which the norm is taken
    :return: np.ndarray | same shape as ``vectors``, normalized along ``axis``
    """
    array_of_vectors= np.asarray(vectors)
    norms_array = np.atleast_1d(np.linalg.norm(array_of_vectors, axis=axis))
    norms_array[norms_array==0] = 1
    if len(array_of_vectors.shape) > 1: # if multiple vectors return an array of shape (number_of_vectors, dims)
        return array_of_vectors / np.expand_dims(norms_array, axis)
    else: # if only one vector return an array of shape (dims,)
        return array_of_vectors / norms_array

def random_nested_3d_vectors_like(item, rng=None):
    """
    Recursively generate random 3D unit vectors,
    matching the shape of nested lists, tuples, or arrays.
    - Lists/tuples of len==3 with all numbers => treated as vector and replaced by a random unit vector.
    - NumPy arrays with last dimension == 3 => generate array of random unit vectors with same shape.
    - Otherwise recurse.
    Raises error if a scalar or other unsupported leaf is found.

    :param rng: np.random.Generator (=None) | source of randomness, threaded down
        into every leaf draw. Defaults to a fresh default_rng().
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(item, (list, tuple)):
        # Check if it is a 3D vector (leaf)
        if len(item) == 3 and all(isinstance(x, (float, int)) for x in item):
            return generate_random_unit_vectors(1, rng=rng).flatten().tolist()
        else:
            return [random_nested_3d_vectors_like(sub, rng) for sub in item]
    elif isinstance(item, np.ndarray):
        if item.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 for 3D vectors, got shape {item.shape}")

        n_vectors = np.prod(item.shape[:-1])
        vectors = generate_random_unit_vectors(n_vectors, rng=rng)
        vectors = vectors.reshape(item.shape)
        # Just to be sure normalize (your function is safe)
        return normalize_vectors(vectors, axis=-1)
    else:
        raise ValueError(f"Expected last dimension to be 3 for 3D vectors, got {item}")

def get_neighbours(points: np.ndarray, box_dim: ArrayLike, cutoff: float = 1., sort: bool = True,
                   max_bytes: int = DEFAULT_MAX_CANDIDATE_BYTES, cells_per_cutoff: int | None = None):
    """Symmetric neighbour lists within one lattice under PBC. Numpy only.

    Every index 0..N-1 is a key, isolated particles map to [], j is in
    result[i] if and only if i is in result[j], and no particle is its own
    neighbour.

    Parameters
    ----------
    sort : bool
        True gives ascending neighbour lists; False leaves the order
        unspecified but deterministic, and is slightly cheaper.
    max_bytes : int
        Soft cap on the transient candidate buffers, default
        DEFAULT_MAX_CANDIDATE_BYTES (8 MiB). Lower it on a memory-constrained
        machine, raise it to trade memory for fewer chunk iterations; results
        are unaffected either way.
    cells_per_cutoff : int or None
        Cells per cutoff length along each axis. None (default) auto-selects by
        minimising a cost model over k = 1, 2, 3, 4, 6, 8; pass an integer only
        to override that choice.
    """
    box_dim = _as_box(box_dim)
    pts = np.ascontiguousarray(points)
    cutoff_limit = 0.5 * box_dim.min()
    if cutoff > cutoff_limit:
        raise ValueError(
            f"cutoff {cutoff:g} exceeds half the smallest box side ({cutoff_limit:g}). "
            "The minimum image convention breaks down above L/2: a particle "
            "would be counted as its own periodic neighbour."
        )
    # simple cases (avoid wrappign and building cell)
    n_points = pts.shape[0]
    if n_points == 0:
        return {}
    if n_points == 1:
        return {0: []}
    wrapped = np.ascontiguousarray(fold_coords(pts, box_dim))
    i, j = _pairs_from_cells(
        wrapped,
        wrapped,
        box_dim,
        cutoff,
        exclude_self=True,
        half=True,
        max_bytes=max_bytes,
        cells_per_cutoff=cells_per_cutoff,
    )
    # Mirror the half list into the symmetric form.
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    return _pairs_rows_cols_to_dict(rows, cols, n_points, sort=sort)

def get_neighbours_cross_lattice(points_a: np.ndarray, points_b: np.ndarray, box_dim: ArrayLike,
        cutoff: float = 1.0, sort: bool = True,
        max_bytes: int = DEFAULT_MAX_CANDIDATE_BYTES, cells_per_cutoff: int | None = None
    ):
    """Neighbours of each points_a point among the points_b points.

    Keys index points_a, values index points_b. Nothing is excluded: a
    coincident pair is reported at distance zero. max_bytes and
    cells_per_cutoff behave as in get_neighbours.
    """
    box_dim = _as_box(box_dim)
    pts_a = np.ascontiguousarray(points_a)
    pts_b = np.ascontiguousarray(points_b)
    cutoff_limit = 0.5 * box_dim.min()
    if cutoff > cutoff_limit:
        raise ValueError(
            f"cutoff {cutoff:g} exceeds half the smallest box side ({cutoff_limit:g}). "
            "The minimum image convention breaks down above L/2: a particle "
            "would be counted as its own periodic neighbour."
        )
    n_a = pts_a.shape[0]
    n_b = pts_b.shape[0]
    if n_a == 0:
        return {}
    if n_b == 0:
        return {i: [] for i in range(n_a)}
    i, j = _pairs_from_cells(
        np.ascontiguousarray(fold_coords(pts_a, box_dim)),
        np.ascontiguousarray(fold_coords(pts_b, box_dim)),
        box_dim,
        cutoff,
        exclude_self=False,
        half=False,
        max_bytes=max_bytes,
        cells_per_cutoff=cells_per_cutoff,
    )
    # _pairs_from_cells emits i in ascending order, so grouping is already done.
    return _pairs_rows_cols_to_dict(i, j, n_a, sort=sort)

def calculate_pair_distances(points_a, points_b, box_dim):
    """
    Calculate the pairwise distances between two sets of points under periodic boundary conditions.

    Parameters
    ----------
    points_a : np.array of shape (N, 3)
        An array of points where N is the number of points in the first set.
    points_b : np.array of shape (M, 3)
        An array of points where M is the number of points in the second set.
    box_dim : array-like of shape (3,)
        The side lengths of the periodic box.

    Returns
    -------
    distances : np.ndarray, shape (N*M,)
        1D array of distances between all pairs (a_i, b_j). dist(i,j) = distances[i*M + j]
    """
    box_dim = _as_box(box_dim)
    # Ensure inputs are numpy arrays
    points_a = np.atleast_2d(points_a)
    points_b = np.atleast_2d(points_b)
    # Pair up every `a` with every `b` by broadcasting rather than materialising an
    # explicit N*M index list; the row-major ordering (a-major, b-minor) is the
    # same either way, so dist(i,j) is still distances[i*M + j].
    displacements = min_img_dist(points_a[:, None, :], points_b[None, :, :],
                                 box_dim=box_dim)
    return np.linalg.norm(displacements, axis=-1).ravel()

def fcc_lattice(radius: float, box_dim, scaling_factor: float = 1.0,
                max_points_per_side: int = 100, mode: str = "pack") -> np.ndarray:
    """
    Generates a face-centered cubic (FCC) lattice of points within a cuboid volume.

    The function creates an FCC crystal structure where spheres of given radius are
    arranged such that they touch along the face diagonal of the unit lattice
    (conventional lattice constant a = 2*sqrt(2)*r).

    Parameters
    ----------
    radius : float
        Radius of the spheres in the lattice.
    box_dim : iterable of float of size 3
        Length of the cuboid volume's sides.
    scaling_factor : float, optional
        Factor to scale the radius of the spheres. Default is 1.0.
    max_points_per_side : int, optional
        Maximum number of points allowed per dimension. Default is 100.
        If exceeded, lattice constant is increased.
    mode : {'pack', 'crystal'}
        'pack' (default) — densest non-overlapping arrangement at the touching
        pitch sqrt(2)*r. Locally FCC with coordination 12, but the leftover
        collects as a void at the periodic seam, so the lattice does not
        continue into its own image.
        'crystal' — the pitch is relaxed per axis until the lattice tiles the
        box exactly, giving a defect-free periodic FCC with no seam, at the
        cost of neighbours sitting slightly beyond 2r.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the coordinates of the lattice points,
        where N is the number of points in the FCC lattice.

    Notes
    -----
    - The half-step (simple-cubic sub-lattice pitch) is p = sqrt(2)*r and the
      conventional constant is a = 2*p. Sites are the (i+j+k) even subset,
      generated here as whole cells times the 4-point basis.
    - mode='crystal' always resolves to an even site count, while
      mode='pack' does not guarantee this.
        Sites are the (i+j+k) even subset of a simple-cubic grid of pitch p.
        An even number of layers per axis guarantees a site and the periodic
        image of its wrap partner differ in parity, putting them at least
        sqrt(2)*p = 2r apart. An odd count is kept only when the seam gap is
        itself >= 2r.
    """
    box_dim = _as_box(box_dim, ndim=3)
    if radius <= 0 or scaling_factor <= 0:
        raise ValueError("radius and scaling_factor must be positive")
    stretch = mode == "crystal"
    if mode not in ("pack", "crystal"):
        raise ValueError(f"mode must be 'pack' or 'crystal', got {mode!r}")

    radius_scaled = radius * scaling_factor
    half_lattice_constant = np.sqrt(2) * radius_scaled
    step = max(half_lattice_constant, (box_dim / max_points_per_side).max())

    radius_fit_per_side = np.floor(box_dim / step + 1e-9).astype(int)
    if np.any(radius_fit_per_side < 2):
        raise ValueError(f"box {box_dim} too small for spheres of radius {radius_scaled}")
    n_half = radius_fit_per_side
    if not stretch:
        seam = box_dim - (n_half - 1) * step
        odd_unsafe = (n_half % 2 == 1) & (seam < 2 * radius_scaled - 1e-9)
        n_half = np.where(odd_unsafe, n_half - 1, n_half)
    else:
        n_half -= n_half % 2
        step = box_dim / n_half
    cells = np.meshgrid(*[np.arange(n) for n in n_half], indexing="ij")
    mask = (cells[0] + cells[1] + cells[2]) % 2 == 0
    points = np.column_stack([g[mask] for g in cells]) * step

    return points

def make_centered_rand_orient_point_array(center=np.array([0,0,0]), sphere_radius=1., num_monomers=1, spacing=None, box_dim=None):
    """
    Creates an array of points centered at a given position with random orientation.This function generates a linear array of points in 3D space, centered at a specified position with random orientation. It also returns the normalized orientation vector of the array.

    Parameters
    ----------
    center : numpy.ndarray, default=np.array([0,0,0])
        The center point of the array in 3D space (x,y,z coordinates)
    sphere_radius : float, default=1.0
        The radius of the sphere containing the points
    num_monomers : int, default=1
        The number of points to generate
    spacing : float, optional
        If provided, sets fixed spacing between points. The total chain length will be spacing * (num_monomers - 1), and the points will be centered around center.
    box_dim : Unused
        Accepted for `build_function` signature compatibility.
    Returns
    -------
    tuple
        A tuple containing:
        - orientation_vector (numpy.ndarray): Normalized vector indicating array orientation
        - points (numpy.ndarray): Array of 3D coordinates for each point
    Notes
    -----
    When spacing is provided, the positions along the line are given by:

        positions = spacing * (np.arange(num_monomers) - (num_monomers - 1)/2)

    ensuring that the distance between consecutive points is exactly 'spacing' and that the center of mass is at 0.
    The points are then rotated by a random orientation and shifted by 'center'.

    The orientation is drawn uniformly on the unit sphere: the azimuth theta is
    uniform on [0, 2*pi) and cos(phi) is uniform on [-1, 1].
    """

    if spacing is not None:
        positions = spacing * (np.arange(num_monomers) - (num_monomers - 1) / 2)
    else:
        shift = sphere_radius / num_monomers
        positions = np.linspace(-sphere_radius,
                        sphere_radius, num_monomers + 1)[:-1] + shift
    theta = np.random.uniform(0, 2 * np.pi)
    cos_phi = np.random.uniform(-1, 1)
    sin_phi = np.sqrt(1 - cos_phi * cos_phi)
    x_points = center[0] + positions * sin_phi * np.cos(theta)
    y_points = center[1] + positions * sin_phi * np.sin(theta)
    z_points = center[2] + positions * cos_phi
    points = np.column_stack((x_points, y_points, z_points))
    direction_vector=points[-1]-points[0]
    orientation_vector = direction_vector / np.linalg.norm(direction_vector)
    orientation_vectors = np.broadcast_to(orientation_vector, points.shape).copy()
    return orientation_vectors,points

def partition_cuboid_volume(box_dim, num_spheres, sphere_diameter, routine_per_volume=RoutineWithArgs(), flag='rand'):
    """
    Partitions a cuboid volume into spherical regions and generates points within them.
    This function creates a face-centered cubic (FCC) lattice of spheres within a cuboid volume and optionally
    generates points within each sphere according to a specified routine.

    Parameters
    ----------
    box_dim : array-like of shape (3,)
        The side lengths of the cuboid volume.
    num_spheres : int
        The desired number of spherical regions to create.
    sphere_diameter : float
        The diameter of each spherical region.
    routine_per_volume : RoutineWithArgs, optional
        A callable object that generates points within each sphere. Default is empty RoutineWithArgs.
    flag : str, optional
        Determines the arrangement of sphere centers. 'rand' for random shuffling. Default is 'rand'.

    Returns
    -------
    sphere_centers : np.ndarray, shape (num_spheres, 3)
        Centers of the chosen lattice sites.
    positions : np.ndarray
        Shape (num_spheres, num_monomers, 3) when the routine generates more than one monomer per
        volume, otherwise (num_spheres, 3) -- the centers themselves.
    orientations : np.ndarray
        Same leading shape as `positions`; the orientation vector for each volume's contents.

    Raises
    ------
    ValueError
        If the box cannot hold `num_spheres` sites even at the minimum packing scale, or if a
        volume's contents cannot be placed without overlap within MAX_PLACEMENT_ATTEMPTS draws.
    """
    box_dim =  _as_box(box_dim)
    sphere_radius = sphere_diameter * 0.5
    scaling = 1.0

    # Adjust scaling until we have enough sphere centers
    scaling_floor = 0.85
    while True:
        sphere_centers = fcc_lattice(radius=sphere_radius, box_dim=box_dim, scaling_factor=scaling, mode="pack")
        volumes_to_fill=len(sphere_centers)
        logging.info('num_spheres_needed, num_spheres_got: %s', (num_spheres, volumes_to_fill))
        if  volumes_to_fill>= num_spheres:
            break
        if scaling <= scaling_floor:
            raise ValueError(
                f"Cannot fit {num_spheres} spheres of diameter {sphere_diameter} into a box of "
                f"{box_dim}: only {volumes_to_fill} lattice sites exist at the minimum "
                f"packing scale ({scaling_floor}). Reduce num_spheres or sphere_diameter, "
                f"or enlarge the box.")
        scaling = max(scaling - 0.1, scaling_floor)
    logging.info('scaling used: %s', scaling)

    # Center point distribution in box
    min_centers = np.min(sphere_centers, axis=0)
    max_centers = np.max(sphere_centers, axis=0)
    sphere_centers += box_dim/2 - (min_centers + max_centers)/2

    # Randomly shuffle the available centers and select the required number of centers
    take_index = np.arange(len(sphere_centers))
    if flag=='rand':
        np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    sphere_centers=sphere_centers[take_index]
    # Initialize an array to store the generated points inside each spherical region
    results = [None] * num_spheres
    res_orientations = [None] * num_spheres
    # Perform the point generation routine if `num_monomers` not 0
    if routine_per_volume.num_monomers>1:
        grouped_positions = defaultdict(list)
        #grouped_volumes is a dictionary that contains all neighouring lattice sites sphere_diameter
        grouped_volumes=get_neighbours(sphere_centers,box_dim=box_dim,cutoff=sphere_diameter)
        for i, center in enumerate(sphere_centers):
            valid_placement = False
            for attempt in range(MAX_PLACEMENT_ATTEMPTS):
                orientations, points = routine_per_volume(
                    center=center, num_monomers=routine_per_volume.num_monomers, sphere_radius=sphere_radius, spacing=routine_per_volume.spacing,
                    box_dim=box_dim)
                should_proceed = True

                # Check for overlaps with points in neighboring spheres
                for volume_id in grouped_volumes[i]:
                    if grouped_positions[volume_id]:
                        distances = calculate_pair_distances(points, grouped_positions[volume_id], box_dim=box_dim)
                        if np.any(distances <= routine_per_volume.monomer_size):
                            should_proceed = False
                            break

                if should_proceed:
                    grouped_positions[i].extend(points)
                    results[i] = points
                    res_orientations[i] = orientations
                    valid_placement = True
                    break
            if not valid_placement:
                raise ValueError(
                    f"Could not place the contents of volume {i} of {len(sphere_centers)} "
                    f"(center {center}) without overlapping its neighbours after "
                    f"{MAX_PLACEMENT_ATTEMPTS} random attempts. The lattice is too dense for "
                    f"{routine_per_volume.num_monomers} monomers of size "
                    f"{routine_per_volume.monomer_size} per volume of diameter {sphere_diameter}. "
                    f"Reduce num_spheres, monomer size or monomer count, or enlarge the box.")
    else:
        results=sphere_centers
        res_orientations=generate_random_unit_vectors(len(sphere_centers))
    return sphere_centers, np.asarray(results), np.asarray(res_orientations)

def partition_cubic_volume_oriented_rectangles(box_dim, num_spheres, small_box_dim, num_monomers):
    """
    Partition a cubic volume into smaller rectangular regions and generate oriented points within each region.

    This function divides a larger cubic box into smaller rectangular volumes based on the dimensions of the
    smaller boxes provided. It then generates a specified number of points within each smaller volume, ensuring
    they are oriented along a random direction.

    Parameters
    ----------
    box_dim : array-like of shape (3,)
        Dimensions of the larger cubic box (lengths along x, y, and z axes).
    num_spheres : int
        Number of smaller rectangular volumes to generate within the larger box.
    small_box_dim : array-like of shape (3,)
        Dimensions of the smaller boxes (lengths along x, y, and z axes).
    num_monomers : int
        Number of points to generate within each smaller box.

    Returns
    -------
    sphere_centers : ndarray of shape (num_spheres, 3)
        Coordinates of the centers of the selected rectangular volumes.
    result : ndarray of shape (num_spheres, num_monomers, 3)
        Generated points within each rectangular volume, oriented along a random direction.

    Raises
    ------
    ValueError
        If the number of available rectangular volumes is less than `num_spheres`.

    Notes
    -----
    - The function uses the dimensions of `small_box_dim` to determine the number of partitions along each axis.
    - When there are fewer partitions along an axis (e.g., one partition), alternate boxes along that axis are
      adjusted to ensure even distribution.
    - The generated points within each smaller box are spaced along a single direction determined by a random angle.

    Examples
    --------
    Partition a 10x10x10 box into smaller 2x2x2 volumes and generate 5 points in each volume:
    >>> box_dim = np.array([10.0, 10.0, 10.0])
    >>> small_box_dim = np.array([2.0, 2.0, 2.0])
    >>> num_spheres = 10
    >>> num_monomers = 5
    >>> centers, points = partition_cubic_volume_oriented_rectangles(box_dim, num_spheres, small_box_dim, num_monomers)
    """
    box_dim = _as_box(box_dim)
    small_box_dim = _as_box(small_box_dim)
    _, _, sphere_diameter = small_box_dim
    sphere_radius = sphere_diameter * 0.5

    x_partitions, y_partitions, z_partitions = (
        box_dim // small_box_dim).astype(int)

    x_len, y_len, z_len = small_box_dim
    x_coords = np.linspace(
        0.5 * x_len, box_dim[0] - 0.5 * x_len, x_partitions)
    y_coords = np.linspace(
        0.5 * y_len, box_dim[1] - 0.5 * y_len, y_partitions)
    z_coords = np.linspace(
        0.5 * z_len, box_dim[2] - 0.5 * z_len, z_partitions)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    sphere_centers = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Adjust coordinates for partitions equal to 1
    if x_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 0] = box_dim[0] - 0.5 * x_len

    if y_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 1] = box_dim[1] - 0.5 * y_len

    if z_partitions == 1:
        for i in range(1, len(sphere_centers), 2):
            sphere_centers[i, 2] = box_dim[2] - 0.5 * z_len

    if not (len(sphere_centers) >= num_spheres):
        raise ValueError('Must be enough possible volumes. Introduce a scaling factor.')

    take_index = np.arange(len(sphere_centers))
    np.random.shuffle(take_index)
    take_index = take_index[:num_spheres]
    shift = sphere_radius / num_monomers
    alphas = np.linspace(-sphere_radius,
                         sphere_radius, num_monomers + 1)[:-1] + shift
    result = np.empty((num_spheres, num_monomers, 3))
    for i, iid in enumerate(take_index):
        center = sphere_centers[iid]
        theta = np.random.uniform(0, 2 * np.pi)
        cos_phi = np.random.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi * cos_phi)
        x_points = center[0] + alphas * sin_phi * np.cos(theta)
        y_points = center[1] + alphas * sin_phi * np.sin(theta)
        z_points = center[2] + alphas * cos_phi
        result[i] = np.column_stack((x_points, y_points, z_points))

    return sphere_centers[take_index], result

def get_orientation_vec(pos):
    '''
    Calculates the principal gyration axis of a filament as the orientation of a filament.

    The gyration tensor is real symmetric by construction, so np.linalg.eigh is
    used: it is guaranteed to return real eigenvalues and eigenvectors, which is
    what espresso needs.

    An eigenvector is only defined up to sign, so the returned axis is pinned to
    point from the first to the last position. Callers rely on this: the vector
    decides which end of a filament carries the 'front' virtual sites, so an
    arbitrary sign would make that assignment a coin flip.

    :param pos: array_like, shape (N, 3) | positions, in order along the object
    :return: np.ndarray, shape (3,) | normalised principal gyration axis
    '''
    dip_3d = np.asarray(pos, dtype=float)
    deviations = dip_3d - dip_3d.mean(axis=0)
    gyration_tensor_element = (deviations.T @ deviations) / len(dip_3d)
    # r_cm = np.mean(dip_3d, axis=0)
    # gyration_tensor_xx = np.mean(
    #     [(x-r_cm[0])*(x-r_cm[0]) for (x, y, z) in dip_3d])
    # gyration_tensor_yy = np.mean(
    #     [(y-r_cm[1])*(y-r_cm[1]) for (x, y, z) in dip_3d])
    # gyration_tensor_zz = np.mean(
    #     [(z-r_cm[2])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    # gyration_tensor_xy = np.mean(
    #     [(x-r_cm[0])*(y-r_cm[1]) for (x, y, z) in dip_3d])
    # gyration_tensor_xz = np.mean(
    #     [(x-r_cm[0])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    # gyration_tensor_yz = np.mean(
    #     [(y-r_cm[1])*(z-r_cm[2]) for (x, y, z) in dip_3d])
    # gyration_tensor_element = [[gyration_tensor_xx, gyration_tensor_xy, gyration_tensor_xz],
    #                            [gyration_tensor_xy, gyration_tensor_yy,
    #                                gyration_tensor_yz],
    #                            [gyration_tensor_xz, gyration_tensor_yz, gyration_tensor_zz]]
    
    res, egiv = np.linalg.eigh(gyration_tensor_element)
    pr_comp = egiv[:, np.argmax(res)]
    pr_comp /= np.linalg.norm(pr_comp)
    # Pin the otherwise arbitrary eigenvector sign to the first->last direction.
    if np.dot(pr_comp, dip_3d[-1] - dip_3d[0]) < 0:
        pr_comp = -pr_comp
    return pr_comp

def get_cross_lattice_nonintersecting_volumes(current_lattice_centers, current_lattice_diam,
                                              other_lattice_centers, other_lattice_diam, box_dim):
    """
    Calculate non-intersecting volumes between two different lattices. This function determines which volumes from one lattice do not intersect with volumes from another lattice,
    considering periodic boundary conditions.

    Parameters
    ----------
    current_lattice_centers : array-like
        Centers of volumes in the first lattice.
    current_lattice_diam : float
        Diameter of the volumes in the first lattice.
    other_lattice_centers : array-like
        Centers of volumes in the second lattice.
    other_lattice_diam : float
        Diameter of the volumes in the second lattice.
    box_dim : array-like of shape (3,)
        Side lengths of the periodic box.

    Returns
    -------
    dict
        Dictionary with volume IDs as keys and lists of boolean masks as values.
        Each mask indicates whether the volume from the first lattice intersects
        with corresponding volumes from the second lattice.

    Notes
    -----
    Volumes are paired within a cutoff of (d1 + d2)/2, where d1, d2 are the
    diameters of the volumes in the respective lattices, and a pair counts as
    non-intersecting when its centre-to-centre separation reaches that same
    distance.
    """

    box_dim = _as_box(box_dim)
    neigh=get_neighbours_cross_lattice(current_lattice_centers,other_lattice_centers,
    box_dim, cutoff=(current_lattice_diam+other_lattice_diam)*0.5)
    aranged_cross_lattice_options={}
    new_crit=(current_lattice_diam+other_lattice_diam)*0.5
    for vol_id,associated_vol_ids in neigh.items():
        mask=[]
        if associated_vol_ids:
            for as_vol_id in associated_vol_ids:
                res=calculate_pair_distances(current_lattice_centers[vol_id], other_lattice_centers[as_vol_id], box_dim=box_dim)
                mask.append(all(x >= new_crit for x in res))
        aranged_cross_lattice_options[vol_id]=mask
    return aranged_cross_lattice_options

def align_vectors(v1, v2):
    """
    Compute the rotation matrix that aligns vector v1 to vector v2.

    Args:
        v1 (numpy.ndarray): The initial vector to align.
        v2 (numpy.ndarray): The target vector to align with.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix that aligns v1 with v2.

    The function handles special cases where the vectors are already aligned or are opposite.
    It uses Rodrigues' rotation formula for general cases.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross_prod = np.cross(v1, v2)
    sin_theta = np.linalg.norm(cross_prod)
    cos_theta = np.dot(v1, v2)
    if np.isclose(cos_theta, 1.0):
        return np.eye(3)
    if np.isclose(cos_theta, -1.0):
        orthogonal_vector = np.array([1.0, 0.0, 0.0]) if not np.isclose(np.abs(v1[0]), 1.0) else np.array([0.0, 1.0, 0.0])
        orthogonal_vector -= v1 * np.dot(orthogonal_vector, v1)
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)
        return -np.eye(3) + 2 * np.outer(orthogonal_vector, orthogonal_vector)
    cross_prod_matrix = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + cross_prod_matrix +
        (np.dot(cross_prod_matrix, cross_prod_matrix) * ((1 - cos_theta) / (sin_theta ** 2)))
    )
    return rotation_matrix

def get_perpendicular(vec, phi=None):
    """
    Returns a unit vector perpendicular to ``vec``.

    A reference direction is projected off ``vec`` to get one perpendicular
    vector, which is then rotated by ``phi`` about ``vec`` (Rodrigues'
    rotation formula), so every azimuthal angle around ``vec`` is reachable.

    :param vec: array_like, shape (3,) | the axis to be perpendicular to; must be non-zero
    :param phi: float (=None) | rotation angle around ``vec``, in radians. If
        None, drawn uniformly from [0, 2*pi)
    :return: np.ndarray, shape (3,) | unit vector perpendicular to ``vec``
    :raises ValueError: if ``vec`` is (numerically) the zero vector
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        raise ValueError("input vector must be non-zero")
    unit_vec = vec / norm

    ref = np.array([1.0, 0.0, 0.0])
    if np.isclose(np.abs(np.dot(ref, unit_vec)), 1.0):
        ref = np.array([0.0, 1.0, 0.0])
    base_perp = ref - np.dot(ref, unit_vec) * unit_vec
    base_perp /= np.linalg.norm(base_perp)
    phi_val = np.random.uniform(0.0, 2.0 * np.pi) if phi is None else float(phi)
    # Rodrigues rotation of base_perp around the input axis by phi.
    perp = (
        base_perp * np.cos(phi_val)
        + np.cross(unit_vec, base_perp) * np.sin(phi_val)
        + unit_vec * np.dot(unit_vec, base_perp) * (1.0 - np.cos(phi_val))
    )
    perp /= np.linalg.norm(perp)
    return perp

def add_box_constraints_func(sys, wall_type=0, wall_epsilon=1E6, sides=['all'], inter=None, types_=None, object_types=None, bottom=None, top=None, left=None, right=None, back=None, front=None):
    """
    Adds wall constraints to the simulation box along specified sides.

    This method places flat wall constraints (using `espressomd.shapes.Wall`) perpendicular to the box axes, typically used to confine particles within the simulation domain. By default, walls are added on all six faces of the box. You can customize which walls to include or exclude, their positions, and interaction types with other particles.
    By default:
        bottom - z=0; top - z=sys.box_l[2];
        left - y=0  ; right - y=sys.box_l[1];
        back - x=0  ; front - x=sys.box_l[0];

    Parameters
    ----------
    wall_type : int, optional
        Particle type used for the wall (default: 0).
    sides : list of str, optional
        Specifies which sides to add walls on. Default is ['all'], which includes all six box faces.
        Supported values:
            - 'all': add walls on all six faces.
            - 'sides': add walls on all but the top and bottom.
            - Individual sides: 'top', 'bottom', 'left', 'right', 'front', 'back'.
            - 'no-<side>': exclude specific sides, e.g., 'no-top', 'no-right', 'no-sides'.
    inter : str or list of str, optional
        Type(s) of interaction to enable between wall and specified particle types. Currently supports:
            - 'wca': Weeks-Chandler-Andersen potential with large epsilon.
    types_ : list of int, optional
        Particle types that will interact with the walls. If None, all non-wall types in the system are used.
    object_types : list of type, optional
        Object classes whose `part_types['real']` should interact with the walls. Consulted only when
        `types_` is None; if both are None, every non-wall type in the system is used.
    bottom, top, left, right, back, front : float, optional
        Position of each wall, defined as the distance to the xOy plane (for top/bottom), xOz plane (for left/right),
        or yOz plane (for front/back). If not specified, the position defaults to the corresponding boundary of the simulation box.
        Passing any of them implicitly adds that side to `sides`.


    Returns
    -------
    list of espressomd.constraints.ShapeBasedConstraint
        List of wall constraint objects added to the system. (can be used to later specify which walls to remove).
        Organized as: bottom->top->left->right->back->front

    Notes
    -----
    - If `sides` includes any entry starting with 'no-', that side will be excluded even if 'all' or 'sides' is specified.
    - The wall interaction can be configured by specifying `inter` and, optionally, `types_`.
    - Walls are defined using outward-pointing normals and placed at specified distances from the origin.
    - The method adds constraints to `sys.constraints` directly.
    - `wall_type` must be given again to `remove_box_constraints_func` to find these walls later.
    """
    existing_part_types = set(sys.part.all().type)
    if wall_type in existing_part_types:
        raise ValueError("wall_type must be unique from all other particle types already present in the system. Default is 0.")

    sides = np.array([sides]).ravel().tolist()
    if any(side.startswith("no-") for side in sides):
        sides.append('all')

    if bottom is None:
        bottom = 0
    else:
        sides.append('bottom')
    if top is None:
        top = sys.box_l[2]
    else:
        sides.append('top')
    if left is None:
        left = 0
    else:
        sides.append('left')
    if right is None:
        right = sys.box_l[1]
    else:
        sides.append('right')
    if back is None:
        back = 0
    else:
        sides.append('back')
    if front is None:
        front = sys.box_l[0]
    else:
        sides.append('front')

    wall_constraints = []

    ###########################
    # top - bottom - const. z #
    ###########################
    if 'bottom' in sides or ('all' in sides and 'no-bottom' not in sides):
        wall = espressomd.shapes.Wall(dist=bottom, normal=[0,0,1])
        wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
        sys.constraints.add(wall_constraint)
        wall_constraints.append(wall_constraint)
    if 'top' in sides or ('all' in sides and 'no-top' not in sides):
        wall = espressomd.shapes.Wall(dist=-top, normal=[0,0,-1])
        wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
        sys.constraints.add(wall_constraint)
        wall_constraints.append(wall_constraint)
    if 'no-sides' not in sides:
        ###########################
        # left - right - const. y #
        ###########################
        if 'left' in sides or ('sides' in sides and 'no-left' not in sides) or ('all' in sides and 'no-left' not in sides):
            wall = espressomd.shapes.Wall(dist=left, normal=[0,1,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        if 'right' in sides or ('sides' in sides and 'no-right' not in sides) or ('all' in sides and 'no-right' not in sides):
            wall = espressomd.shapes.Wall(dist=-right, normal=[0,-1,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        ###########################
        # back - front - const. x #
        ###########################
        if 'back' in sides or ('sides' in sides and 'no-back' not in sides) or ('all' in sides and 'no-back' not in sides):
            wall = espressomd.shapes.Wall(dist=back, normal=[1,0,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)
        if 'front' in sides or ('sides' in sides and 'no-front' not in sides) or ('all' in sides and 'no-front' not in sides):
            wall = espressomd.shapes.Wall(dist=-front, normal=[-1,0,0])
            wall_constraint = espressomd.constraints.ShapeBasedConstraint(shape=wall, particle_type=wall_type)
            sys.constraints.add(wall_constraint)
            wall_constraints.append(wall_constraint)

    # set interactions
    if inter is not None:
        inter= np.array([inter]).ravel()

        if types_ is None:
            if object_types is None:
                types_= set([type_ for type_ in sys.part.all().type if type_ != wall_type])
            else:
                types_ = set([ele.part_types['real'] for ele in object_types])
        else:
            types_= np.array([types_]).ravel()

        if 'wca' in inter:
            for type_ in types_:
                # Dividing by WCA_CONTACT_FACTOR is what keeps particles sitting at
                # their equilibrium distance from the wall instead of a stiff overlap.
                sigma = sys.non_bonded_inter[type_,type_].wca.sigma/2 / WCA_CONTACT_FACTOR
                if sigma < 0.001:
                    warnings.warn(f"Interaction of type {type_} with wall is 0, has these particles have no interaction defined. If you would like to have no interactions between particles, but only with wall, then hange this function or do it with normal espresso constraints.")
                sys.non_bonded_inter[wall_type,type_].wca.set_params(epsilon=wall_epsilon, sigma=sigma)

    return wall_constraints

def remove_box_constraints_func(sys, wall_type=0, wall_constraints=None, part_types=None, object_types=None):
    """ Removes wall_constraints from system. Default: removes all espressomd.shapes.Wall constraints.
        If part_types is not None, remove only interactions with those particle types.
    system
    list of espressomd.constraints.ShapeBasedConstraint wall_constraints
    list of particles types to stop interactoin with box part_types
    """
    system_constraints = list(sys.constraints)
    if wall_constraints is None:
        wall_constraints = [constraint for constraint in system_constraints
                            if ( isinstance(constraint, espressomd.constraints.ShapeBasedConstraint) and isinstance(constraint.shape, espressomd.shapes.Wall)
                            and ( constraint.particle_type == wall_type or wall_type == 'all') ) ]
    else:
        wall_constraints = np.array([wall_constraints]).ravel()


    if part_types is None and object_types is None: #removes actual cosntraints (removes interactions, if no more walls of that type)
        part_types= set([type_ for type_ in sys.part.all().type])

        original_wall_types = set([constraint.particle_type for constraint in system_constraints])
        for wall in wall_constraints: #remove walls
            sys.constraints.remove(wall)
        leftover_wall_types = set([constraint.particle_type for constraint in list(sys.constraints)])
        box_types_remove = original_wall_types - leftover_wall_types
    elif part_types is None: # removes only interactions (based on objects)
        box_types_remove = set([constraint.particle_type for constraint in wall_constraints])
        object_types = np.array([object_types]).ravel()
        part_types = set([ele.part_types[typ] for ele in object_types for typ in ele.part_types])
    else: # removes only interactions (based on part_types)
        box_types_remove = set([constraint.particle_type for constraint in wall_constraints])
        part_types = np.array([part_types]).ravel()

    # remove inter for specific types
    for box_type in box_types_remove:
        for type_ in part_types:
            sys.non_bonded_inter[box_type, type_].reset()

def check_free_cuboid(sys, cuboid_l, cuboid_l_shift=None):
    """
    Checks that no existing particle lies inside a given cuboid region.

    :param sys: espressomd.System | the simulation system to inspect
    :param cuboid_l: array_like, shape (3,) | cuboid side lengths
    :param cuboid_l_shift: array_like, shape (3,) (=None) | cuboid's lower
        corner; defaults to the origin
    :return: bool | True if the cuboid is empty of particles (or the system
        has no particles at all), False if at least one particle lies inside it

    Positions are folded, so a particle that has drifted a box length does not
    read as outside the cuboid while its image sits squarely inside it.
    """
    if cuboid_l_shift is None:
        cuboid_l_shift = np.zeros((3))
    pos = sys.part.all().pos_folded
    if len(pos) == 0:
        return True
    else:
        return np.all(np.any((pos < cuboid_l_shift) | (pos > cuboid_l_shift + cuboid_l), axis=1))

def require_min_global_cut(sys, cutoff):
    """Raise unless ``sys.min_global_cut`` can hold a pair bond of length ``cutoff`` across ranks.

    A bonded partner must lie within the owning rank's ghost layer, whose width
    is ``min_global_cut``; on more than one MPI rank a longer bond cannot be
    resolved by the bond loop. Requires ``min_global_cut >= 1.5 * cutoff`` when
    ``n_nodes > 1``. A single rank has no rank boundary to cross and always passes.

    :param sys: espressomd.System | the system about to receive the bonds
    :param cutoff: float | the longest bond length (``r_0`` / catch radius) to be bonded
    :raises RuntimeError: when the cut is too small for the rank layout
    """
    n_nodes = int(sys.cell_system.get_state()['n_nodes'])
    needed = 1.5 * float(cutoff)
    if n_nodes > 1 and sys.min_global_cut < needed:
        raise RuntimeError(
            f"min_global_cut={sys.min_global_cut} is too small for bonds of length {cutoff} on "
            f"{n_nodes} MPI ranks (needs >= 1.5 x {cutoff} = {needed}). Call "
            f"set_sys(min_global_cut={needed}) before bonding, or run on a single rank.")

def _as_box(box_dim: ArrayLike, ndim: int = 3) -> np.ndarray:
    """Coerce a scalar or sequence into a strictly-positive (ndim,) float array."""
    box = np.asarray(box_dim, dtype=np.float64)
    assert box.ndim <= 1
    if box.ndim == 0:
        box = np.full(ndim, float(box))
    else:
        box = np.atleast_1d(box).ravel()
    if box.shape != (ndim,):
        raise ValueError(f"box_dim must be a scalar or shape ({ndim},), got shape {box.shape}")
    if not np.all(np.isfinite(box)):
        raise ValueError("box_dim contains non-finite values")
    if np.any(box <= 0):
        raise ValueError(f"box_dim must be strictly positive, got {box}")
    return box

# Neighbor helper functions
# A candidate costs: i(int32) + j(int32) + three float64 buffers (d, dy, dz).
_BYTES_PER_CANDIDATE = 4 + 4 + 8 * 3
def _pairs_from_cells(query_pts, target_pts, box_dim, cutoff,
    exclude_self, half,
    max_bytes, cells_per_cutoff):
    """Return (i, j) index arrays of all pairs within the cutoff.

    i indexes query_pts, j indexes target_pts. With half=True only pairs with
    j > i are returned (valid only when the two point sets are the same array).
    """
    box_dim = _as_box(box_dim)
    ndim = box_dim.shape[0]
    n_cells, cell_size, stencil = _choose_grid(
        box_dim, cutoff, target_pts.shape[0], cells_per_cutoff
    )
    n_total = int(n_cells.prod())
    
    cell_coords_t = np.floor(target_pts / cell_size).astype(np.int64)
    np.clip(cell_coords_t, 0, n_cells - 1, out=cell_coords_t)
    linear_idx_t = (cell_coords_t[:, 0] * n_cells[1] + cell_coords_t[:, 1]) * n_cells[2] + cell_coords_t[:, 2]
    
    order = np.argsort(linear_idx_t, kind="stable").astype(np.int32)
    counts = np.bincount(linear_idx_t, minlength=n_total)
    start = np.concatenate(([0], np.cumsum(counts)))

    if query_pts is target_pts:
        linear_idx_q = linear_idx_t
    else:
        cell_coords_q = np.floor(query_pts / cell_size).astype(np.int64)
        np.clip(cell_coords_q, 0, n_cells - 1, out=cell_coords_q)
        linear_idx_q = (cell_coords_q[:, 0] * n_cells[1] + cell_coords_q[:, 1]) * n_cells[2] + cell_coords_q[:, 2]
        del cell_coords_q

    del cell_coords_t

    qx, qy, qz = (np.ascontiguousarray(query_pts[:, d]) for d in range(ndim))
    tx, ty, tz = (np.ascontiguousarray(target_pts[:, d]) for d in range(ndim))

    n_q = query_pts.shape[0]
    n_stencil = stencil.shape[0]
    cutoff_sq = ( cutoff * cutoff ) * (1 + 1e-9) # some slack for numerical error
    inv_box = 1.0 / box_dim

    mean_occupancy = max(target_pts.shape[0] / max(n_total, 1), 1e-9)
    bytes_per_point = max(n_stencil * mean_occupancy * _BYTES_PER_CANDIDATE, 1.0)
    chunk = int(max(1, min(n_q, max_bytes // bytes_per_point)))

    out_i, out_j = [], []
    for lo in range(0, n_q, chunk):
        hi = min(lo + chunk, n_q)
        linear_idx_q_in_range = linear_idx_q[lo:hi]

        nz = int(n_cells[2])
        ny = int(n_cells[1])
        cz = linear_idx_q_in_range % nz
        cy = (linear_idx_q_in_range // nz) % ny
        cx = linear_idx_q_in_range // (nz * ny)
        cell_coords = np.stack([cx, cy, cz], axis=1)
        # broadcast to get stencils per cell_coords
        neighbour_coords = cell_coords[:, None, :] + stencil[None, :, :]
        # Periodic boundaries
        neighbour_coords = np.mod(neighbour_coords, n_cells)
        neighbour_ids = (neighbour_coords[:, :, 0] * ny + neighbour_coords[:, :, 1]) * nz + neighbour_coords[:, :, 2]
        del cell_coords, neighbour_coords

        grp_counts = counts[neighbour_ids].ravel()
        total_count = int(grp_counts.sum())
        if total_count == 0:
            continue
        grp_start = start[neighbour_ids].ravel()
        del neighbour_ids

        if total_count == 0:
            slots = np.empty(0, dtype=np.int64)
        else:
            output_offsets = np.concatenate(([0], np.cumsum(grp_counts)[:-1]))
            slots = np.repeat(grp_start - output_offsets, grp_counts) + np.arange(total_count, dtype=np.int64)
        j = order[slots]
        del slots, output_offsets
        i = np.repeat(
            np.repeat(np.arange(lo, hi, dtype=np.int32), n_stencil), grp_counts
        )
        del grp_counts, grp_start

        # Halve the work before touching any coordinates.
        if half:
            keep = j > i
            i, j = i[keep], j[keep]
            del keep
            if i.size == 0:
                continue
        elif exclude_self:
            keep = i != j
            i, j = i[keep], j[keep]
            del keep
            if i.size == 0:
                continue

        # Accumulate the squared distance one axis at a time
        # weird syntax is to avoid extra temporary arrays that might be large
        d2 = None
        periodic_shift = np.empty(i.size, dtype=np.float64)
        for d, (qc, tc) in enumerate(((qx, tx), (qy, ty), (qz, tz))):
            delta = tc[j]
            delta -= qc[i]
            # Periodic boundary
            np.multiply(delta, inv_box[d], out=periodic_shift)
            np.round(periodic_shift, out=periodic_shift)
            np.multiply(periodic_shift, box_dim[d], out=periodic_shift)
            delta -= periodic_shift
            delta *= delta
            if d == 0:
                d2 = delta
            else:
                d2 += delta
                del delta
        del periodic_shift

        keep = d2 <= cutoff_sq
        del d2
        if keep.any():
            out_i.append(i[keep])
            out_j.append(j[keep])

    if not out_i:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty.copy()
    return np.concatenate(out_i), np.concatenate(out_j)

def _choose_grid(box, cutoff, n_target, cells_per_cutoff=None, max_cells=10_000_000):
    """
    Pick the cell subdivision, and return the grid with its pruned stencil.
    """
    ndim = box.shape[0]
    max_cells = min(max_cells, max(8 * n_target, 64))
    best = None
    k_values = (
        [max(int(cells_per_cutoff), 1)] if cells_per_cutoff is not None
        else [1, 2, 3, 4, 6, 8]
    )
    for k in k_values:
        n = np.maximum(np.floor(box * k / cutoff).astype(np.int64), 1)
        if n.prod() > max_cells:
            continue
        size = box / n
        stencil = _stencil(n, size, cutoff, ndim)
        occupancy = n_target / float(n.prod())
        score = stencil.shape[0] * (1.0 + occupancy)
        if best is None or score < best[0]:
            best = (score, n, size, stencil)
    if best is None:  # all divisons had > max_cells cells
        n = np.maximum(np.floor(box / cutoff).astype(np.int64), 1)
        lo, hi = 1, int(n.max())
        while lo < hi:  # bisect the largest common cap c with prod(min(n, c)) <= max_cells
            c = (lo + hi + 1) // 2
            lo, hi = (c, hi) if np.minimum(n, c).prod() <= max_cells else (lo, c - 1)
        n = np.minimum(n, lo)
        size = box / n
        return n, size, _stencil(n, size, cutoff, ndim)
    return best[1], best[2], best[3]

def _stencil(n_cells, cell_size, cutoff, ndim):
    """
    Offsets to every cell that can hold a neighbour, with no repeats.
    """
    per_axis = []
    for d in range(ndim):
        n = int(n_cells[d])
        k = int(np.ceil(cutoff / cell_size[d]))
        if 2 * k + 1 >= n:
            per_axis.append(np.arange(n, dtype=np.int64))  # every distinct cell
        else:
            per_axis.append(np.arange(-k, k + 1, dtype=np.int64))
    grids = np.meshgrid(*per_axis, indexing="ij")
    offsets = np.stack([g.ravel() for g in grids], axis=1)

    gap_sq = np.zeros(offsets.shape[0])
    for d in range(ndim):
        n = int(n_cells[d])
        # Separation in cells, measured the short way round the periodic box.
        sep = np.minimum(offsets[:, d] % n, (-offsets[:, d]) % n)
        gap = np.maximum(sep - 1, 0) * cell_size[d]
        gap_sq += gap * gap

    keep = gap_sq <= cutoff * cutoff
    return np.ascontiguousarray(offsets[keep])

def _pairs_rows_cols_to_dict(rows, cols, n_keys, sort=True):
    """Group directed pairs into dict[i] -> list of j, one key per index."""
    grouped_pairs = {i: [] for i in range(n_keys)}
    if rows.size == 0:
        return grouped_pairs
    if sort:
        order = np.lexsort((cols, rows))
    else:
        order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = np.ascontiguousarray(cols[order])
    counts = np.bincount(rows, minlength=n_keys)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i in np.flatnonzero(counts):
        grouped_pairs[int(i)] = cols[offsets[i] : offsets[i + 1]].tolist()
    return grouped_pairs
