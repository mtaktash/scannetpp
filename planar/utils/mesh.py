from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def compute_triangle_planes(mesh):
    """Compute [a, b, c, d] for each triangle in the mesh."""
    mesh.compute_triangle_normals()
    mesh.orient_triangles()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.triangle_normals)

    planes = []
    for tri, normal in zip(triangles, normals):
        v0 = vertices[tri[0]]
        d = -np.dot(normal, v0)  # offset so that n @ v0 + d = 0
        planes.append(np.append(normal, d))
    return np.array(planes)


def normalize_planes(planes):
    """Normalize plane equations so that [a, b, c] is a unit vector."""
    normals = planes[:, :3]
    d = planes[:, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_normalized = normals / norms
    d_normalized = d / norms.squeeze()
    return np.hstack([normals_normalized, d_normalized[:, None]])


def cluster_planes(planes, eps=0.05, min_samples=5, cluster_size=100):
    """Cluster normalized planes using DBSCAN in 4D."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(planes)

    counts = Counter(labels)
    filtered_labels = labels.copy()
    for cluster_id, count in counts.items():
        if count < cluster_size:
            filtered_labels[filtered_labels == cluster_id] = -1

    return filtered_labels

def pick_arbitrary_vector(normal):
    normal = normal / np.linalg.norm(normal)
    # pick axis with smallest absolute component to avoid near-parallel
    idx = np.argmin(np.abs(normal))
    arbitrary = np.zeros(3)
    arbitrary[idx] = 1.0
    return arbitrary

def convex_hull_area_on_plane(points, plane):
    # Unpack plane parameters
    a, b, c, d = plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Find two orthonormal vectors in the plane (local 2D basis)
    # v1: arbitrary vector not parallel to normal
    arbitrary = pick_arbitrary_vector(normal)

    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project points onto the plane basis (u, v)
    centered_points = points - (np.dot(points, normal) + d)[..., None] * normal
    points_2d = np.stack(
        [np.dot(centered_points, u), np.dot(centered_points, v)], axis=1
    )

    hull = ConvexHull(points_2d)
    return hull.volume


def propagate_plane_labels(mesh, labels, planes, inlier_eps=0.005, fragment_thres=0.6):
    """
      1) Build mean plane per existing label
      2) Filter fragmented labels (drop them)
      3) Remap kept labels to [0..K-1]
      4) Propagate only to the nearest *kept* plane within inlier_eps
    Returns:
      new_labels (per-triangle, remapped) and kept_planes (K x 4).
    """
    labels = np.asarray(labels).copy()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # --- 0) Early out if no seeds
    seed_mask = labels != -1
    if not np.any(seed_mask):
        return labels

    # --- 1) Mean plane per label (seed set)
    unique_labels = np.unique(labels[seed_mask])

    unique_planes = []
    for lbl in unique_labels:
        unique_planes.append(planes[labels == lbl].mean(axis=0))

    unique_planes = np.asarray(unique_planes)
    unique_planes = normalize_planes(unique_planes)  # assumes provided elsewhere

    # --- 2) Add plane inliers to the labels
    for tid, tri in enumerate(triangles):
        if labels[tid] != -1:
            continue

        tri_vertices = vertices[tri]

        # Find out if this triangle is close to any of the unique planes
        distances = np.abs(
            np.dot(tri_vertices, unique_planes[:, :3].T) + unique_planes[:, 3]
        )
        distances = np.mean(distances, axis=0)

        # Assign the label of the closest plane
        closest_plane_idx = np.argmin(distances)
        if distances[closest_plane_idx] < inlier_eps:  # threshold for inlier
            labels[tid] = unique_labels[closest_plane_idx]

    # --- 3) Filter fragmented planes
    kept_labels = []
    for lbl, plane in zip(unique_labels, unique_planes):
        tri_idx = np.where(labels == lbl)[0]
        if tri_idx.size == 0:
            continue

        label_tris = triangles[tri_idx]  # (n, 3)
        label_verts = vertices[label_tris]  # (n, 3, 3)

        # Convex hull area on plane (guard against degenerate)
        hull_area = convex_hull_area_on_plane(label_verts.reshape(-1, 3), plane)
        if hull_area <= 1e-6 or not np.isfinite(hull_area):
            # drop degenerate clusters
            labels[tri_idx] = -1
            continue

        # Sum of triangle areas
        e1 = label_verts[:, 1] - label_verts[:, 0]
        e2 = label_verts[:, 2] - label_verts[:, 0]
        tri_areas = np.linalg.norm(np.cross(e1, e2), axis=1)
        covered_ratio = tri_areas.sum() / hull_area

        if covered_ratio >= fragment_thres:
            kept_labels.append(lbl)
        else:
            labels[tri_idx] = -1  # drop fragmented label

    return labels


def assign_vertex_labels_from_triangle_labels(mesh, triangle_labels, triangle_planes):
    n_vertices = len(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vertex_votes = [[] for _ in range(n_vertices)]
    for tid, tri in enumerate(triangles):
        lbl = int(triangle_labels[tid])
        for vi in tri:
            vertex_votes[vi].append(lbl)

    # Majority vote per vertex
    vertex_labels = np.full(n_vertices, -1, dtype=int)
    for vi, votes in enumerate(vertex_votes):
        if votes:
            vertex_labels[vi] = Counter(votes).most_common(1)[0][0]

    # Remap kept labels to [0..K-1]
    kept_labels = np.unique(vertex_labels[vertex_labels != -1])

    if len(kept_labels) == 0:
        return vertex_labels, np.zeros((0, 4), dtype=np.float32)

    # Gather plane equations for kept labels
    planes = []
    for old in kept_labels:
        planes.append(triangle_planes[triangle_labels == old].mean(axis=0))

    planes = np.array(planes)
    planes = normalize_planes(planes)

    return vertex_labels, planes


def planar_segmentation(mesh, cluster_eps=0.005, inlier_eps=0.005, fragment_thres=0.6):
    triangle_planes = compute_triangle_planes(mesh)
    triangle_planes = normalize_planes(triangle_planes)
    triangle_labels = cluster_planes(
        triangle_planes,
        eps=cluster_eps,
        min_samples=5,
        cluster_size=100,
    )
    propagated_triangle_labels = propagate_plane_labels(
        mesh,
        triangle_labels,
        triangle_planes,
        inlier_eps=inlier_eps,
        fragment_thres=fragment_thres,
    )
    vertex_labels, planes = assign_vertex_labels_from_triangle_labels(
        mesh, propagated_triangle_labels, triangle_planes
    )
    return mesh, vertex_labels, planes
