from collections import deque, Counter

import numpy as np


def planar_segmentation(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray,
    angle_thresh: float = 5.0,
    dist_thresh: float = 0.01,  # 1 cm
    region_min_size: float = 1e-2,  # 10x10 cm^2
):

    centroids = vertices[triangles].mean(axis=1)
    cos_thresh = np.cos(np.deg2rad(angle_thresh))

    # Adjacency construction (triangles sharing edges)
    adj = [[] for _ in range(len(triangles))]
    edge_to_tri = {}
    for tidx, tri in enumerate(triangles):
        for e in [(tri[i], tri[(i + 1) % 3]) for i in range(3)]:
            key = tuple(sorted(e))
            if key in edge_to_tri:
                other = edge_to_tri[key]
                adj[tidx].append(other)
                adj[other].append(tidx)
            else:
                edge_to_tri[key] = tidx

    labels = -np.ones(len(triangles), dtype=int)
    region_id = 0

    for i in range(len(triangles)):
        if labels[i] != -1:
            continue

        queue = deque([i])
        labels[i] = region_id
        region_triangles = [i]

        while queue:
            tidx = queue.popleft()

            for nidx in adj[tidx]:
                if labels[nidx] != -1:
                    continue

                # check normal angle
                cos_dist = np.abs(np.dot(normals[tidx], normals[nidx]))
                if cos_dist < cos_thresh:
                    continue

                # check centroid distance
                dist = np.abs(np.dot(normals[tidx], centroids[nidx] - centroids[tidx]))
                if dist > dist_thresh:
                    continue

                labels[nidx] = region_id
                region_triangles.append(nidx)
                queue.append(nidx)

        region_id += 1

    # Fit planes to each region
    planes = []
    plane_centroids = []
    new_id = 0
    new_labels = -np.ones_like(labels)

    for rid in range(region_id):
        region_mask = labels == rid
        pts = centroids[region_mask]

        # compute surface area of triangles in this region
        region_tri_idxs = np.nonzero(region_mask)[0]
        if len(region_tri_idxs) > 0:
            tri_verts = vertices[triangles[region_tri_idxs]]  # (n_tri, 3, 3)
            v0 = tri_verts[:, 0, :]
            v1 = tri_verts[:, 1, :]
            v2 = tri_verts[:, 2, :]
            cross = np.cross(v1 - v0, v2 - v0)
            tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
            total_area = float(tri_areas.sum())
        else:
            total_area = 0.0

        # invalidate small regions (by area size in meter squared)
        if total_area < region_min_size:
            labels[region_mask] = -1  # invalidate small regions
            continue

        # Fit plane via least squares (PCA)
        subset = np.random.choice(len(pts), size=min(1000, len(pts)), replace=False)
        pts = pts[subset]

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[-1, :]
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, centroid)

        plane = np.concatenate((normal, [d]), axis=0)
        planes.append(plane)
        plane_centroids.append(centroid)
        new_labels[region_mask] = new_id
        new_id += 1

    planes = np.array(planes)
    plane_centroids = np.array(plane_centroids)
    labels = new_labels
    assert len(planes) == len(np.unique(labels[labels != -1])) == new_id

    # Now merge planes with similar parameters
    merged_labels = -np.ones_like(labels, dtype=int)
    merged_planes = []
    visited = np.zeros(len(planes), dtype=bool)
    merge_id = 0
    for pid in range(len(planes)):
        if visited[pid]:
            continue

        visited[pid] = True
        merged_labels[labels == pid] = merge_id

        normal1 = planes[pid, :3]
        offset1 = plane_centroids[pid]

        for qid in range(pid + 1, len(planes)):
            if visited[qid]:
                continue

            normal2 = planes[qid, :3]
            offset2 = plane_centroids[qid]

            cos_dist = np.abs(np.dot(normal1, normal2))
            if cos_dist < cos_thresh:
                continue

            dist = np.abs(np.dot(offset1 - offset2, normal1))
            if dist > dist_thresh:
                continue

            visited[qid] = True
            merged_labels[labels == qid] = merge_id

        merge_id += 1
        merged_planes.append(planes[pid])

    planes = np.array(merged_planes)
    labels = merged_labels
    assert len(planes) == len(np.unique(labels[labels != -1])) == merge_id

    # Now assign planes to vertices with majority voting
    n_vertices = len(vertices)
    vertex_votes = [[] for _ in range(n_vertices)]
    for tid, tri in enumerate(triangles):
        lbl = int(labels[tid])
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
    kept_labels = set(kept_labels)
    filtered_planes = []
    for rid in range(new_id):
        if rid in kept_labels:
            filtered_planes.append(planes[rid])
    planes = np.array(filtered_planes)

    return vertex_labels, planes
