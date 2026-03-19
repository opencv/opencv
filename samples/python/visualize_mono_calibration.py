# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""
3d calibration visualisation tool.

Loads YAMLS from calibration.cpp sample code and displays an interactive 3d plot using meshlab and matplotlib

usage:
    python3 visualize_mono_calibration.py cam1.yml
    python3 visualize_mono_calibration.py cam1.yml cam2.yml --view board --export scene

Arguments:
    calib_files       YAML files with calibration data
    --view            Reference frame: {camera, board} (default:board)
    --export NAME     Export scene as name.obj/.mtl
    --no-gui          Disable GUI (only export)


"""

import argparse
import sys
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_calib(
    filename: str,
) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open calibration file: {filename}")
    errs = (
        fs.getNode("per_view_reprojection_errors")
        .mat()
        .flatten()
        .astype(float)
        .tolist()
    )
    extr = fs.getNode("extrinsic_parameters").mat().astype(np.float64)
    # grid points may be stored either as a sequence of 3‑D points or as a matrix
    node = fs.getNode("grid_points")
    if node.isSeq():
        vals = [float(node.at(i).real()) for i in range(node.size())]
        grid = np.array(vals, dtype=np.float32).reshape(-1, 3)
    else:
        grid = node.mat().astype(np.float32).reshape(-1, 3)
    K = fs.getNode("camera_matrix").mat().astype(np.float64)
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    bw = int(fs.getNode("board_width").real())
    bh = int(fs.getNode("board_height").real())
    fs.release()
    return errs, extr, grid, K, (w, h), (bw, bh)


def invert_extrinsic(
    rvec: np.ndarray, tvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3)
    return R_inv, t_inv


def draw_frustum(
    ax,
    K: np.ndarray,
    imsize: Tuple[int, int],
    scale: float,
    pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    color: str = "cyan",
    alpha: float = 0.1,
) -> None:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = imsize
    # define a near and far plane for the pyramid
    near = scale
    far = scale * 3.0
    # compute  the 8 vertices of the frustum in camera coordinates
    pts = []
    for d in (near, far):
        for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
            x = (u - cx) / fx * d
            y = (v - cy) / fy * d
            pts.append([x, y, d])
    P = np.array(pts, dtype=np.float64)
    # apply pose transformation if provided
    if pose is not None:
        R_pose, t_pose = pose
        P = (R_pose @ P.T).T + t_pose.reshape(1, 3)
        origin = t_pose.reshape(3)
    else:
        origin = np.zeros(3)
    # define edges of the frustum
    for i in range(4):
        ax.plot(*zip(origin, P[i]), color=color, lw=1)
        ax.plot(*zip(origin, P[i + 4]), color=color, lw=1)
        ax.plot(*zip(P[i], P[(i + 1) % 4]), color=color, lw=1)
        ax.plot(*zip(P[i + 4], P[4 + (i + 1) % 4]), color=color, lw=1)
    for indices in ([0, 1, 2, 3], [4, 5, 6, 7]):
        quad = P[list(indices)]
        ax.add_collection3d(
            Poly3DCollection(
                [quad],
                facecolors=color,
                edgecolors="white",
                linewidths=0.5,
                alpha=alpha,
            )
        )


def build_board_mesh(grid: np.ndarray, board_width: int, board_height: int) -> List[List[np.ndarray]]:
    pts = grid.reshape((board_width, board_height, 3), order="F")
    faces: List[List[np.ndarray]] = []
    for ix in range(board_width - 1):
        for iy in range(board_height - 1):
            p00 = pts[ix, iy]
            p10 = pts[ix + 1, iy]
            p11 = pts[ix + 1, iy + 1]
            p01 = pts[ix, iy + 1]
            faces.append([p00, p10, p11, p01])
    return faces


def plot_scene_camera_view(
    ax,
    extr: np.ndarray,
    grid: np.ndarray,
    K: np.ndarray,
    imsize: Tuple[int, int],
    board_size: Tuple[int, int],
    errors: Optional[List[float]] = None,
    colormap: Optional[callable] = None,
) -> List[Poly3DCollection]:
    Rts: List[Tuple[np.ndarray, np.ndarray]] = []
    for row in extr:
        rvec = row[:3].reshape(3, 1)
        tvec = row[3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        Rts.append((R, tvec.reshape(3)))
    # get the plot limits based on transformed board points
    all_points = []
    for R, t in Rts:
        pts_h = (R @ grid.T).T + t.reshape(1, 3)
        all_points.append(pts_h)
    all_points_arr = np.concatenate(all_points, axis=0)
    mins = all_points_arr.min(axis=0)
    maxs = all_points_arr.max(axis=0)
    spans = maxs - mins
    margin = 0.1
    mins -= spans * margin
    maxs += spans * margin
    ax.set_xlim(mins[0], maxs[0])
    # invert the y axis for camera view so that boards appear upright
    ax.set_ylim(maxs[1], mins[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))
    cmap = colormap or plt.get_cmap("tab20")
    faces_template = build_board_mesh(grid, board_size[0], board_size[1])
    meshes: List[Poly3DCollection] = []
    for idx, (R, t) in enumerate(Rts):
        faces_transformed: List[List[np.ndarray]] = []
        for face in faces_template:
            transformed = [(R @ v.reshape(3, 1)).flatten() + t for v in face]
            faces_transformed.append(transformed)
        mesh = Poly3DCollection(
            faces_transformed, alpha=0.3, linewidths=0.2, picker=True
        )
        mesh.set_facecolor(cmap(idx % 20))
        mesh.set_edgecolor("white")
        ax.add_collection3d(mesh)
        meshes.append(mesh)
        board_centroid = np.mean(
            [(R @ v.reshape(3, 1)).flatten() + t for v in grid], axis=0
        )
        ax.text(*board_centroid, str(idx), color="white", fontsize=8)
    scale = spans.max() * 0.05
    draw_frustum(ax, K, imsize, scale=scale)
    L = scale
    ax.quiver(0, 0, 0, L, 0, 0, color="r", length=L)
    ax.quiver(0, 0, 0, 0, -L, 0, color="g", length=L)
    ax.quiver(0, 0, 0, 0, 0, L, color="b", length=L)
    ax.text(L, 0, 0, "Xc", color="r", fontsize=10)
    ax.text(0, -L, 0, "Yc", color="g", fontsize=10)
    ax.text(0, 0, L, "Zc", color="b", fontsize=10)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.set_xlabel("X (mm)", color="white")
    ax.set_ylabel("Y (mm)", color="white")
    ax.set_zlabel("Z (mm)", color="white")
    ax.tick_params(colors="white")
    # set a pleasing viewing angle
    ax.view_init(elev=20, azim=45)
    return meshes


def plot_scene_board_view(
    ax,
    calibs: List[Dict[str, object]],
    board_size: Tuple[int, int],
    colormap: Optional[callable] = None,
    max_frustums_per_cam: Optional[int] = None,
) -> None:
    # determine bounding box across all camera poses
    camera_positions = []
    board_points = calibs[0][
        "grid"
    ]  # assumes that all cameras use the same board pattern
    for calib in calibs:
        extr = calib["extr"]
        for i, row in enumerate(extr):
            # optional subsample
            if max_frustums_per_cam is not None and i >= max_frustums_per_cam:
                break
            rvec = row[:3]
            tvec = row[3:]
            R_inv, t_inv = invert_extrinsic(rvec, tvec)
            camera_positions.append(t_inv)
    camera_positions_arr = np.array(camera_positions)
    mins = np.min(np.vstack((camera_positions_arr, board_points)), axis=0)
    maxs = np.max(np.vstack((camera_positions_arr, board_points)), axis=0)
    spans = maxs - mins
    margin = 0.1
    mins -= spans * margin
    maxs += spans * margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(maxs[1], mins[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))
    cmap = colormap or plt.get_cmap("tab10")
    faces = build_board_mesh(board_points, board_size[0], board_size[1])
    board_mesh = Poly3DCollection(faces, alpha=0.3, linewidths=0.2)
    board_mesh.set_facecolor((0.2, 0.8, 1.0, 0.3))
    board_mesh.set_edgecolor("white")
    ax.add_collection3d(board_mesh)
    L = spans.max() * 0.05
    ax.quiver(0, 0, 0, L, 0, 0, color="r", length=L)
    ax.quiver(0, 0, 0, 0, L, 0, color="g", length=L)
    ax.quiver(0, 0, 0, 0, 0, L, color="b", length=L)
    ax.text(L, 0, 0, "Xb", color="r", fontsize=10)
    ax.text(0, L, 0, "Yb", color="g", fontsize=10)
    ax.text(0, 0, L, "Zb", color="b", fontsize=10)
    for cam_idx, calib in enumerate(calibs):
        extr = calib["extr"]
        K = calib["K"]
        imsize = calib["imsize"]
        label = calib.get("label", f"Cam{cam_idx}")
        colour = cmap(cam_idx % 10)
        scale = spans.max() * 0.05
        for i, row in enumerate(extr):
            if max_frustums_per_cam is not None and i >= max_frustums_per_cam:
                break
            rvec = row[:3]
            tvec = row[3:]
            R_inv, t_inv = invert_extrinsic(rvec, tvec)
            draw_frustum(
                ax, K, imsize, scale=scale, pose=(R_inv, t_inv), color=colour, alpha=0.1
            )
            # draw label at camera position for first frustum only
            if i == 0:
                ax.text(*t_inv, label, color=colour, fontsize=9)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.set_xlabel("X (board mm)", color="white")
    ax.set_ylabel("Y (board mm)", color="white")
    ax.set_zlabel("Z (board mm)", color="white")
    ax.tick_params(colors="white")
    ax.view_init(elev=20, azim=45)


def export_scene_to_obj_multi(
    base_name: str,
    calibs: List[Dict[str, object]],
    board_size: Tuple[int, int],
    view: str = "camera",
    max_frustums_per_cam: Optional[int] = None,
) -> None:
    # prepare the data structures for obj/mrl
    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []
    materials: List[Tuple[str, Tuple[float, float, float], float]] = []
    group_info: List[Tuple[str, int, int, str]] = []
    if view == "camera":
        # first calibration is used in camera view
        calib = calibs[0]
        grid = calib["grid"]
        extr = calib["extr"]
        board_mat = "board_mat"
        materials.append((board_mat, (0.2, 0.8, 1.0), 0.3))
        start_faces = len(faces)
        for row in extr:
            rvec = row[:3]
            tvec = row[3:]
            R, _ = cv2.Rodrigues(rvec)
            pts_h = (R @ grid.T).T + tvec.reshape(1, 3)
            world = pts_h.reshape((board_size[0], board_size[1], 3), order="F")
            corners = [
                world[0, 0],
                world[board_size[0] - 1, 0],
                world[board_size[0] - 1, board_size[1] - 1],
                world[0, board_size[1] - 1],
            ]
            v0 = len(vertices)
            vertices.extend([tuple(v.tolist()) for v in corners])
            faces.append([v0 + 1, v0 + 2, v0 + 3, v0 + 4])
        count = len(faces) - start_faces
        group_info.append(("all_boards", start_faces, count, board_mat))
        frustum_mat = "frustum_mat"
        materials.append((frustum_mat, (1.0, 0.4, 0.2), 1.0))
        start_faces = len(faces)
        K = calib["K"]
        imsize = calib["imsize"]
        board_extent = np.max(np.ptp(grid, axis=0))
        near = 0.05 * board_extent
        far = near * 3
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = imsize
        pts_cam: List[List[float]] = []
        for d in (near, far):
            for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
                x = (u - cx) / fx * d
                y = (v - cy) / fy * d
                pts_cam.append([x, y, d])
        pts_cam_arr = np.array(pts_cam)
        apex_idx = len(vertices)
        vertices.append((0.0, 0.0, 0.0))
        start_verts = len(vertices)
        vertices.extend([tuple(p.tolist()) for p in pts_cam_arr])
        for i in range(4):
            faces.append(
                [apex_idx + 1, start_verts + i + 1, start_verts + ((i + 1) % 4) + 1]
            )
        quad = [start_verts + i + 1 for i in range(4, 8)]
        faces.append(quad)
        count = len(faces) - start_faces
        group_info.append(("camera_frustum", start_faces, count, frustum_mat))
    else:
        # single mesh
        board_points = calibs[0]["grid"]
        board_mat = "board_mat"
        materials.append((board_mat, (0.2, 0.8, 1.0), 0.3))
        start_faces = len(faces)
        # use only the outer quad of the board
        ux = np.unique(board_points[:, 0])
        uy = np.unique(board_points[:, 1])
        nx, ny = ux.size, uy.size
        world = board_points.reshape((nx, ny, 3), order="F")
        corners = [
            world[0, 0],
            world[nx - 1, 0],
            world[nx - 1, ny - 1],
            world[0, ny - 1],
        ]
        v0 = len(vertices)
        vertices.extend([tuple(v.tolist()) for v in corners])
        faces.append([v0 + 1, v0 + 2, v0 + 3, v0 + 4])
        count = len(faces) - start_faces
        group_info.append(("board", start_faces, count, board_mat))
        frustum_mat = "frustum_mat"
        materials.append((frustum_mat, (1.0, 0.4, 0.2), 1.0))
        start_faces = len(faces)
        for calib in calibs:
            extr = calib["extr"]
            K = calib["K"]
            imsize = calib["imsize"]
            board_extent = np.max(np.ptp(board_points, axis=0))
            near = 0.05 * board_extent
            far = near * 3
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            w, h = imsize
            # compute camera pose only from the first extrinsic entry per camera
            if extr.shape[0] == 0:
                continue
            rvec = extr[0, :3]
            tvec = extr[0, 3:]
            R_inv, t_inv = invert_extrinsic(rvec, tvec)
            pts_cam: List[List[float]] = []
            for d in (near, far):
                for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
                    x = (u - cx) / fx * d
                    y = (v - cy) / fy * d
                    pts_cam.append([x, y, d])
            pts_cam_arr = np.array(pts_cam)
            world_pts = (R_inv @ pts_cam_arr.T).T + t_inv.reshape(1, 3)
            apex_idx = len(vertices)
            vertices.append(tuple(t_inv.tolist()))
            start_verts = len(vertices)
            vertices.extend([tuple(p.tolist()) for p in world_pts])
            # pyramid sides
            for i in range(4):
                faces.append(
                    [
                        apex_idx + 1,
                        start_verts + i + 1,
                        start_verts + ((i + 1) % 4) + 1,
                    ]
                )
            # far quad
            quad = [start_verts + i + 1 for i in range(4, 8)]
            faces.append(quad)
        count = len(faces) - start_faces
        group_info.append(("camera_frustums", start_faces, count, frustum_mat))
    # write the mtl file
    mtl_path = base_name + ".mtl"
    with open(mtl_path, "w") as f_mtl:
        for name, colour, alpha in materials:
            f_mtl.write(f"newmtl {name}\n")
            f_mtl.write(f"Kd {colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f}\n")
            f_mtl.write(f"d {alpha:.2f}\nKa 0 0 0\nKs 0 0 0\n\n")
    # write obj file
    obj_path = base_name + ".obj"
    with open(obj_path, "w") as f_obj:
        f_obj.write(f"mtllib {base_name}.mtl\n")
        # vertices
        for v in vertices:
            f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # groups
        for group, start, cnt, mat in group_info:
            f_obj.write(f"g {group}\nusemtl {mat}\n")
            for fi in range(start, start + cnt):
                face = faces[fi]
                # obj indices are 1‑based
                f_obj.write("f " + " ".join(str(idx) for idx in face) + "\n")
    print(
        f"Exported scene to {obj_path} (vertices={len(vertices)}, faces={len(faces)})"
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    # parse command line options
    parser = argparse.ArgumentParser(
        description="Visualise OpenCV calibration in 3‑D and export to OBJ"
    )
    parser.add_argument("calib_files", nargs="+", help="Calibration YAML files")
    parser.add_argument(
        "--view",
        choices=["camera", "board"],
        default=None,
        help="Reference frame: 'camera' shows board relative to a single camera, 'board' shows cameras around the board."
        "The default is 'camera' for a single file and 'board' for multiple files.",
    )
    parser.add_argument(
        "--export", metavar="BASE", help="Export scene as BASE.obj/BASE.mtl for MeshLab"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Do not open the interactive viewer"
    )
    parser.add_argument(
        "--max-frustums",
        type=int,
        default=None,
        help="Limit the number of frustums per camera when exporting or drawing (board view only)",
    )
    args = parser.parse_args(argv)
    view = args.view
    if view is None:
        view = "camera" if len(args.calib_files) == 1 else "board"
    calibs: List[Dict[str, object]] = []
    for idx, fname in enumerate(args.calib_files):
        errs, extr, grid, K, imsize, board_size = load_calib(fname)
        calib_data: Dict[str, object] = {
            "errors": errs,
            "extr": extr,
            "grid": grid,
            "K": K,
            "imsize": imsize,
            "board_size": board_size,
            "label": f"Cam{idx}",
        }
        calibs.append(calib_data)
    # export if requested
    if args.export:
        export_scene_to_obj_multi(
            args.export,
            calibs,
            board_size,
            view=view,
            max_frustums_per_cam=args.max_frustums,
        )
    # interactive plot can be displayed optionally
    if not args.no_gui:
        plt.style.use("dark_background")
        if len(calibs) == 1 and view == "camera":
            fig = plt.figure(figsize=(14, 6))
            fig.patch.set_facecolor("black")
            ax_err = fig.add_subplot(1, 2, 1)
            ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
            ax_3d.set_facecolor("black")
            # Plot error bars
            errors = calibs[0]["errors"]
            idxs = np.arange(1, len(errors) + 1)
            bars = ax_err.bar(
                idxs, errors, picker=5, color="#00BCD4", edgecolor="white"
            )
            m = np.mean(errors)
            ax_err.axhline(
                m, color="#FF5722", linestyle="--", label=f"Mean = {m:.3f}px"
            )
            ax_err.set_xlabel("Image index", color="white")
            ax_err.set_ylabel("Error (px)", color="white")
            ax_err.set_title("Per-view RMS Reprojection Error", color="white")
            ax_err.tick_params(colors="white")
            ax_err.legend(loc="upper right", facecolor="black", edgecolor="white")
            meshes = plot_scene_camera_view(
                ax_3d,
                calibs[0]["extr"],
                calibs[0]["grid"],
                calibs[0]["K"],
                calibs[0]["imsize"],
                calibs[0]["board_size"],
                errors=calibs[0]["errors"],
            )

            def onselect_y(vmin, vmax):
                threshold = min(vmin, vmax)
                for i, bar in enumerate(bars):
                    if bar.get_height() >= threshold:
                        bar.set_color("#FFEB3B")
                        bar.set_edgecolor("yellow")
                        meshes[i].set_alpha(0.8)
                        meshes[i].set_edgecolor("yellow")
                    else:
                        bar.set_color("#00BCD4")
                        bar.set_edgecolor("white")
                        meshes[i].set_alpha(0.1)
                        meshes[i].set_edgecolor("white")
                fig.canvas.draw_idle()

            SpanSelector(
                ax_err,
                onselect_y,
                "vertical",
                useblit=True,
                props=dict(alpha=0.3, facecolor="yellow"),
                minspan=0,
            )

            def on_pick(event):
                artist = event.artist
                if artist in bars:
                    idx = list(bars).index(artist)
                elif artist in meshes:
                    idx = meshes.index(artist)
                else:
                    return
                for j, bar in enumerate(bars):
                    if j == idx:
                        bar.set_color("#FFEB3B")
                        bar.set_edgecolor("yellow")
                        meshes[j].set_alpha(0.8)
                        meshes[j].set_edgecolor("yellow")
                    else:
                        bar.set_color("#00BCD4")
                        bar.set_edgecolor("white")
                        meshes[j].set_alpha(0.1)
                        meshes[j].set_edgecolor("white")
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("pick_event", on_pick)
            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(figsize=(8, 6))
            fig.patch.set_facecolor("black")
            ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
            ax_3d.set_facecolor("black")
            if view == "camera":
                # single camera without error bar
                plot_scene_camera_view(
                    ax_3d,
                    calibs[0]["extr"],
                    calibs[0]["grid"],
                    calibs[0]["K"],
                    calibs[0]["imsize"],
                    calibs[0]["board_size"]
                )
            else:
                # board view for multiple cameras
                plot_scene_board_view(
                    ax_3d,
                    calibs,
                    calibs[0]["board_size"],
                    max_frustums_per_cam=args.max_frustums,
                )
            plt.tight_layout()
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
