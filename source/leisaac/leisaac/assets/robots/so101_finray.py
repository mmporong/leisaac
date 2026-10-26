"""SO101 follower with 80 mm TPU FinRay fingers on both jaws.

The fingers are rigid convex-hull proxies of the STL; the real TPU fingers bend.
The STL is read from ASSETS_ROOT/robots/so101_finray_80mm_lip3_tpu95a.stl and its
hash is checked, because the mesh itself is not stored in this repository.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from leisaac.utils.constant import ASSETS_ROOT
from pxr import UsdGeom, UsdPhysics, Vt

from .lerobot import SO101_FOLLOWER_CFG
from .so101_finray_geometry import fixed_finger_in_gripper, load_binary_stl, moving_finger_in_jaw

FINRAY_STL_PATH = Path(ASSETS_ROOT) / "robots" / "so101_finray_80mm_lip3_tpu95a.stl"
_FINRAY_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.93, 0.9), roughness=0.7)


def _define_finger(stage, prim_path: str, triangles, material_path: str) -> None:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(triangles.reshape(-1, 3).astype("float32")))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(3 * len(triangles)))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    sim_utils.bind_visual_material(prim_path, material_path)


def spawn_so101_with_finray(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the SO101 follower and add a FinRay finger to the gripper and jaw links."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    triangles = load_binary_stl(FINRAY_STL_PATH)
    fixed, moving = fixed_finger_in_gripper(triangles), moving_finger_in_jaw(triangles)
    stage = get_current_stage()
    for robot_path in find_matching_prim_paths(prim_path):
        material_path = f"{robot_path}/FinRayMaterial"
        _FINRAY_MATERIAL.func(material_path, _FINRAY_MATERIAL)
        _define_finger(stage, f"{robot_path}/gripper/finray_fixed", fixed, material_path)
        _define_finger(stage, f"{robot_path}/jaw/finray_moving", moving, material_path)
    return prim


SO101_FOLLOWER_FINRAY_CFG = SO101_FOLLOWER_CFG.replace(
    spawn=SO101_FOLLOWER_CFG.spawn.replace(func=spawn_so101_with_finray)
)
