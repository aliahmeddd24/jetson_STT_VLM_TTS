"""Minimal URDF reader: joint limits and forward kinematics (numpy only)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis / (np.linalg.norm(axis) or 1.0)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    lower: Optional[float] = None
    upper: Optional[float] = None
    velocity: Optional[float] = None


class URDF:
    def __init__(self, path: str):
        root = ET.parse(path).getroot()
        self.joints: Dict[str, Joint] = {}
        self.child_to_joint: Dict[str, Joint] = {}
        for j in root.findall("joint"):
            T = np.eye(4)
            o = j.find("origin")
            if o is not None:
                xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
                T[:3, :3] = _rpy_to_R(*rpy)
                T[:3, 3] = xyz
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            lim = j.find("limit")
            jt = Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                       j.find("child").get("link"), T, axis)
            if lim is not None:
                jt.lower = float(lim.get("lower")) if lim.get("lower") is not None else None
                jt.upper = float(lim.get("upper")) if lim.get("upper") is not None else None
                jt.velocity = float(lim.get("velocity")) if lim.get("velocity") is not None else None
            self.joints[jt.name] = jt
            self.child_to_joint[jt.child] = jt

    def limits(self, name: str) -> Tuple[float, float]:
        j = self.joints[name]
        if j.lower is None or j.upper is None:
            raise KeyError(f"joint {name} has no position limits")
        return j.lower, j.upper

    def chain(self, link: str) -> List[Joint]:
        out: List[Joint] = []
        while link in self.child_to_joint:
            j = self.child_to_joint[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def link_pose(self, link: str, q: Dict[str, float]) -> np.ndarray:
        """Pose of `link` in the URDF root frame for joint values q (missing joints = 0)."""
        T = np.eye(4)
        for j in self.chain(link):
            T = T @ j.origin
            if j.type in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q.get(j.name, 0.0))
                T = T @ R
            elif j.type == "prismatic":
                P = np.eye(4)
                P[:3, 3] = j.axis * q.get(j.name, 0.0)
                T = T @ P
        return T

    def position_in(self, link: str, frame: str, q: Dict[str, float]) -> np.ndarray:
        Tf = self.link_pose(frame, q)
        Tl = self.link_pose(link, q)
        return (np.linalg.inv(Tf) @ Tl)[:3, 3]
