#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photoye 熟人底库与多姿态质心特征服务 (V3.0)

负责人物质心特征管理、已知人像毫秒级比对判定、质心平滑移动平均更新，
以及跨批次未知面孔的 DBSCAN 增量聚类发现。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from data.database import (
    assign_face_to_person,
    blob_to_array,
    create_person,
    get_connection,
    get_person_by_id,
    list_all_persons,
    mark_face_as_noise,
    update_person_centroids,
)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """对特征向量执行 L2 单位化。"""
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    return vec


def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个单位特征向量的余弦相似度。"""
    a = normalize_vector(vec_a)
    b = normalize_vector(vec_b)
    return float(np.dot(a, b))


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """计算一组特征向量的算术平均质心并进行 L2 归一化。"""
    if not embeddings:
        raise ValueError("无法为特征向量为空的列表计算质心")
    mean_vec = np.mean(embeddings, axis=0)
    return normalize_vector(mean_vec)


@dataclass
class FaceMatchResult:
    """人脸底库匹配比对结果"""
    person_id: Optional[int]
    person_name: Optional[str]
    similarity: float
    matched_pose: str  # "front", "profile", 或 "none"
    is_matched: bool


class FaceLibrary:
    """熟人持久底库管理器

    特性:
    - 多姿态质心支持: 每个人物同时维护正脸质心 (front) 与侧脸质心 (profile)
    - 动态自适应比对: 取双质心相似度最大值判定
    - 在线滑动平均更新: 命中已知人物后自动平滑微调对应姿态质心
    - 增量聚类发现: 未匹配的陌生人脸批次调用 DBSCAN 自动聚合成待命名的新人物
    """

    def __init__(
        self,
        db_conn: Optional[sqlite3.Connection] = None,
        match_threshold: float = 0.65,
        centroid_alpha: float = 0.85,
    ) -> None:
        """初始化底库管理器。

        Args:
            db_conn: SQLite 连接对象。若未传入则默认使用全局连接。
            match_threshold: 判定为已知熟人的余弦相似度阈值 (默认 0.65)。
            centroid_alpha: 质心移动平均保留历史权重的衰减系数 (默认 0.85)。
        """
        self.conn = db_conn
        self.match_threshold = match_threshold
        self.centroid_alpha = centroid_alpha

    def _get_connection(self) -> sqlite3.Connection:
        """获取可用数据库连接。"""
        if self.conn is not None:
            return self.conn
        return get_connection()

    def match_face(
        self,
        embedding: np.ndarray,
        yaw_angle: Optional[float] = None,
    ) -> FaceMatchResult:
        """在底库中查找与输入人脸特征最匹配的已知人物。

        比对策略:
        Sim = max(dot(f, C_front), dot(f, C_profile))
        """
        conn = self._get_connection()
        persons = list_all_persons(conn=conn)
        
        if not persons:
            return FaceMatchResult(
                person_id=None,
                person_name=None,
                similarity=0.0,
                matched_pose="none",
                is_matched=False,
            )

        norm_emb = normalize_vector(embedding)
        best_person_id: Optional[int] = None
        best_person_name: Optional[str] = None
        highest_similarity = -1.0
        best_matched_pose = "none"

        for p in persons:
            p_id = p["id"]
            p_name = p["name"]
            sim_front = -1.0
            sim_profile = -1.0

            if p["centroid_front"] is not None:
                c_front = blob_to_array(p["centroid_front"])
                sim_front = compute_similarity(norm_emb, c_front)

            if p["centroid_profile"] is not None:
                c_profile = blob_to_array(p["centroid_profile"])
                sim_profile = compute_similarity(norm_emb, c_profile)

            current_best_sim = max(sim_front, sim_profile)
            current_pose = "front" if sim_front >= sim_profile else "profile"

            if current_best_sim > highest_similarity:
                highest_similarity = current_best_sim
                best_person_id = p_id
                best_person_name = p_name
                best_matched_pose = current_pose

        is_hit = highest_similarity >= self.match_threshold
        return FaceMatchResult(
            person_id=best_person_id if is_hit else None,
            person_name=best_person_name if is_hit else None,
            similarity=highest_similarity if is_hit else (highest_similarity if highest_similarity > 0 else 0.0),
            matched_pose=best_matched_pose if is_hit else "none",
            is_matched=is_hit,
        )

    def enroll_face_sample(
        self,
        person_id: int,
        face_id: int,
        embedding: np.ndarray,
        is_profile: bool = False,
        cover_face_id: Optional[int] = None,
    ) -> bool:
        """为已有熟人汇入新的人脸样本，并动态平滑微调对应姿态质心。

        C_new = Normalize(alpha * C_old + (1 - alpha) * f)
        """
        conn = self._get_connection()
        person = get_person_by_id(person_id, conn=conn)
        if not person:
            return False

        norm_emb = normalize_vector(embedding)

        if is_profile:
            old_blob = person["centroid_profile"]
            if old_blob is not None:
                old_centroid = blob_to_array(old_blob)
                updated_centroid = normalize_vector(
                    self.centroid_alpha * old_centroid + (1.0 - self.centroid_alpha) * norm_emb
                )
            else:
                updated_centroid = norm_emb
            update_person_centroids(
                person_id=person_id,
                centroid_profile=updated_centroid,
                additional_face_count=1,
                cover_face_id=cover_face_id,
                conn=conn,
            )
        else:
            old_blob = person["centroid_front"]
            if old_blob is not None:
                old_centroid = blob_to_array(old_blob)
                updated_centroid = normalize_vector(
                    self.centroid_alpha * old_centroid + (1.0 - self.centroid_alpha) * norm_emb
                )
            else:
                updated_centroid = norm_emb
            update_person_centroids(
                person_id=person_id,
                centroid_front=updated_centroid,
                additional_face_count=1,
                cover_face_id=cover_face_id,
                conn=conn,
            )

        # 建立数据库人脸归属
        assign_face_to_person(face_id=face_id, person_id=person_id, conn=conn)
        return True

    def cluster_unknown_faces(
        self,
        unassigned_faces: List[Dict[str, Any]],
        eps: float = 0.65,
        min_samples: int = 2,
    ) -> Dict[str, Any]:
        """对批次中未匹配到已知底库的未知人脸进行 DBSCAN 聚类。

        Args:
            unassigned_faces: 未命名人脸字典列表，需包含 id 与 embedding (bytes 或 ndarray)
            eps: 邻域最大余弦距离 (距离 = 1 - 相似度，默认 0.65)
            min_samples: 构成独立人物聚类簇的最少人脸数 (默认 2)

        Returns:
            {
                "clusters": {cluster_idx: [face_id1, face_id2, ...]},
                "cluster_centroids": {cluster_idx: centroid_array},
                "noise_face_ids": [face_id3, ...]
            }
        """
        if not unassigned_faces:
            return {
                "clusters": {},
                "cluster_centroids": {},
                "noise_face_ids": [],
            }

        face_ids: List[int] = []
        embedding_list: List[np.ndarray] = []

        for item in unassigned_faces:
            fid = item["id"]
            raw_emb = item["embedding"]
            emb = blob_to_array(raw_emb) if isinstance(raw_emb, bytes) else raw_emb
            face_ids.append(fid)
            embedding_list.append(normalize_vector(emb))

        matrix = np.array(embedding_list, dtype=np.float32)
        matrix = normalize(matrix, norm="l2")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
        labels = dbscan.fit_predict(matrix)

        clusters: Dict[int, List[int]] = {}
        cluster_embeddings: Dict[int, List[np.ndarray]] = {}
        noise_face_ids: List[int] = []

        for idx, label in enumerate(labels):
            fid = face_ids[idx]
            if label == -1:
                noise_face_ids.append(fid)
            else:
                if label not in clusters:
                    clusters[label] = []
                    cluster_embeddings[label] = []
                clusters[label].append(fid)
                cluster_embeddings[label].append(embedding_list[idx])

        # 为聚类出的每个簇计算初始质心
        cluster_centroids: Dict[int, np.ndarray] = {}
        for c_idx, embs in cluster_embeddings.items():
            cluster_centroids[c_idx] = compute_centroid(embs)

        return {
            "clusters": clusters,
            "cluster_centroids": cluster_centroids,
            "noise_face_ids": noise_face_ids,
        }

    def register_new_person_from_cluster(
        self,
        name: str,
        face_ids: List[int],
        centroid: np.ndarray,
        cover_face_id: Optional[int] = None,
    ) -> int:
        """将用户确认命名的新人物聚类写入底库并关联所有人脸。"""
        conn = self._get_connection()
        person_id = create_person(
            name=name,
            centroid_front=centroid,
            cover_face_id=cover_face_id or (face_ids[0] if face_ids else None),
            face_count=len(face_ids),
            conn=conn,
        )
        for fid in face_ids:
            assign_face_to_person(face_id=fid, person_id=person_id, conn=conn)
        return person_id

    def mark_faces_as_noise(self, face_ids: List[int]) -> None:
        """批量标记孤立或模糊的人脸为噪声。"""
        conn = self._get_connection()
        for fid in face_ids:
            mark_face_as_noise(face_id=fid, conn=conn)
