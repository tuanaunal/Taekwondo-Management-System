"""
tracker.py — Centroid Tracking (Ağırlık Merkezi Takibi)
========================================================
Segmente edilen nesnelerin kare-kare merkez koordinatlarını
izleyerek hareket yörüngesi oluşturur.
"""

import numpy as np
from collections import OrderedDict


class CentroidTracker:
    """
    Basit Centroid Tracking algoritması.

    Her karede tespit edilen nesnelerin merkez noktalarını,
    önceki karedeki nesnelerle Öklid mesafesi bazlı eşleştirir.

    Parameters
    ----------
    max_disappeared : int
        Bir nesne kaç kare boyunca kaybolabilir (oklüzyon toleransı).
    """

    def __init__(self, max_disappeared: int = 10):
        self.next_id = 0
        self.objects = OrderedDict()        # id -> (cx, cy)
        self.disappeared = OrderedDict()    # id -> kayıp kare sayısı
        self.max_disappeared = max_disappeared

        # Hareket geçmişi: id -> [(cx, cy, frame_idx), ...]
        self.trajectories = {}

    def register(self, centroid: tuple, frame_idx: int = 0):
        """Yeni nesne kaydı."""
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = [(centroid[0], centroid[1], frame_idx)]
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id: int):
        """Nesneyi kayıttan siler."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        # Trajectory'yi silmiyoruz — analiz için lazım

    def update(self, centroids: list, frame_idx: int = 0) -> OrderedDict:
        """
        Yeni karedeki merkez noktalarını mevcut nesnelerle eşleştirir.

        Parameters
        ----------
        centroids : list of tuple
            Bu karedeki tespit edilen merkez noktaları [(cx, cy), ...].
        frame_idx : int
            Mevcut kare indeksi (zaman takibi için).

        Returns
        -------
        OrderedDict
            Güncellenmiş nesne eşleşmeleri: id -> (cx, cy).
        """
        # Hiç tespit yoksa, tüm nesnelerin kayıp sayacını artır
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array(centroids)

        # Henüz takip edilen nesne yoksa, hepsini kaydet
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid, frame_idx)
            return self.objects

        # Mevcut nesne ID'leri ve konumları
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        # Öklid mesafe matrisi hesapla
        # D[i][j] = i. mevcut nesne ile j. yeni tespit arasındaki mesafe
        D = np.linalg.norm(
            obj_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2,
        )

        # En yakın eşleşmeleri bul (Greedy yöntem)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            obj_id = obj_ids[row]
            self.objects[obj_id] = tuple(input_centroids[col])
            self.disappeared[obj_id] = 0

            # Yörünge geçmişine ekle
            cx, cy = input_centroids[col]
            self.trajectories[obj_id].append((int(cx), int(cy), frame_idx))

            used_rows.add(row)
            used_cols.add(col)

        # Eşleşmeyen mevcut nesneler → kayıp
        unused_rows = set(range(len(obj_ids))) - used_rows
        for row in unused_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Eşleşmeyen yeni tespitler → yeni nesne
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]), frame_idx)

        return self.objects

    def get_trajectory(self, obj_id: int) -> list:
        """
        Belirli bir nesnenin hareket yörüngesini döner.

        Returns
        -------
        list of tuple
            [(cx, cy, frame_idx), ...]
        """
        return self.trajectories.get(obj_id, [])

    def get_all_trajectories(self) -> dict:
        """Tüm nesnelerin yörüngelerini döner."""
        return dict(self.trajectories)

    def reset(self):
        """Tüm takip verilerini sıfırlar (yeni video için)."""
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = {}


class MultiObjectTracker:
    """
    Birden fazla ekipman tipini (kask, ayak koruyucu) eş zamanlı
    takip eden üst-seviye takip yöneticisi.
    """

    def __init__(self, max_disappeared: int = 10):
        self.trackers = {
            "red_helmet": CentroidTracker(max_disappeared),
            "red_foot": CentroidTracker(max_disappeared),
            "blue_helmet": CentroidTracker(max_disappeared),
            "blue_foot": CentroidTracker(max_disappeared),
        }

    def update(self, segmentation_results: dict, frame_idx: int = 0) -> dict:
        """
        Segmentasyon sonuçlarından merkez noktalarını alıp takip eder.

        Parameters
        ----------
        segmentation_results : dict
            segmentor.segment_frame() çıktısı.
        frame_idx : int
            Mevcut kare indeksi.

        Returns
        -------
        dict
            Her ekipman tipi için güncel nesne konumları.
        """
        tracking_results = {}
        for key, tracker in self.trackers.items():
            centroids = segmentation_results.get(key, {}).get("centroids", [])
            objects = tracker.update(centroids, frame_idx)
            tracking_results[key] = dict(objects)
        return tracking_results

    def get_all_trajectories(self) -> dict:
        """Tüm ekipmanların yörünge geçmişlerini döner."""
        all_traj = {}
        for key, tracker in self.trackers.items():
            all_traj[key] = tracker.get_all_trajectories()
        return all_traj

    def reset(self):
        """Tüm takipçileri sıfırlar."""
        for tracker in self.trackers.values():
            tracker.reset()
