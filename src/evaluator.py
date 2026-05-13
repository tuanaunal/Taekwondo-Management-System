"""
evaluator.py — Performans Değerlendirme ve Hata Matrisi
========================================================
Sistemin Ghost Hit / Real Hit sınıflandırma başarısını
Confusion Matrix, Precision, Recall ve F1-Score ile ölçer.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score,
)
from src.config import GRAPHS_DIR, REPORTS_DIR


class PerformanceEvaluator:
    """
    Toplu analiz sonuçlarını değerlendiren ve raporlayan sınıf.
    """

    def __init__(self):
        self.y_true = []     # Gerçek etiketler
        self.y_pred = []     # Tahmin edilen etiketler
        self.video_names = []
        self.details = []    # Her video için detaylı sonuçlar

    def add_result(
        self,
        video_name: str,
        true_label: str,
        predicted_label: str,
        confidence: float = 0.0,
        details: dict = None,
    ):
        """
        Bir video analizinin sonucunu kaydeder.

        Parameters
        ----------
        video_name : str
            Video dosya adı.
        true_label : str
            Gerçek etiket ("Ghost_Hit" veya "Real_Hit").
        predicted_label : str
            Tahmin edilen etiket ("GHOST_HIT" veya "REAL_HIT").
        confidence : float
            Tahmin güven skoru.
        details : dict
            Ek analiz detayları.
        """
        # Etiketleri standartlaştır
        true_binary = 1 if "ghost" in true_label.lower() else 0
        pred_binary = 1 if "ghost" in predicted_label.lower() else 0

        self.y_true.append(true_binary)
        self.y_pred.append(pred_binary)
        self.video_names.append(video_name)
        self.details.append({
            "video": video_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
            "confidence": confidence,
            **(details or {}),
        })

    def compute_metrics(self) -> dict:
        """
        Tüm kayıtlı sonuçlar üzerinden metrikleri hesaplar.

        Returns
        -------
        dict
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'confusion_matrix': np.ndarray,
                'classification_report': str,
                'total_samples': int,
                'correct': int,
                'incorrect': int,
            }
        """
        if len(self.y_true) == 0:
            return {"error": "Henüz sonuç kaydedilmedi."}

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report = classification_report(
            y_true, y_pred,
            target_names=["Real Hit", "Ghost Hit"],
            zero_division=0,
        )

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        correct = int(np.sum(y_true == y_pred))
        total = len(y_true)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "total_samples": total,
            "correct": correct,
            "incorrect": total - correct,
        }

    def plot_confusion_matrix(self, save_path: str = None) -> str:
        """
        Confusion Matrix ısı haritasını çizer ve kaydeder.

        Returns
        -------
        str
            Kaydedilen grafik dosya yolu.
        """
        metrics = self.compute_metrics()
        if "error" in metrics:
            return ""

        cm = metrics["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        # Isı haritası
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        classes = ["Real Hit", "Ghost Hit"]
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=classes, yticklabels=classes,
            xlabel="Tahmin Edilen",
            ylabel="Gerçek Değer",
            title="Confusion Matrix (Karışıklık Matrisi)",
        )

        # Hücre değerlerini yaz
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                label = f"{cm[i, j]}"
                # TP/TN/FP/FN etiketleri
                if i == 0 and j == 0:
                    label += "\n(TN)"
                elif i == 0 and j == 1:
                    label += "\n(FP)"
                elif i == 1 and j == 0:
                    label += "\n(FN)"
                else:
                    label += "\n(TP)"

                ax.text(
                    j, i, label,
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def plot_metrics_summary(self, save_path: str = None) -> str:
        """
        Metrik özetini bar grafiği olarak çizer.
        """
        metrics = self.compute_metrics()
        if "error" in metrics:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

        # ── Sol: Metrik Çubukları ──
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        metric_values = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        ]
        colors = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6"]

        bars = axes[0].bar(metric_names, metric_values, color=colors)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title("Sınıflandırma Metrikleri", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Değer")

        for bar, val in zip(bars, metric_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center", va="bottom", fontweight="bold",
            )

        # ── Sağ: Doğru/Yanlış Pasta ──
        correct = metrics["correct"]
        incorrect = metrics["incorrect"]
        axes[1].pie(
            [correct, incorrect],
            labels=[f"Doğru ({correct})", f"Yanlış ({incorrect})"],
            colors=["#2ECC71", "#E74C3C"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        axes[1].set_title("Genel Doğruluk", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(GRAPHS_DIR, "metrics_summary.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def generate_report(self, save_path: str = None) -> str:
        """
        Detaylı CSV raporu oluşturur.

        Returns
        -------
        str
            Kaydedilen rapor dosya yolu.
        """
        if not self.details:
            return ""

        df = pd.DataFrame(self.details)

        if save_path is None:
            save_path = os.path.join(REPORTS_DIR, "analysis_report.csv")

        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return save_path

    def print_summary(self) -> str:
        """Konsola yazdırılacak metin raporu."""
        metrics = self.compute_metrics()
        if "error" in metrics:
            return metrics["error"]

        lines = [
            "",
            "=" * 60,
            "  PERFORMANS DEĞERLENDİRME RAPORU",
            "=" * 60,
            "",
            f"  Toplam Video:     {metrics['total_samples']}",
            f"  Doğru Tahmin:     {metrics['correct']}",
            f"  Yanlış Tahmin:    {metrics['incorrect']}",
            "",
            f"  Accuracy:         {metrics['accuracy']:.1%}",
            f"  Precision:        {metrics['precision']:.1%}",
            f"  Recall:           {metrics['recall']:.1%}",
            f"  F1-Score:         {metrics['f1_score']:.1%}",
            "",
            "─" * 60,
            "  DETAYLI SINIFLANDIRMA RAPORU",
            "─" * 60,
            metrics["classification_report"],
            "=" * 60,
        ]
        return "\n".join(lines)

    def reset(self):
        """Tüm kayıtları sıfırlar."""
        self.y_true = []
        self.y_pred = []
        self.video_names = []
        self.details = []
