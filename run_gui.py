"""
run_gui.py — Taekwondo Ghost Hit Detection: GUI Giriş Noktası
===============================================================
PyQt5 masaüstü uygulamasını başlatır.

Kullanım:
    python run_gui.py
"""

import sys
import os

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # Uygulama geneli font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Uygulama stili
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
