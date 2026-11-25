"""Testler proje kök klasöründen çalışır: python -m pytest"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
