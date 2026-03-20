"""Filters module - aggregates all filter implementations.

This module provides a convenient single import for all filters.
"""

from src.part_a.filter_interface import ImageFilterInterface
from src.part_a.mean_filter import MeanFilter
from src.part_a.gaussian_filter import GaussianFilter
from src.part_a.median_filter import MedianFilter
from src.part_a.laplacian_sharpener import LaplacianSharpener

__all__ = [
    "ImageFilterInterface",
    "MeanFilter",
    "GaussianFilter",
    "MedianFilter",
    "LaplacianSharpener",
]
