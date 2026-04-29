"""Marker package so tests can ``import scripts.patch_checkpoint`` etc.

The :mod:`scripts` directory only ships utility scripts. Marking it as a
package keeps test imports stable without forcing each script to be on
``sys.path``.
"""
