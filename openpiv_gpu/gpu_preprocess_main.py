"""This module contains algorithms for PIV image preprocessing with a GPU."""

import numpy as np
import cupy as cp
from . import DTYPE_u
DTYPE_f = np.float32

# Default settings.
OVERLAP = 1
Y_OFFSET = 0

# Allowed settings.
ALLOWED_DATA_TYPES = {"uint8", "uint16", "uint32", "uint64"}

class stitch_gpu:
    """Wrapper-class for StitchGPU that further applies input validation and provides user inetrface.
    
    Parameters
    ----------
    frame_a_shape : tuple
        Shape of the left image in pixels.
    frame_b_shape : tuple
        Shape of the right image in pixels.
    **kwargs
        Stitching settings. See StitchGPU.
    
    Attributes
    ----------
    frame_shape : tuple
        Shape of the stitched frame in pixels.
    
    """
    def __init__(self, frame_a_shape, frame_b_shape, **kwargs):
        overlap = kwargs["overlap"] if "overlap" in kwargs else OVERLAP
        y_offset = kwargs["y_offset"] if "y_offset" in kwargs else Y_OFFSET
        dtype_u = kwargs["dtype_u"] if "dtype_u" in kwargs else "uint16"
        
        # Check the settings.
        self.overlap = overlap
        assert isinstance(self.overlap, int) and \
            self.overlap > 0, "{} must be an {} number greater than zero.".format("overlap", "int")
        
        self.frame_a_shape = frame_a_shape
        assert isinstance(self.frame_a_shape, tuple) and \
            len(self.frame_a_shape) == 2 and \
                all(isinstance(item, int) for item in self.frame_a_shape) and \
                    all(item >= self.overlap for item in self.frame_a_shape), \
                        "{} must be a tuple of {} values greater than {} of {}." \
                            .format("frame_a_shape", "int", "overlap", self.overlap)
        
        self.frame_b_shape = frame_b_shape
        assert isinstance(self.frame_b_shape, tuple) and \
            len(self.frame_b_shape) == 2 and \
                all(isinstance(item, int) for item in self.frame_b_shape) and \
                    all(item >= self.overlap for item in self.frame_b_shape), \
                        "{} must be a tuple of {} values greater than {} of {}." \
                            .format("frame_b_shape", "int", "overlap", self.overlap)
        
        ht_a, ht_b = self.frame_a_shape[0], self.frame_b_shape[0]
        assert ht_a == ht_b, "Both frames must have the same height."
        
        self.y_offset = y_offset
        assert isinstance(self.y_offset, int) and \
            self.y_offset >= -ht_b and self.y_offset <= ht_b\
                , "{} must be an {} number less than {}.".format("y_offset", "int", ht_a)
        
        self.dtype_u = dtype_u
        assert isinstance(self.dtype_u, str) and \
            self.dtype_u in ALLOWED_DATA_TYPES, "{} must be one of {}.".format("dtype_u", ALLOWED_DATA_TYPES)
        
        # Initialize the process.
        self.gpu_stitch = StitchGPU(frame_a_shape, frame_b_shape, **kwargs)
        
    def __call__(self, frame_a, frame_b):
        """Stitches two frames of the same height.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        Returns
        -------
        frame_stitched : ndarray
            CuPy 2D array containing grey levels of the stitched frame.
        
        """
        assert isinstance(frame_a, np.ndarray) and \
            frame_a.shape == self.frame_a_shape and \
                np.issubdtype(frame_a.dtype, np.number) and \
                    not np.iscomplex(frame_a).any(), \
                        "Left frame must be an ndarray of {} values with shape {}.".format("real", self.frame_a_shape)
        
        assert isinstance(frame_b, np.ndarray) and \
            frame_b.shape == self.frame_b_shape and \
                np.issubdtype(frame_b.dtype, np.number) and \
                    not np.iscomplex(frame_b).any(), \
                        "Right frame must be an ndarray of {} values with shape {}.".format("real", self.frame_a_shape)
        
        return self.gpu_stitch(frame_a, frame_b)
    
    @property
    def frame_shape(self):
        "Returns the shape of the stitched frames."
        return self.gpu_stitch.frame_shape

class StitchGPU:
    """Stitches two frames of the same height by linearly merging the intensity values in the overlapping region.
    
    Parameters
    ----------
    frame_a_shape, frame_b_shape : tuple
        Shape of the images in pixels.
    overlap : int, optional
        Value of the overlap between two frames.
    y_offset : int, optional
        Value of the vertical shift for the right frame.
    dtype_u : str, optional
        Uint data type.
    """
    def __init__(self, frame_a_shape, frame_b_shape,
                 overlap=OVERLAP,
                 y_offset=Y_OFFSET,
                 dtype_u=DTYPE_u):
        
        self.ht_a, self.wd_a = frame_a_shape
        self.ht_b, self.wd_b = frame_b_shape
        self.overlap = overlap
        self.y_offset = y_offset if y_offset >= 0 else -y_offset
        self.is_reversed = False if y_offset >= 0 else True
        
        # Swap the frame widths if y_offset is negative.
        if self.is_reversed:
            wd_t = self.wd_a
            self.wd_a = self.wd_b
            self.wd_b = wd_t
        
        self.x_offset = self.wd_a - self.overlap
        self.frame_shape = (self.ht_a + self.y_offset, self.wd_a + self.wd_b - self.overlap)
        
        # Data type settings.
        self.dtype_f = DTYPE_f
        if dtype_u == "uint8":
            self.dtype_u = cp.uint8
        elif dtype_u == "uint32":
            self.dtype_u = cp.uint32
        elif dtype_u == "uint64":
            self.dtype_u = cp.uint64
        else:
            self.dtype_u = DTYPE_u
        
        # Initialize the temporary arrays.
        self.zl = cp.zeros((self.y_offset, self.x_offset), dtype=self.dtype_u)
        self.zr = cp.zeros((self.y_offset, self.wd_b - self.overlap), dtype=self.dtype_u)
        self.fm = cp.tile(cp.linspace(1, 0, self.overlap, dtype=self.dtype_f), (self.ht_a - self.y_offset, 1))
    
    def __call__(self, frame_a, frame_b):
        """Performs the stitching for two frames of the same height.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        Returns
        -------
        frame_stitched : ndarray
            CuPy 2D array containing grey levels of the stitched frame.
        
        """
        # Send frames to GPU.
        frame_a = cp.asarray(frame_a)
        frame_b = cp.asarray(frame_b)
        
        # Swap the frames if y_offset is negative.
        frame_a, frame_b = frame_a.astype(self.dtype_u), frame_b.astype(self.dtype_u)
        if self.is_reversed:
            frame_t = frame_a
            frame_a = frame_b[:, ::-1]
            frame_b = frame_t[:, ::-1]
        
        # Slice the left and right frames.
        fl = cp.concatenate((self.zl, frame_a[:, :self.x_offset]), axis=0)
        fr = cp.concatenate((frame_b[:, self.overlap:], self.zr), axis=0)
        
        # Merge the overlapping values linearly.
        fu = frame_b[:self.y_offset, :self.overlap]
        fd = frame_a[self.ht_b - self.y_offset:, self.x_offset:]
        fm = self.fm * frame_a[:self.ht_a - self.y_offset, self.x_offset:] + (1 - self.fm) * frame_b[self.y_offset:, :self.overlap]
        fm = cp.concatenate((fu, fm.astype(self.dtype_u), fd), axis=0)
        
        # Merge the left, middle, and right sections.
        frame_stitched = cp.concatenate((fl, fm, fr), axis=1).astype(self.dtype_u)
        
        # Clear the stored frames.
        fu, fd, fl, fr, fm = None, None, None, None, None
        frame_a, frame_b = None, None
        if self.is_reversed:
            frame_stitched = frame_stitched[:, ::-1]
        
        return frame_stitched