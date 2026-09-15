"""Face Recognition with ArcFace (ONNX) and 5-point alignment.

Stages are separated into modules so each one can be validated independently:

* :mod:`camera`    -- camera access and frame-rate sanity
* :mod:`detect`    -- face bounding boxes
* :mod:`landmarks` -- 5-point landmark extraction
* :mod:`align`     -- geometric alignment to a canonical 112x112 crop
* :mod:`embed`     -- ArcFace embedding extraction (ONNX, CPU)
* :mod:`enroll`    -- reference template enrollment
* :mod:`evaluate`  -- threshold tuning on genuine/impostor pairs
* :mod:`recognize` -- live multi-face identification
"""

__version__ = "1.0.0"