from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything)': SAMModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
    "AutomaticSAMSegment (segment anything)": AutomaticSAMSegment,
    "RAMModelLoader (segment anything)": RAMModelLoader,
    "RAMSAMSegment (segment anything)": RAMSAMSegment,
    "CalculateMaskCenters": CalculateMaskCenters,
    "MaskToRandomLatent": MaskToRandomLatentNode,
    "ComputeSurfaceTiltAngle": ComputeSurfaceTiltAngleNode
}

__all__ = ['NODE_CLASS_MAPPINGS']


