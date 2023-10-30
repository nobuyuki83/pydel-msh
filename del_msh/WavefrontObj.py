from .del_msh import load_wavefront_obj

class WavefrontObj:

    def __init__(self):
        self.vtxxyz2xyz = None
        self.vtxuv2uv = None
        self.vtxnrm2nrm = None
        self.elem2idx = None
        self.idx2vtxxyz = None
        self.idx2vtxuv = None
        self.idx2vtxnrm = None
        self.elem2group = None
        self.group2name = None
        self.elem2mtl = None
        self.mtl2name = None
        self.mtl_file_name = None

def load(file_path:str):
    o = WavefrontObj()
    o.vtxxyz2xyz, o.vtxuv2uv, o.vtxnrm2nrm, \
        o.elem2idx, o.idx2vtxxyz, o.idx2vtxuv, o.idx2vtxnrm, \
        o.elem2group, o.group2name, \
        o.elem2mtl, o.mtl2name, o.mtl_file_name = load_wavefront_obj(file_path)
    return o