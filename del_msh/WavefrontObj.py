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


def load(file_path: str):
    o = WavefrontObj()
    o.vtxxyz2xyz, o.vtxuv2uv, o.vtxnrm2nrm, \
    o.elem2idx, o.idx2vtxxyz, o.idx2vtxuv, o.idx2vtxnrm, \
    o.elem2group, o.group2name, \
    o.elem2mtl, o.mtl2name, o.mtl_file_name = load_wavefront_obj(file_path)
    return o


def read_material(path: str):
    with open(path) as f:
        dict_mtl = {}
        cur_mtl = {}
        cur_name = ""
        for line in f:
            if line.startswith('#'):
                continue
            words = line.split()
            if len(words) == 2 and words[0] == 'newmtl':
                cur_name = words[1]
                cur_mtl = {}
            if len(words) == 0 and cur_name != "":
                dict_mtl[cur_name] = cur_mtl
            if len(words) == 4 and words[0] == 'Kd':
                cur_mtl['Kd'] = (float(words[1]), float(words[2]), float(words[3]))
    return dict_mtl
