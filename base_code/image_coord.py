import numpy as np
from rect import rect

class image_coord:
    def __init__(self, size=None, origin=None, spacing=None, direction=None):
        self.size = np.array(size).astype(int)
        self.origin = np.array(origin).astype(float)
        self.spacing = np.array(spacing).astype(float)
        if direction is not None:
            self.direction = np.array(direction).astype(float)
        else:
            self.direction = np.array(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def to_np_array(self):
        return np.concatenate([self.size.astype(float), self.spacing, self.origin, self.direction])

    def from_np_array(self, arr):
        self.size = np.array([arr[0], arr[1], arr[2]]).astype(int)
        self.origin = np.array([arr[3], arr[4], arr[5]]).astype(float)
        self.spacing = np.array([arr[6], arr[7], arr[8]]).astype(float)
        self.direction = np.array([arr[9], arr[10], arr[11], arr[12],
                                   arr[13], arr[14], arr[15], arr[16], arr[17]]).astype(float)

    def __str__(self):
        return 'size:{self.size},origin:{self.origin},spacing:{self.spacing},direction:{self.direction}'.format(self=self)

    def rect_o(self):
        raise Exception('DO NOT USE THIS AMBICUOUS CODE!!! This is the rect of the coordiante system with respect to w only when the direction is indentity')
        return rect(self.origin, self.origin+self.size*self.spacing)

    def size_phys(self):
        return self.size*self.spacing

    def rect_I(self):
        return rect(np.array([0, 0, 0]).astype(int), self.size)

    # convert a point in w to I.
    def w2I(self, pt_w):
        pt_o = self.w2o(pt_w)
        return np.round(pt_o/self.spacing)

    # convert a point in w to o
    def w2o(self, pt_w):
        wDo = self.direction.reshape(3,3)
        oDw = np.linalg.inv(wDo)
        pt_o = np.matmul(oDw, pt_w-self.origin)
        return np.array(pt_o)

    # convert a point in w to u (normalized coordinate system)
    def w2u(self, pt_w):
        pt_o=self.w2o(pt_w)
        return np.array(pt_o/self.size_phys())

    # convert a point in o to I.
    def o2I(self, pt_o):
        return np.array(np.round(pt_o/self.spacing))

    # convert a point in I to o.
    def I2o(self, pt_I):
        return pt_I * self.spacing
    
    # convert a point in w to u (normalized coordinate system)
    def o2u(self, pt_o):
        return pt_o/self.size_phys()

    # convert a point in I to w
    def o2w(self, pt_o):
        wDo = self.direction.reshape(3,3)
        pt_w = np.matmul(wDo, pt_o)+self.origin
        return pt_w

    # convert a point in I to w
    def I2w(self, pt_I):
        pt_o = pt_I*self.spacing
        pt_w = self.o2w(pt_o)
        return pt_w
    
    # w_H_o
    def w_H_o(self):
        # Construct the homogeneous transformation matrix
        T = np.zeros((4, 4))
        T[:3, :3] = self.direction.reshape((3,3))
        T[:3, 3] = self.origin
        T[3, 3] = 1
        return T
    def o_H_w(self):
        return np.linalg.inv(self.w_H_o())

    # o_H_I
    def o_H_I(self):
        diag = np.append(self.spacing, 1.0)
        T = np.diag(diag)
        return T
    def I_H_o(self):
        return np.linalg.inv(self.o_H_I())

    # w_H_I
    def w_H_I(self):
        return self.w_H_o()@self.o_H_I()
    
    def I_H_w(self):
        return np.linalg.inv(self.w_H_I())



