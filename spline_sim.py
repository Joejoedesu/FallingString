import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import PIL
import imageio.v2 as imageio
import random
import argparse

class Spline:

    def __init__(self, P, res=100):
        self.P = P
        self.res = res
        self.L = self.inter_spline_5(self.P, res=self.res)
        # self.L = np.array(self.L)
        d = 0
        for i in range(len(self.L)-1):
            d += self.dist(self.L[i], self.L[i+1])
            # print(self.dist(self.L[i], self.L[i+1]))
        self.delta = d / (len(self.L)-1)
        self.V = np.zeros((len(self.L), 3))
        self.F = np.zeros((len(self.L), 3))
        if res == 8:
            self.stiff_st = 300
            self.damp_st = 4.5
            self.stiff_bn = 10
        elif res == 64:
            self.stiff_st = 100
            self.damp_st = 2.5
            self.stiff_bn = 5
        self.T = 0

    def dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def translate(self, p):
        self.L += p
        self.P += p

    def vom(self):
        return np.mean(self.V, axis=0)

    def catmull_spline(self, p0, p1, p2, p3, res=100, seg=0):
        """
        Catmull-Rom spline interpolation
        """
        M = np.array([[-0.5, 1.5, -1.5, 0.5],\
                    [1.0, -2.5, 2.0, -0.5],\
                    [-0.5, 0.0, 0.5, 0.0],\
                    [0.0, 1.0, 0.0, 0.0]])
        G = np.array([p0, p1, p2, p3])
        line = []
        num = res//2
        for i in range(num):
            if seg == 0:
                t = i / (res-1) * 2
            else:
                t = (i + num) / (res-1) * 2 - 1
            T = np.array([t**3, t**2, t, 1])
            x = np.dot(T, np.dot(M, G[:, 0]))
            y = np.dot(T, np.dot(M, G[:, 1]))
            z = np.dot(T, np.dot(M, G[:, 2]))
            line.append(np.array([x, y, z]))
        return np.array(line)

    def random_spline(self, res = 1024):
        p0 = np.array([0, 0, 0])
        p = [p0]
        for i in range(4):
            phi = random.uniform(0, 2*np.pi)
            theta = random.uniform(0, np.pi)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            p.append(p[i] + np.array([x, y, z]))
        self.P = p
        return p

    def inter_spline_5(self, p, res =1024):
        if len(p) < 5:
            p = self.random_spline()
        L = []
        line = self.catmull_spline(p[0], p[1], p[2], p[3], res=res, seg=0)
        # print(line)
        L.extend(line)
        line = self.catmull_spline(p[1], p[2], p[3], p[4], res=res, seg=1)
        # print(line)
        L.extend(line)
        return np.array(L)

    def hook_op(self):
        vom = self.vom()
        for i in range(0, len(self.L)-1):
            dir = self.L[i+1] - self.L[i] # i+1 is the r
            l = np.linalg.norm(dir)
            dir = dir / np.linalg.norm(dir)
            f_abs = self.stiff_st * (l - self.delta)
            f_1 = f_abs * dir
            f_2 = -1.0 * f_1
            f_1 -= self.damp_st * (self.V[i] - vom)
            f_2 -= self.damp_st * (self.V[i+1] - vom)
            self.F[i] += f_1
            self.F[i+1] += f_2

    def bend_op(self):
        for i in range(1, len(self.L)-1):
            p0 = self.L[i-1]
            p1 = self.L[i]
            p2 = self.L[i+1]
            a_d = p1 - p0
            b_d = p2 - p1
            dot_ab = np.dot(a_d, b_d)
            a_h = a_d / np.linalg.norm(a_d)
            b_h = b_d / np.linalg.norm(b_d)
            a = np.linalg.norm(a_d)
            b = np.linalg.norm(b_d)
            f0 = -self.stiff_bn / (2*a) * (b_h - a_h *dot_ab)
            if np.linalg.norm(f0) < 3:
                f0 = np.zeros(3)
            f2 = self.stiff_bn / (2*b)* (a_h - b_h *dot_ab)
            if np.linalg.norm(f2) <3:
                f2 = np.zeros(3)
            self.F[i] += (-f0-f2)
            if i == 1:
                self.F[i-1] += f2
            if i == len(self.L)-2:
                self.F[i+1] += f0

    def external_force(self, t, condi = "floor"):
        # self.T += t
        # self.V[0][0] = -1.
        # self.V[0][1] = 0.
        # self.V[0][2] = 0.
        fri_coeff = 0.8 * 9.8
        fri_shreshold = 0.1
        if condi == "floor":
            for i in range(0, len(self.L)):
                self.F[i] += np.array([0, 0, -9.8])
            for i in range(0, len(self.L)):
                if self.L[i][2] + self.V[i][2] * t < 0:
                    self.L[i][2] = 0.001
                    self.V[i][2] = 0
                    self.F[i][2] = 0
                    v_2d = np.array([self.V[i][0], self.V[i][1]])
                    v_dir = v_2d / np.linalg.norm(v_2d)
                    if np.linalg.norm(v_2d) < fri_shreshold:
                        self.F[i][0] = 0
                        self.V[i][0] = 0
                        self.F[i][1] = 0
                        self.V[i][1] = 0
                    else:
                        self.F[i][0] -= v_dir[0] * fri_coeff
                        self.F[i][1] -= v_dir[1] * fri_coeff

        elif condi == "spring":
            for i in range(0, len(self.L)):
                lim = 3
                z = self.L[i][2]
                if z > lim:
                    self.F[i] += np.array([0, 0, -9.8])
                else:
                    coeff = 15
                    self.F[i] += np.array([0, 0, (lim-z)**2]) * coeff

        elif condi == "wave":
            for i in range(0, len(self.L)):
                z = self.L[i][2]
                lim = np.sin(self.L[i][0] + self.L[i][1]) + 3
                if z > lim:
                    self.F[i] += np.array([0, 0, -9.8])
                else:
                    coeff = 15
                    self.F[i] += np.array([0, 0, (lim-z)**2]) * coeff
        
        elif condi == "wind":
            center = np.array([1, 1])
            for i in range(0, len(self.L)):
                angle = np.arctan2(self.L[i][1] - center[1], self.L[i][0] - center[0])
                dir = np.array([np.cos(angle), np.sin(angle)])
                dis = np.linalg.norm(self.L[i][:2] - center)
                spin = 10 * min(0, 2-dis)
                attr = -2 * dis
                f_x = np.sin(angle) * spin + dir[0] * attr
                f_y = np.cos(angle) * spin + dir[1] * attr
                lim = 3
                if self.L[i][2] > lim:
                    self.F[i] += np.array([f_x, f_y, -9.8])
                else:
                    self.F[i] += np.array([f_x, f_y, (lim-self.L[i][2])**2]) * 8


    def update(self, t, env = "floor"):
        self.hook_op()
        # self.bend_op()
        # self.external_force(t, "floor")
        self.external_force(t, env)

        self.L += self.V * t
        self.V += self.F * t
        # print(self.F[1])
        self.F = np.zeros((len(self.L), 3))

def create_sample(duration=80, gran=100, env="floor", sample=1, image=True):
    for sample_ in range(sample):
        Location = []
        s = Spline([], res=64)
        s.translate(np.array([random.uniform(0, 4), random.uniform(0, 4), random.uniform(4, 10)]))
        Location.append(np.copy(s.L))
        for i in range(duration*gran):
            s.update(.1/gran, env)
            Location.append(np.copy(s.L))
            if i % gran == 0 and image:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                L = np.array(s.L)
                #fix ax limits
                ax.set_xlim(-1, 4)
                ax.set_ylim(-1, 4)
                ax.set_zlim(-1, 10)
                ax.plot(L[:, 0], L[:, 1], L[:, 2])
                ax.scatter(L[:, 0], L[:, 1], L[:, 2])
                #color the first point red
                ax.scatter(L[0, 0], L[0, 1], L[0, 2], color='r')
                #show x y z axis
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                #save to frame folder
                plt.savefig('frames/frame' + str(i//gran) + '.png')
                plt.close(fig)
                # plt.show()
        
        if image:
            image_list = []
            for i in range(duration):
                image_list.append(imageio.imread('frames/frame' + str(i) + '.png'))
            imageio.mimsave(f'clips/movie_{sample_}.gif', image_list)
        np.save(f'data/start/spline_{env}_{sample_}.npy', np.array(s.P))
        np.save(f'data/sim/location_{env}_{sample_}.npy', np.array(Location))

def main():
    create_sample(duration=40, gran=100, env="floor", sample=200, image=False)


if __name__ == "__main__":
    main()