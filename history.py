import numpy as np
np.set_printoptions(precision=6)
ts = 0
bp1 = 1
bp2 = 2
bp3 = 3
bp4 = 4
bp5 = 5
bp6 = 6
bp7 = 7
bp8 = 8
bp9 = 9
bp10 = 10
bv1 = 11
bv2 = 12
bv3 = 13
bv4 = 14
bv5 = 15
bv6 = 16
bv7 = 17
bv8 = 18
bv9 = 19
bv10 = 20
ap1 = 21
ap2 = 22
ap3 = 23
ap4 = 24
ap5 = 25
ap6 = 26
ap7 = 27
ap8 = 28
ap9 = 29
ap10 = 30
av1 = 31
av2 = 32
av3 = 33
av4 = 34
av5 = 35
av6 = 36
av7 = 37
av8 = 38
av9 = 39
av10 = 40
lv = 41
ld = 42
cf = 43
wp = 44
o = 45
c = 46
h = 47
low = 48


class History:
    def __init__(self, size):
        self.size = size
        self.cur_size = 0
        self.history = self.__init_matrix()

    def __init_matrix(self):
        return np.zeros(
            self.size * 49,
            dtype=np.float64).reshape(
            self.size,
            49)

    def clear(self):
        del self.history
        self.cur_size = 0
        self.history = self.__init_matrix()

    def is_full(self):
        return self.size == self.cur_size

    def append(self, data):
        if self.is_full():
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = data
        else:
            self.history[self.cur_size] = data
            self.cur_size = self.cur_size + 1

    def get_obs(self):
        lr = np.diff(np.log(self.history[:, wp]), 1, axis=0)
        bpd1 = np.diff(
            np.log(self.history[:, bp1] * self.history[:, bv1]), 1, axis=0)
        bpd2 = np.diff(
            np.log(self.history[:, bp2] * self.history[:, bv2]), 1, axis=0)
        bpd3 = np.diff(
            np.log(self.history[:, bp3] * self.history[:, bv3]), 1, axis=0)
        bpd4 = np.diff(
            np.log(self.history[:, bp4] * self.history[:, bv4]), 1, axis=0)
        bpd5 = np.diff(
            np.log(self.history[:, bp5] * self.history[:, bv5]), 1, axis=0)
        apd1 = np.diff(
            np.log(self.history[:, ap1] * self.history[:, av1]), 1, axis=0)
        apd2 = np.diff(
            np.log(self.history[:, ap2] * self.history[:, av2]), 1, axis=0)
        apd3 = np.diff(
            np.log(self.history[:, ap3] * self.history[:, av3]), 1, axis=0)
        apd4 = np.diff(
            np.log(self.history[:, ap4] * self.history[:, av4]), 1, axis=0)
        apd5 = np.diff(
            np.log(self.history[:, ap5] * self.history[:, av5]), 1, axis=0)
        s1 = (self.history[:, ap1] - self.history[:, bp1]) /\
            (self.history[:, ap1] + self.history[:, bp1])
        s2 = (self.history[:, ap2] - self.history[:, bp2]) /\
            (self.history[:, ap2] + self.history[:, bp2])
        s3 = (self.history[:, ap3] - self.history[:, bp3]) /\
            (self.history[:, ap3] + self.history[:, bp3])
        s4 = (self.history[:, ap4] - self.history[:, bp4]) /\
            (self.history[:, ap4] + self.history[:, bp4])
        s5 = (self.history[:, ap5] - self.history[:, bp5]) /\
            (self.history[:, ap5] + self.history[:, bp5])
        cf5 = np.sum(self.rolling_window(self.history[:, cf], 5), axis=1)
        cf3 = np.sum(self.rolling_window(self.history[:, cf], 3), axis=1)

        lr = self.scale(lr, 5)
        bpd1 = self.scale(bpd1, 10)
        bpd2 = self.scale(bpd2, 10)
        bpd3 = self.scale(bpd3, 10)
        bpd4 = self.scale(bpd4, 10)
        bpd5 = self.scale(bpd5, 10)
        apd1 = self.scale(apd1, 10)
        apd2 = self.scale(apd2, 10)
        apd3 = self.scale(apd3, 10)
        apd4 = self.scale(apd4, 10)
        apd5 = self.scale(apd5, 10)
        s1 = self.scale(s1, 10)
        s2 = self.scale(s2, 10)
        s3 = self.scale(s3, 10)
        s4 = self.scale(s4, 10)
        s5 = self.scale(s5, 10)
        cf_1 = self.scale(self.history[:, cf], 10)

        cf5 = self.scale(cf5, 10)
        cf3 = self.scale(cf3, 10)

        re = np.array([lr, bpd1, bpd2, bpd3, bpd4, bpd5, apd1, apd2,
                      apd3, apd4, apd5, s1, s2, s3, s4, s5, cf_1, cf5, cf3])
        return np.round(re, decimals=6)

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)

        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def scale(self, a, d):
        l = len(a)
        b = a[l - d:l]
        x = a[-1]

        return (x - np.min(b)) / (np.max(b) - np.min(b))

    def info(self):
        print(self.history)
