import numpy as np
from numba import guvectorize, njit

########################################################################################################################### #normal function


@guvectorize(
    ["void(float64[:], float64[:], float64[:])"],
    "(n),(n)->(n)",
)
def a_filter(a, b, out):

    for i in range(len(a)):
        if b[i] > 0.5:
            out[i] = a[i]
        else:
            out[i] = np.nan


def a_min(a, b):
    return np.fmin(a, b)


def a_max(a, b):
    return np.fmax(a, b)


# ts function
@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_forwardfill(a, n, out):
    pre_pos = -1
    pre_val = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]):
            pre_pos, pre_val = i, a[i]
        else:
            if i - pre_pos > n:
                pre_val = np.nan
        out[i] = pre_val


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_interpolate(a, n, out):
    pre_pos = -1
    cur_pos = -1
    for i in range(len(a)):
        if np.isfinite(a[i]):
            out[i] = a[i]
            if cur_pos == -1:
                cur_pos = i

            else:
                pre_pos = cur_pos
                cur_pos = i
                if cur_pos - pre_pos <= n + 1:
                    for j in range(pre_pos + 1, cur_pos):
                        out[j] = (a[pre_pos] * (cur_pos - j) + a[cur_pos] * (j - pre_pos)) / (
                            cur_pos - pre_pos
                        )
        else:
            out[i] = np.nan


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_delta(a, n, out):
    ra = np.zeros(n)  # ring buffer
    first_idx, end_idx = 0, 0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, end_idx = 0, 0
            out[i] = np.nan
        else:
            ra[end_idx] = a[i]
            end_idx = (end_idx + 1) % n
            out[i] = a[i] - ra[first_idx]
            if end_idx == first_idx:
                first_idx = (first_idx + 1) % n


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_delay(a, n, out):
    ra = np.zeros(n)  # ring buffer
    first_idx, end_idx = 0, 0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, end_idx = 0, 0
            out[i] = np.nan
        else:
            ra[end_idx] = a[i]
            end_idx = (end_idx + 1) % n
            out[i] = ra[first_idx]
            if end_idx == first_idx:
                first_idx = (first_idx + 1) % n


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_sum(a, n, out):

    ra = np.zeros(n)  # ring buffer
    first_idx, end_idx, a_s = 0, 0, 0.0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, end_idx, a_s = 0, 0, 0.0
            out[i] = np.nan
        else:
            ra[end_idx] = a[i]
            end_idx = (end_idx + 1) % n
            a_s += a[i]
            out[i] = a_s
            if end_idx == first_idx:
                a_s -= ra[first_idx]
                first_idx = (first_idx + 1) % n


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_mean(a, n, out):

    ra = np.zeros(n)  # ring buffer
    first_idx, end_idx, a_s = 0, 0, 0.0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, end_idx, a_s = 0, 0, 0.0
            out[i] = np.nan
        else:
            ra[end_idx] = a[i]
            end_idx = (end_idx + 1) % n
            a_s += a[i]
            cnt = (end_idx - 1 - first_idx) % n + 1
            out[i] = a_s / cnt
            if end_idx == first_idx:
                a_s -= ra[first_idx]
                first_idx = (first_idx + 1) % n


@guvectorize(
    ["void(float64[:], int64, float64, float64[:])"],
    "(n),(),()->(n)",
)
def ts_ema(a, n, w, out):

    ra = np.zeros(n)  # ring buffer
    first_idx, end_idx, a_s = 0, 0, 0.0
    ratio = pow(1 - w, n - 1)

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, end_idx, a_s = 0, 0, 0.0
            out[i] = np.nan
        else:
            ra[end_idx] = a[i]
            end_idx = (end_idx + 1) % n
            cnt = (end_idx - 1 - first_idx) % n + 1
            if cnt == 1:
                a_s = a[i]
            else:
                a_s = a_s * (1 - w) + w * a[i]
            out[i] = a_s

            if end_idx == first_idx:
                a_s -= ra[first_idx] * ratio

                first_idx = (first_idx + 1) % n
                a_s += ra[first_idx] * ratio


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_min(a, n, out):

    ra = np.zeros(n)
    rpos = np.zeros(n)
    first_idx, end_idx, ii = 0, 0, 0
    for i in range(len(a)):

        if not np.isfinite(a[i]):
            out[i] = np.nan
            first_idx, end_idx = 0, 0
            ii = 0
            continue

        while rpos[first_idx] <= ii - n:
            first_idx = (first_idx + 1) % n

        while first_idx != end_idx and ra[(end_idx - 1) % n] >= a[i]:
            end_idx = (end_idx - 1) % n

        ra[end_idx], rpos[end_idx] = a[i], ii
        end_idx = (end_idx + 1) % n
        out[i] = ra[first_idx]
        ii += 1


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_max(a, n, out):

    ra = np.zeros(n)
    rpos = np.zeros(n)
    first_idx, end_idx, ii = 0, 0, 0

    for i in range(len(a)):

        if not np.isfinite(a[i]):
            out[i] = np.nan
            first_idx, end_idx = 0, 0
            ii = 0
            continue

        while rpos[first_idx] <= ii - n:
            first_idx = (first_idx + 1) % n

        while first_idx != end_idx and ra[(end_idx - 1) % n] <= a[i]:
            end_idx = (end_idx - 1) % n

        ra[end_idx], rpos[end_idx] = a[i], ii
        end_idx = (end_idx + 1) % n
        out[i] = ra[first_idx]
        ii += 1


@guvectorize(["void(float64[:], int64, float64[:])"], "(n),()->(n)", nopython=True)
def ts_rank(a, n, out):
    ra = np.full((n,), np.nan)  # ring buffer

    first_idx = 0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            out[i] = np.nan
            first_idx = 0
            ra[:] = np.nan
            continue

        ra[first_idx] = a[i]

        first_idx += 1
        if first_idx >= n:
            first_idx = 0

        rk = 0.0
        rk_cnt = 0.0
        cnt = 0
        for j in range(n):
            if np.isfinite(ra[j]):
                if ra[j] < a[i]:
                    rk += 1
                elif ra[j] == a[i]:
                    rk_cnt += 1
                cnt += 1
        if cnt == 1:
            out[i] = 0.5
        else:
            out[i] = (2 * rk + rk_cnt - 1) / (cnt - 1) / 2


@guvectorize(
    ["void(float64[:], float64[:], int64, float64[:])"],
    "(n),(n),()->(n)",
)
def ts_corr(a, b, n, out):

    ra = np.zeros(n)  # ring buffer
    rb = np.zeros(n)  # ring buffer

    first_idx, a_s, b_s, aa_s, bb_s, ab_s, cnt = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for i in range(len(a)):
        if not np.isfinite(a[i]) or not np.isfinite(b[i]):
            first_idx, a_s, b_s, aa_s, bb_s, ab_s, cnt = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            out[i] = np.nan
            continue
        if cnt < n:
            cnt += 1
        next_idx = first_idx - 1
        if next_idx < 0:
            next_idx = n - 1

        a_s += a[i] - ra[next_idx]
        aa_s += a[i] * a[i] - ra[next_idx] * ra[next_idx]
        b_s += b[i] - rb[next_idx]
        bb_s += b[i] * b[i] - rb[next_idx] * rb[next_idx]
        ab_s += a[i] * b[i] - ra[next_idx] * rb[next_idx]

        ra[next_idx] = a[i]
        rb[next_idx] = b[i]

        first_idx += 1
        if first_idx >= n:
            first_idx = 0

        ab_ = (aa_s - a_s * a_s / cnt) * (bb_s - b_s * b_s / cnt)
        if ab_ == 0.0:
            out[i] = np.nan
        else:
            out[i] = (ab_s - a_s * b_s / cnt) / pow(ab_, 0.5)


@guvectorize(
    ["void(float64[:], float64[:], int64, float64[:])"],
    "(n),(n),()->(n)",
)
def ts_cov(a, b, n, out):

    ra = np.zeros(n)  # ring buffer
    rb = np.zeros(n)  # ring buffer

    first_idx, a_s, b_s, ab_s, cnt = 0, 0.0, 0.0, 0.0, 0
    for i in range(len(a)):
        if not np.isfinite(a[i]) or not np.isfinite(b[i]):
            out[i] = np.nan
            first_idx, a_s, b_s, ab_s, cnt = 0, 0.0, 0.0, 0.0, 0
            continue
        if cnt < n:
            cnt += 1
        next_idx = first_idx - 1
        if next_idx < 0:
            next_idx = n - 1

        a_s += a[i] - ra[next_idx]

        b_s += b[i] - rb[next_idx]

        ab_s += a[i] * b[i] - ra[next_idx] * rb[next_idx]

        ra[next_idx] = a[i]
        rb[next_idx] = b[i]

        first_idx += 1
        if first_idx >= n:
            first_idx = 0

        out[i] = (ab_s - a_s * b_s / cnt) / cnt


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_std(a, n, out):

    ra = np.zeros(n)  # ring buffer

    first_idx, a_s, aa_s, cnt = 0, 0.0, 0.0, 0
    for i in range(len(a)):
        if not np.isfinite(a[i]):
            first_idx, a_s, aa_s, cnt = 0, 0.0, 0.0, 0
            out[i] = np.nan
            continue
        if cnt < n:
            cnt += 1

        a_s += a[i] - ra[first_idx]
        aa_s += a[i] * a[i] - ra[first_idx] * ra[first_idx]

        ra[first_idx] = a[i]

        first_idx += 1
        if first_idx >= n:
            first_idx = 0

        tmp_ = aa_s - a_s * a_s / cnt
        if tmp_ > 1e-15:
            out[i] = pow((aa_s - a_s * a_s / cnt) / cnt, 0.5)
        else:
            out[i] = 0.0


"""
Part I. Incremental:
- Tn = x1^2 + ... + xn^2,   Sn = x1 + ... + xn,    Vn = (1/n) Tn - (1/n * Sn)^2
- Tn = T(n-1) + xn^2,       Sn = S(n-1) + xn,
- n*Vn - (n-1) * V(n-1) = (Tn - T(n-1)) - (1/n * Sn^2 + 1/(n-1)* S(n-1)^2)
                        = xn^2 - 1/n * Sn * (S(n-1) + xn) + 1/(n-1)* S(n-1)^2)
                        = xn(xn - 1/n * Sn) - S(n-1) * (1/n * Sn - 1/(n-1) * S(n-1))
                        = 1/n/(n-1) * ((n-1)xn - S(n-1))^2
- Vn = (n-1) / n * V(n-1) + 1/ n^2 / (n-1) * ((n-1)xn - S(n-1))^2

Part II. Equal rolling:
- Tm = x(m-n+1)^2 + ... + xm^2,   Sm = x(m-n+1) + ... + xm,    Vm = (1/n) Tm - (1/n * Sm)^2
- Tm = T(m-1) + xm^2 - x(m-n)^2,       Sm = S(m-1) + xm - x(m-n),
- n* Vm - n * V(m-1) = (Tm - T(m-1)) - 1/n (Sm^2 - S(m-1)^2)
                     = (xm^2 - x(m-n)^2) - 1/n * (xm -x(m-n)) * (Sm + S(m-1))
                     = (xm - x(m-n)) * (xm + x(m-n) - 1/n * (Sm + S(m-1)))
- Vm = V(m-1) + (xm - x(m-n)) * (xm + x(m-n) - 1/n * (Sm + S(m-1))) / n
"""


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_std_welford(a, n, out):

    ra = np.zeros(n)
    first_idx = 0
    v_ = np.float64(0.0)
    sum_ = np.float64(0.0)
    cnt = 0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            out[i] = np.nan
            continue
        if cnt == 0:
            sum_ = a[i]
            ra[first_idx] = a[i]
            first_idx += 1
        elif cnt < n:

            m = cnt + 1
            r1 = np.float64(m - 1) / m
            r2 = 1 / np.float64(m - 1) / m / m
            tmp = (m - 1) * a[i] - sum_
            v_ = r1 * v_ + r2 * tmp * tmp
            sum_ += a[i] - ra[first_idx]
            ra[first_idx] = a[i]
            first_idx = (first_idx + 1) % n

        else:
            sum_old = sum_
            sum_ += a[i] - ra[first_idx]
            v_ += (a[i] - ra[first_idx]) * (a[i] + ra[first_idx] - (sum_ + sum_old) / n) / n
            ra[first_idx] = a[i]
            first_idx = (first_idx + 1) % n

        if v_ < 0:
            out[i] = 0
        else:
            out[i] = pow(v_, 0.5)
        cnt += 1


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_std_welford_wiki(a, n, out):
    ra = np.zeros(n)
    first_idx = 0

    cnt = 0
    mean = 0.0
    M2 = 0.0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            out[i] = np.nan
            continue

        if cnt < n:

            newValue = a[i]
            cnt += 1
            delta = newValue - mean
            mean += delta / cnt
            delta2 = newValue - mean
            M2 += delta * delta2

            ra[first_idx] = a[i]
            first_idx = (first_idx + 1) % n
        else:

            newValue = a[i]
            oldValue = ra[first_idx]
            mean_old = mean
            mean += (newValue - oldValue) / n
            M2 += (newValue - oldValue) * ((newValue + oldValue) - (mean + mean_old))

            ra[first_idx] = a[i]
            first_idx = (first_idx + 1) % n

        # print('after round ', i, 'sum =', sum_ , ', variance=', v_)
        v_ = M2 / cnt
        if v_ < 0:
            out[i] = 0
        else:
            out[i] = pow(v_, 0.5)


# cross-sectional function
@guvectorize(
    ["void(float64[:], float64[:])"],
    "(n)->(n)",
)
def c_rank(a, out):
    m = np.arange(len(a))
    b = np.zeros(len(a))
    cnt = 0
    for i in range(len(a)):
        if ~np.isfinite(a[i]):
            out[i] = np.nan
        else:
            b[cnt] = a[i]
            m[cnt] = i
            cnt += 1
    ivec = np.argsort(b[:cnt])

    sumranks = 0
    dupcount = 0

    for i in range(cnt):
        sumranks += i
        dupcount += 1
        if (
            i == cnt - 1
            or abs(b[ivec[i]] - b[ivec[i + 1]]) / (abs(b[ivec[i]]) + abs(b[ivec[i + 1]]) + 1e-10)
            > 1e-6
        ):
            averank = sumranks / float(dupcount)
            for j in range(i - dupcount + 1, i + 1):
                out[m[ivec[j]]] = averank / (cnt - 1)
            sumranks = 0
            dupcount = 0


def c_scale(a, target=2e5):
    sum_ = np.nansum(np.abs(a), axis=1)
    sum_[sum_ < 1e-10] = 0.0
    return target * a / sum_[:, None]


def c_demean(x):
    return x - np.nanmean(x, axis=1, keepdims=True)


# group function


@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->(n)", nopython=True)
def g_demean(a, b, out):
    n = len(a)
    g_max = 0
    for i in range(n):
        if (
            not np.isnan(a[i])
            and not np.isinf(a[i])
            and not np.isnan(a[i])
            and not np.isinf(a[i])
            and np.isfinite(b[i])
        ):
            g_max = max(g_max, abs(int(b[i])) * 2)

    grp_sum = np.zeros(g_max + 1)
    grp_cnt = np.zeros(g_max + 1)

    for i in range(n):
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            g = int(b[i])
            grp_sum[g] += a[i]
            grp_cnt[g] += 1

    for g in range(len(grp_sum)):
        grp_sum[g] /= grp_cnt[g]

    for i in range(n):
        out[i] = a[i] - grp_sum[int(b[i])]


# how to eliminate nan
@guvectorize(
    ["void(float64[:], float64[:])"],
    "(n)->(n)",
)
def log(a, out):
    n = len(a)

    for i in range(n):
        if not np.isfinite(a[i]) or a[i] < 1e-20:
            out[i] = np.nan
        else:
            out[i] = np.log(a[i])


"""
my_assert(tsmin([1,2,3,4,5],2), np.array([1.,1.,2.,3.,4.]))
my_assert(tsmin([5,4,3,2,1],2), np.array([5.,4.,3.,2.,1.]))
my_assert(tsmin([5,4,9,2,1,6,3,5],3), np.array([5.,4.,4.,2.,1.,1.,1.,3.]))
my_assert(tsmin([13, 10, 18, 3, 15, 19, 17, 6, 15, 11],4), np.array([13.,10.,10.,3.,3.,3.,3.,6., 6.,6.]))
my_assert(tsmax([1,2,3,4,5],2), np.array([1.,2.,3.,4.,5.]))
my_assert(tsmax([5,4,3,2,1],2), np.array([5.,5.,4.,3.,2.]))
my_assert(tsmax([5,4,9,2,1,6,3,5],3), np.array([5.,5.,9.,9.,9.,6.,6.,6.]))
my_assert(tsmax([13, 10, 18, 3, 15, 19, 17, 6, 15, 11],4), np.array([13.,13.,18.,18.,18.,19.,19.,19., 19.,17.]))
my_assert(tsmin([ 0.44,0.42,0.409,0.33,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,-0.0341,0.41,0.635,0.7231,], 5), np.array([0.44,0.42,0.409,0.33,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,-0.0341,-0.0341,-0.0341,-0.0341]))

my_assert(tsmax([ 0.44,0.42,0.409,0.33,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,-0.0341,0.41,0.635,0.7231], 5), np.array([0.44,0.44,0.44,0.44,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0.44,0.42,0.635,0.7231]))

my_assert(tsrank([ 0.44,0.42,0.409,0.33,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,-0.0341,0.42,0.635,0.7231,-0.0341,0.42,0.935,1.7231,-0.03,0.99], 5), np.array([0.0,0.0,0.0,0.0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0.0,3.0,4.0,4.0,0.0,1.0,4.0,4.0,1.0,3.0,]))
"""


@guvectorize(
    ["void(float64[:], int64, float64[:])"],
    "(n),()->(n)",
)
def ts_forwardfill_mean(a, n, out):

    ra = np.zeros(n)  # ring buffer

    first_idx = 0
    a_s = 0.0

    for i in range(len(a)):
        if not np.isfinite(a[i]):
            out[i] = a_s / n
        else:
            if first_idx >= n:
                first_idx = 0
            a_s += a[i] - ra[first_idx]

            ra[first_idx] = a[i]
            first_idx += 1
            out[i] = a[i]


def a_mask(a, b):
    c = a.copy()
    c[~b] = np.nan
    return c


def op_lin_group(a, *args):
    a_ = rank(a.arr)
    r_ = args[0].arr.copy()
    n = len(args)
    for i in range(1, n):
        mask_ = a_ > 1.0 * i / n
        r_[mask_] = args[i].arr[mask_]

    return matf(r_)


@guvectorize(
    ["void(float64[:], float64[:])"],
    "(n)->(n)",
)
def ts_ffillna(a, out):

    flag = False
    for i in range(len(a)):
        if not flag and np.isfinite(a[i]):
            flag = True
            out[i] = a[i]
        elif flag and ~np.isfinite(a[i]):
            out[i] = 0.0
        else:
            out[i] = a[i]

def c_scale(arr, sum_to=1):
    arr[~np.isfinite(arr)] = 0
    x = np.abs(arr).sum(axis = 1)
    return (arr.T / x).T * sum_to