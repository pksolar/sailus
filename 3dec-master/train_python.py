import numpy as np
import scipy.sparse as sp
import re

def trainfq(in, samname, outname):
    # 1:9 10CR 10 SR2 11 SR2^2 12 SIG 13 1/ENDCYC 14-21 firstcycles
    # 22-69 idx[-1]-idx-subidx 70-133 idx[-3]idx[-2]idx[-1]
    prob = sp.lil_matrix((1, 1))
    param = svm_parameter('-s 2 -c 100 -t 0 -b 1 -q')
    nrfeature = 70
    loc = np.array([0, 1, 2, 4, 6, 8, 11, 14, 17])
    inter = np.array([1, 1, 2, 2, 2, 3, 3, 3])
    in.rotate(dchn)
    mdif = 0
    mdif2 = 0
    mychn = in.size[dchn]  # chn
    mycyc = in.size[dcyc]
    mycluster = in.size[dcluster]  # cluster number
    unmatched = 0
    maxnode = 29000000
    buffer = bytearray(mycyc*20 + 102)
    str1 = bytearray(mycyc*3)
    cigar = bytearray(1000)
    match = np.zeros(mycyc+2, dtype=np.int8)
    feature = []
    d = np.zeros(mychn)
    sr = np.zeros(mycyc)
    fid = np.zeros(mycyc, dtype=np.int8)
    sid = np.zeros(mycyc, dtype=np.int8)
    matchvalue = 1

    with open(samname, "r") as fin, open(outname, "w") as out:
        for tk in range(mycluster):
            i = 0
            j = 0
            cj = 0
            inverse = 0
            for kk in range(mycyc):
                match[kk] = 0
            ci = 0

            # read sam
            buffer[0] = '@'
            while buffer[0] == '@' or buffer[0] == '\n':
                buffer = fin.readline().encode()
            if tk < ((mycluster / 2 - n / mycyc) / 2):
                continue
            _, inverse, str1, _, _, cigar = buffer.decode().split()
            inverse = (int(inverse) & 16) > 0
            if str1[0] == '*':
                continue
            tt = 0
            noind = 0
            while cigar[tt]:
                if cigar[tt] == 'D' or cigar[tt] == 'I':
                    noind = 1
                    break
                tt += 1
            if noind:
                continue

            i = 70
            while i < 1900 and not (buffer[i] == 'M' and buffer[i + 1] == 'D' and buffer[i + 2] == ':' and buffer[i + 3] == 'Z' and buffer[i + 4] == ':'):
                i += 1
            if i > 189




                i = 0
                while (i < 1900 and not (buffer[i] == 'M' and buffer[i + 1] == 'D' and buffer[i + 2] == ':'
                                         and buffer[i + 3] == 'Z' and buffer[i + 4] == ':')):
                    i += 1
                if i > 1899:
                    continue
                i += 5
                j = 0

                python
                Copy
                code
                while True:
                    ci = buffer[i]
                    if ci >= '0' and ci <= '9':
                        j = j * 10 + int(ci) - int('0')
                    elif ci == 'I':
                        for k in range(j):
                            match[cj] = 1 + (not matchvalue)
                            cj += 1
                            unmatched += 1
                        j = 0
                    elif ci == 'D':
                        j = 0
                    else:
                        for kk in range(j):
                            match[cj] = 1 + matchvalue
                            cj += 1
                        j = 0
                        if ci == 'A' or ci == 'C' or ci == 'G' or ci == 'T':
                            match[cj] = 1 + (not matchvalue)
                            cj += 1
                            unmatched += 1
                        elif ci == 'N':
                            match[cj] = 1 + (not matchvalue)
                            cj += 1
                            unmatched += 1
                        elif ci == '^':
                            while buffer[i + 1] <= 'Z' and buffer[i + 1] >= 'A' or buffer[i + 1] <= 'z' and buffer[
                                i + 1] >= 'a':
                                i += 1
                                ci = buffer[i]
                        else:
                            break
                    i += 1
                    if ci == 0 or ci == ' ' or ci == '\t':
                        break
                tsr = 0
                tsr2 = 0
                sig = 0
                id = 0
                id2 = 0
                e = 0
                e2 = 0

                for j in range(mycyc):
                    tj = j + in.size[dcyc] * tk
                for kk in range(mychn):
                    d[kk] = in.data.coeff(tj, kk) / 4000
                id = 0
                e = d[0]
                for kk in range(1, mychn):
                    if e < d[kk]:
                        e = d[kk]
                id = kk
                id2 = 0
                e2 = -100000
                for k in range(mychn):
                    if k != id and e2 < d[k]:
                        id2 = k
                e2 = d[k]
                fid[j] = id
                sid[j] = id2
                tsr += e - e2
                sig += e
                tsr2 += (e - e2) * (e - e2)
                sr[j] = e - e2

                tsr /= mycyc
                sig /= mycyc
                tsr2 = tsr2 / mycyc - tsr * tsr

                for j in range(mycyc):
                    mm = 0
                sr[j] *= 5
                if sr[j] > 19.5:
                    sr[j] = 19.5
                mtk += 1
                mdif += sr[j]
                mdif2 += sr[j] * sr[j]

                python
                Copy
                code
                if match[j] == 0:
                    continue
                if sr[j














                mylong j = 0;
                while (j < mycyc)
                    {
                        int32_t mm = 0;

                    sr[j] *= 5;

                    if (sr[j] > 19.5) sr[j] = 19.5;

                    mtk++;
                    mdif += sr[j];
                    mdif2 += sr[j] * sr[j];

                    int32_t cloc;
                    if (match[j] != 0)
                    {
                    if (sr[j] >= 8)
                    cloc = floor((sr[j] + 1) / 3 + 2);
                    else if (sr[j] >= 2)
                    cloc = floor((sr[j] / 2) + 1);
                    else
                    cloc = floor(sr[j]);

                    if (cloc > 7)
                    cloc = 7;

                    float beta = 1 - (sr[j] - loc[cloc]) / inter[cloc];
                    feature[m][mm].index = cloc + 1;
                    feature[m][mm].value = beta;
                    mm++;
                    feature[m][mm].index = cloc + 2;
                    feature[m][mm].value = 1 - beta;
                    mm++;
                    }

                    if (tsr2 > 1)
                    tsr2 = 1;
                    if (sig > 3)
                    sig = 3;
                    if (sig < 0)
                    sig = 0;
                    float term = sig > 3 ? sig: 3;
                    term = term > 30 ? 30: term;
                    feature[m][mm].index = 10;
                    feature[m][mm].value = 1 / (0.02 + sqrt(tsr2));
                    mm + +;
                    feature[m][mm].index = 11;
                    feature[m][mm].value = log(0.02 + tsr2);
                    mm + +;
                    feature[m][mm].index = 12;
                    feature[m][mm].value = 1 / (0.01 + sig);
                    mm + +;
                    feature[m][mm].index = 13;
                    feature[m][mm].value = 1 / (double)(mycyc - j);
                    mm + +;
                    feature[m][mm].index = 14;
                    feature[m][mm].value = term;
                    mm + +;

                    if (j <= 6)
                    {
                    feature[m][mm].index = 15 + j;
                    feature[m][mm].value = 1;
                    mm++;
                    }

                    int8_t cc = fid[j] * 3 + sid[j] - (sid[j] > fid[j]);
                    if (j > 0)
                    {
                    cc += fid[j - 1] * 12;
                    feature[m][mm].index = 22 + cc;
                    feature[m][mm].value = 1;
                    mm++;
                    }

                    feature[m][mm].index = nrfeature;
                    feature[m][mm].value = 1;
                    mm++;

                    feature[m][mm].index = -1;
                    y[m] = match[inverse ? mycyc - 1 - j: j];
                    m + +;

                    j + +;
                    if (m == n)
                        break;
                }