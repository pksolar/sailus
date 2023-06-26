
def bleedFit(posFile):
    peakPos = np.loadtxt(posFile, skiprows=2).T

    coordMat1 = np.zeros((5000, 3000), int)

    for pointNum, coords in enumerate(peakPos.T):
        coordMat1[round(coords[0]), round(coords[1])] = pointNum + 1

    pixleSet = []
    peakNum = 0
    recordSet = set()
    posNUms = 0

    distanceSet = []
    for pos in range(0, len(peakPos[0]), 1):
        posNUms += 1
        # if 1792 < peakPos[0][pos] < 2304 and 1336 > peakPos[1][pos] > 824:# >3584    1648

        centerX = round(peakPos[0][pos])
        centerY = round(peakPos[1][pos])
        # print(centerX, centerY)
        matTemp = coordMat1[max(centerX - 5, 0): (centerX + 5), max(centerY - 5, 0): (centerY + 5)]

        if np.max(matTemp) > 0:
            posss = np.where(matTemp > 0)

            minsDis = 1000000
            for posNum in range(len(posss[0])):
                posTemp = matTemp[posss[0][posNum]][posss[1][posNum]]

                if (posTemp != (pos + 1)):
                    dis_x = np.square(peakPos[0][posTemp - 1] - peakPos[0][pos])
                    dis_y = np.square(peakPos[1][posTemp - 1] - peakPos[1][pos])

                    distance1 = dis_x + dis_y

                    if minsDis > distance1:
                        minsDis = distance1

            if minsDis > 0:
                if minsDis < 100:
                    distanceSet.append(math.sqrt(minsDis))
        else:
            print(pos)

    plt.hist(distanceSet, bins=100)
    plt.title(np.std(np.array(minsDis)))
    plt.show()