import csv
import random
import math
import os
import subprocess

def dotproduct(veca, vecb):
    tot = 0.0
    for i in range(len(veca)):
        tot += veca[i] * vecb[i]
    return tot

def sigmoid(val):
    if val >= 0.0:
        return 1.0 / (1.0 + math.exp(-val))
    ex = math.exp(val)
    return ex / (1.0 + ex)

def predictproba(weights, bias, feats):
    return sigmoid(dotproduct(weights, feats) + bias)

def loaddata(filepath):
    rows = []
    with open(filepath, "r") as filehandle:
        reader = csv.reader(filehandle)
        header = next(reader)
        for line in reader:
            feats = [float(val) for val in line[:4]]
            target = int(line[4])
            rows.append((feats, target))
    return rows

def splitdata(rows, seed, trainsize):
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    trainrows = shuffled[:trainsize]
    testrows = shuffled[trainsize:]
    return trainrows, testrows

def fitscaler(trainrows):
    mins = []
    maxs = []
    for colidx in range(4):
        colvals = [row[0][colidx] for row in trainrows]
        mins.append(min(colvals))
        maxs.append(max(colvals))
    return mins, maxs

def scaledata(rows, mins, maxs):
    normrows = []
    for row in rows:
        normrow = []
        for colidx in range(4):
            colmin = mins[colidx]
            colmax = maxs[colidx]
            if colmax == colmin:
                normrow.append(0.0)
            else:
                normrow.append((row[0][colidx] - colmin) / (colmax - colmin))
        normrows.append((normrow, row[1]))
    return normrows

def evalmodel(weights, bias, dataset):
    totloss = 0.0
    correct = 0
    n = len(dataset)
    for feats, target in dataset:
        p = predictproba(weights, bias, feats)
        pclip = min(max(p, 1e-15), 1.0 - 1e-15)
        totloss += -(target * math.log(pclip) + (1.0 - target) * math.log(1.0 - pclip))
        pred = 1 if p >= 0.5 else 0
        if pred == target:
            correct += 1
    return totloss / n, correct / n

def trainmodel(trainrows, learningrate, maxepochs, tol):
    numfeats = len(trainrows[0][0])
    weights = [0.0] * numfeats
    bias = 0.0
    sampledlosses = []
    prevloss = float("inf")
    converged = False
    n = len(trainrows)

    for epoch in range(maxepochs):
        dw = [0.0] * numfeats
        db = 0.0
        totloss = 0.0

        for feats, target in trainrows:
            p = predictproba(weights, bias, feats)
            pclip = min(max(p, 1e-15), 1.0 - 1e-15)
            totloss += -(target * math.log(pclip) + (1.0 - target) * math.log(1.0 - pclip))
            diff = p - target
            for i in range(numfeats):
                dw[i] += diff * feats[i]
            db += diff

        curloss = totloss / n
        if epoch % 50 == 0:
            sampledlosses.append((epoch, curloss))

        if abs(prevloss - curloss) < tol:
            converged = True
            sampledlosses.append((epoch, curloss))
            break

        prevloss = curloss
        for i in range(numfeats):
            weights[i] -= learningrate * (dw[i] / n)
        bias -= learningrate * (db / n)

    if not converged and (maxepochs - 1, curloss) not in sampledlosses:
        sampledlosses.append((maxepochs, curloss))

    return weights, bias, sampledlosses, converged, curloss

def generatechart(history, outpath):
    width = 750
    height = 500
    padleft = 80
    padright = 40
    padtop = 60
    padbottom = 70
    plotw = width - padleft - padright
    ploth = height - padtop - padbottom

    minx = 0
    maxx = 10000
    miny = 0.30
    maxy = 0.75

    def tox(ep):
        return padleft + (ep - minx) / (maxx - minx) * plotw

    def toy(l):
        return padtop + (maxy - l) / (maxy - miny) * ploth

    colormapping = {
        0.001: "#d9534f",
        0.1: "#0275d8",
        10.0: "#f0ad4e"
    }

    svg = []
    svg.append(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">")
    svg.append("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    svg.append(f"<text x=\"{width // 2}\" y=\"35\" font-family=\"sans-serif\" font-size=\"20\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#333333\">Loss Curves for Three Learning Rates</text>")

    for yval in [0.30, 0.40, 0.50, 0.60, 0.70]:
        ypix = toy(yval)
        svg.append(f"<line x1=\"{padleft}\" y1=\"{ypix}\" x2=\"{width - padright}\" y2=\"{ypix}\" stroke=\"#e8e8e8\" stroke-width=\"1\"/>")
        svg.append(f"<text x=\"{padleft - 12}\" y=\"{ypix + 5}\" font-family=\"sans-serif\" font-size=\"12\" text-anchor=\"end\" fill=\"#666666\">{yval:.2f}</text>")

    for xval in [0, 2000, 4000, 6000, 8000, 10000]:
        xpix = tox(xval)
        svg.append(f"<line x1=\"{xpix}\" y1=\"{padtop}\" x2=\"{xpix}\" y2=\"{height - padbottom}\" stroke=\"#e8e8e8\" stroke-width=\"1\"/>")
        svg.append(f"<text x=\"{xpix}\" y=\"{height - padbottom + 22}\" font-family=\"sans-serif\" font-size=\"12\" text-anchor=\"middle\" fill=\"#666666\">{xval}</text>")

    svg.append(f"<line x1=\"{padleft}\" y1=\"{height - padbottom}\" x2=\"{width - padright}\" y2=\"{height - padbottom}\" stroke=\"#333333\" stroke-width=\"2\"/>")
    svg.append(f"<line x1=\"{padleft}\" y1=\"{padtop}\" x2=\"{padleft}\" y2=\"{height - padbottom}\" stroke=\"#333333\" stroke-width=\"2\"/>")
    svg.append(f"<text x=\"{width // 2}\" y=\"{height - 18}\" font-family=\"sans-serif\" font-size=\"14\" text-anchor=\"middle\" fill=\"#333333\">Epoch</text>")
    svg.append(f"<text x=\"24\" y=\"{height // 2}\" font-family=\"sans-serif\" font-size=\"14\" text-anchor=\"middle\" fill=\"#333333\" transform=\"rotate(-90 24 {height // 2})\">Binary Cross-Entropy Loss</text>")

    legx = width - padright - 170
    legy = padtop + 20
    svg.append(f"<rect x=\"{legx - 10}\" y=\"{legy - 15}\" width=\"170\" height=\"95\" fill=\"#ffffff\" stroke=\"#cccccc\" stroke-width=\"1\" rx=\"4\"/>")

    idx = 0
    for lr, sampled in history.items():
        color = colormapping.get(lr, "#333333")
        pathparts = []
        for pointidx, (ep, lossval) in enumerate(sampled):
            ptx = tox(ep)
            pty = toy(lossval)
            prefix = "M" if pointidx == 0 else "L"
            pathparts.append(f"{prefix} {ptx:.1f} {pty:.1f}")
        svg.append(f"<path d=\"{' '.join(pathparts)}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.5\"/>")

        curlegy = legy + idx * 25
        svg.append(f"<line x1=\"{legx}\" y1=\"{curlegy}\" x2=\"{legx + 25}\" y2=\"{curlegy}\" stroke=\"{color}\" stroke-width=\"3\"/>")
        svg.append(f"<text x=\"{legx + 35}\" y=\"{curlegy + 4}\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#333333\">lr = {lr}</text>")
        idx += 1

    svg.append("</svg>")
    svgcontent = "\n".join(svg)
    svgfile = outpath.replace(".png", ".svg")
    with open(svgfile, "w") as filehandle:
        filehandle.write(svgcontent)
    subprocess.run(["rsvg-convert", "-o", outpath, svgfile], check=True)

def finddatafile():
    paths = [
        "../../datasets/binary_classification.csv",
        "datasets/binary_classification.csv",
        "binary_classification.csv",
        "../datasets/binary_classification.csv",
        "../../Hacktoberfest26-AIML-Challenges/datasets/binary_classification.csv",
        "/home/sdheeraj/Documents/GitHub/Hacktoberfest26-AIML-Challenges/datasets/binary_classification.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return ""

def run():
    datapath = finddatafile()
    if not datapath:
        print("dataset file not found")
        return

    dataset = loaddata(datapath)
    trainrows, testrows = splitdata(dataset, seed=42, trainsize=400)
    mins, maxs = fitscaler(trainrows)
    trainnorm = scaledata(trainrows, mins, maxs)
    testnorm = scaledata(testrows, mins, maxs)

    lrs = [0.001, 0.1, 10.0]
    history = {}
    results = {}

    print("=== training logistic regression model ===")
    print("split: 400 train / 100 test (seed 42)")
    print("convergence criterion: abs(loss[t] - loss[t-1]) < 1e-6 or 10000 epochs\n")

    for lr in lrs:
        weights, bias, sampled, converged, trainloss = trainmodel(trainnorm, learningrate=lr, maxepochs=10000, tol=1e-6)
        testloss, testacc = evalmodel(weights, bias, testnorm)
        dummyloss, trainacc = evalmodel(weights, bias, trainnorm)
        epochs = sampled[-1][0]
        history[lr] = sampled
        results[lr] = (weights, bias, testacc, trainacc, trainloss, epochs, converged)
        print(f"lr={lr}:")
        print(f"  epochs: {epochs}")
        print(f"  converged: {converged}")
        print(f"  final train loss: {trainloss:.5f}")
        print(f"  train accuracy: {trainacc * 100:.1f}%")
        print(f"  test accuracy:  {testacc * 100:.1f}%")
        print(f"  weights: {[round(w, 4) for w in weights]}")
        print(f"  bias: {bias:.4f}\n")

    outimg = "loss_curves.png"
    generatechart(history, outimg)
    print(f"saved loss curves to {outimg}")

    bestlr = 0.1
    bestw, bestb, bestacc, dummya, dummyb, dummyc, dummyd = results[bestlr]
    featnames = ["queue length", "plate waste fraction", "menu board kcal", "handwriting neatness"]
    print(f"\nlearned feature weights at best lr ({bestlr}):")
    for j in range(4):
        print(f"  {featnames[j]}: {bestw[j]:.4f}")
    print(f"  bias: {bestb:.4f}")

run()
