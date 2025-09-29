import random
import math

# -------------------------------
# Objective Function: Kapur's Entropy
# -------------------------------
def kapur_entropy(thresholds, image):
    thresholds = sorted([int(round(t)) for t in thresholds])
    thresholds = [0] + thresholds + [256]
    # compute histogram
    hist = [0]*256
    total_pixels = 0
    for row in image:
        for pixel in row:
            hist[pixel] += 1
            total_pixels += 1
    # normalize histogram
    prob = [h/total_pixels for h in hist]

    total_entropy = 0
    for i in range(len(thresholds)-1):
        start = thresholds[i]
        end = thresholds[i+1]
        P = [p for p in prob[start:end] if p>0]
        total_entropy += -sum([p*math.log(p) for p in P])
    return -total_entropy  # negative because GWO minimizes

# -------------------------------
# Grey Wolf Optimizer
# -------------------------------
def GWO_image(image, D, N=10, MaxIter=50, lb=0, ub=255):
    # Initialize wolves
    wolves = [[random.uniform(lb, ub) for _ in range(D)] for _ in range(N)]
    alpha_pos = [0]*D
    beta_pos = [0]*D
    delta_pos = [0]*D
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    for t in range(MaxIter):
        a = 2 - 2*t/MaxIter
        for i in range(N):
            fitness = kapur_entropy(wolves[i], image)
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos[:]
                beta_score, beta_pos = alpha_score, alpha_pos[:]
                alpha_score, alpha_pos = fitness, wolves[i][:]
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos[:]
                beta_score, beta_pos = fitness, wolves[i][:]
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i][:]

        # Update positions
        for i in range(N):
            for d in range(D):
                r1, r2 = random.random(), random.random()
                A1 = 2*a*r1 - a; C1 = 2*r2
                r1, r2 = random.random(), random.random()
                A2 = 2*a*r1 - a; C2 = 2*r2
                r1, r2 = random.random(), random.random()
                A3 = 2*a*r1 - a; C3 = 2*r2

                D_alpha = abs(C1*alpha_pos[d] - wolves[i][d])
                D_beta = abs(C2*beta_pos[d] - wolves[i][d])
                D_delta = abs(C3*delta_pos[d] - wolves[i][d])

                X1 = alpha_pos[d] - A1*D_alpha
                X2 = beta_pos[d] - A2*D_beta
                X3 = delta_pos[d] - A3*D_delta

                wolves[i][d] = (X1 + X2 + X3)/3
                # Clip to bounds
                if wolves[i][d] < lb: wolves[i][d] = lb
                if wolves[i][d] > ub: wolves[i][d] = ub

    return [int(round(x)) for x in alpha_pos]

# -------------------------------
# User Interaction
# -------------------------------
def main():
    # Load image as 2D list
    filename = input("Enter PGM image filename (grayscale): ")
    image = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.startswith('#')]
    if lines[0].strip() != 'P2':
        print("Only ASCII PGM (P2) supported.")
        return
    idx = 2
    while len(image) < int(lines[1].split()[1]):
        row = list(map(int, lines[idx].split()))
        image.append(row)
        idx += 1

    D = int(input("Enter number of thresholds: "))
    N = int(input("Enter number of wolves: "))
    MaxIter = int(input("Enter maximum iterations: "))

    best_thresholds = GWO_image(image, D, N, MaxIter)
    print("Best thresholds found:", best_thresholds)

    # Segment image
    thresholds = sorted(best_thresholds)
    thresholds = [0] + thresholds + [256]
    segmented = [[0 for _ in row] for row in image]
    for i in range(len(thresholds)-1):
        for r in range(len(image)):
            for c in range(len(image[0])):
                if thresholds[i] <= image[r][c] < thresholds[i+1]:
                    segmented[r][c] = int((i+1)*(255/(len(thresholds)-1)))

    # Save segmented image
    out_file = "segmented.pgm"
    with open(out_file, 'w') as f:
        f.write("P2\n")
        f.write(f"{len(segmented[0])} {len(segmented)}\n")
        f.write("255\n")
        for row in segmented:
            f.write(' '.join(map(str,row)) + '\n')

    print(f"Segmented image saved as {out_file}")

if __name__ == "__main__":
    main()
