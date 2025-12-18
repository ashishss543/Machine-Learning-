import csv, copy

def load_csv(file):
    with open(file) as f:
        data = list(csv.reader(f))[1:]
    return data

def consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

def candidate_elimination(data):
    n = len(data[0]) - 1
    S = ['Ø'] * n
    G = [['?'] * n]

    for row in data:
        x, label = row[:-1], row[-1]

        if label == 'Yes':
            G = [g for g in G if consistent(g, x)]
            for i in range(n):
                if S[i] == 'Ø': S[i] = x[i]
                elif S[i] != x[i]: S[i] = '?'
        else:
            G_new = []
            for g in G:
                if consistent(g, x):
                    for i in range(n):
                        if g[i] == '?' and S[i] != 'Ø':
                            h = g.copy()
                            h[i] = S[i]
                            if h not in G_new:
                                G_new.append(h)
                else:
                    G_new.append(g)
            G = G_new
    return S, G

data = load_csv("training_data.csv")
S, G = candidate_elimination(data)

print("S =", S)
print("G =")
for g in G:
    print(g)
