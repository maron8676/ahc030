import getpass
import math
from collections import defaultdict


def li():
    return list(map(int, input().split()))


def get_input(q_type, points):
    """
    入力取得
    :param q_type: クエリタイプ
    :param points: 点のリスト
    :return: 入力
    """
    if not is_local:
        # ローカル以外は標準入力
        return input()

    # 乱数取得
    e = e_list.pop()

    if q_type == "a":
        # 合っているか判定
        ans_set = set()
        for point in points:
            ans_set.add(point[0] * N + point[1])
        if oil_truth == ans_set:
            return "1"
        else:
            return "0"

    if len(points) == 1:
        return str(oil_grid[points[0][0]][points[0][1]])

    # 平均と分散を計算
    k = len(points)
    vs = 0
    for point in points:
        vs += oil_grid[point[0]][point[1]]
    mean = (k - vs) * eps + vs * (1 - eps)
    variance = k * eps * (1 - eps)

    # 標準の値から変換
    return str(e * math.sqrt(variance) + mean)


is_local = getpass.getuser() == "omotl"

# read prior information
line = input().split()
N = int(line[0])
M = int(line[1])
eps = float(line[2])
fields = []
for _ in range(M):
    line = input().split()
    ps = []
    for i in range(int(line[0])):
        ps.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
    fields.append(ps)

if is_local:
    d_list = []
    for _ in range(M):
        d_list.append(li())
    oil_grid = []
    for _ in range(N):
        oil_grid.append(li())
    oil_truth = set()
    for i in range(N):
        for j in range(N):
            if oil_grid[i][j] > 0:
                oil_truth.add(i * N + j)
    e_list = []
    for _ in range(2 * N ** 2):
        e_list.append(float(input()))
    # popして使いたいのでreverse
    e_list = e_list[::-1]

# drill every square
cost = 0
has_oil = []
for i in range(N):
    for j in range(N):
        points = [[i, j]]
        print("q {} {}".format(len(points), ' '.join(map(lambda x: "{} {}".format(x[0], x[1]), points))))
        resp = get_input("q", points)
        if resp != "0":
            has_oil.append([i, j])

print("a {} {}".format(len(has_oil), ' '.join(map(lambda x: "{} {}".format(x[0], x[1]), has_oil))))
resp = get_input("a", has_oil)
assert resp == "1"
