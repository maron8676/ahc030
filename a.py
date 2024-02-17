import getpass
import math
import random
import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel


def li():
    return list(map(int, input().split()))


def operate(o_type, points):
    """
    入力取得
    :param o_type: 操作タイプ
    :param points: 点のリスト
    :return: 入力
    """
    global e_list, cost, oil_truth, oil_grid, eps

    print("{} {} {}".format(o_type, len(points), ' '.join(map(lambda x: "{} {}".format(x[0], x[1]), points))))

    if not is_local:
        # ローカル以外は標準入力
        return input()

    # 乱数取得
    e = e_list.pop()

    if o_type == "a":
        # 合っているか判定
        ans_set = set()
        for point in points:
            ans_set.add(point[0] * N + point[1])
        if oil_truth == ans_set:
            return "1"
        else:
            cost += 1
            return "0"

    cost += 1 / math.sqrt(len(points))
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
    value = round(e * math.sqrt(variance) + mean)
    return str(max(0, value))


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

# ローカル固有の入力
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

all_oil = 0
for field in fields:
    all_oil += len(field)

move_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# いもす法で全パターンの重ね合わせを計算
acc_grid = [[0] * N for _ in range(N)]
for field in fields:
    max_x = 0
    max_y = 0
    for cell in field:
        max_y = max(max_y, cell[0])
        max_x = max(max_x, cell[1])
    move_y = N - 1 - max_y
    move_x = N - 1 - max_x
    area = (move_y + 1) * (move_x + 1)
    for cell in field:
        acc_grid[cell[0]][cell[1]] += 1 / area
        if cell[0] + move_y + 1 < N:
            acc_grid[cell[0] + move_y + 1][cell[1]] -= 1 / area
        if cell[1] + move_x + 1 < N:
            acc_grid[cell[0]][cell[1] + move_x + 1] -= 1 / area
        if cell[0] + move_y + 1 < N and cell[1] + move_x + 1 < N:
            acc_grid[cell[0] + move_y + 1][cell[1] + move_x + 1] += 1 / area
for i in range(N):
    for j in range(1, N):
        acc_grid[i][j] = acc_grid[i][j - 1] + acc_grid[i][j]
for i in range(1, N):
    for j in range(N):
        acc_grid[i][j] = acc_grid[i - 1][j] + acc_grid[i][j]

# drill
cost = 0
has_oil = []
found_oil = 0
drilled_dict = [[-1] * N for _ in range(N)]
regressor = GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(), alpha=0.)
adjacent = set()

# 0がいくつか見つかるまで掘る
zero_num = 15 if eps <= 0.05 else 8 if eps <= 0.15 else 3 if eps <= 0.18 else 0
zero_cell_set = set()
# 調べなくても0と分かるセルを記録
for i in range(N):
    for j in range(N):
        if acc_grid[i][j] < 1 / N ** 2:
            point = (i, j)
            drilled_dict[point[0]][point[1]] = 0
            if len(zero_cell_set) < zero_num:
                zero_cell_set.add(point)

for i in range(N):
    if len(zero_cell_set) >= zero_num:
        break
    for j in range(N):
        if len(zero_cell_set) >= zero_num:
            break
        point = (i, j)
        points = [point]
        resp = int(operate("q", points))

        found_oil += resp
        drilled_dict[point[0]][point[1]] = resp
        if resp == 0:
            zero_cell_set.add(point)
        else:
            has_oil.append(point)
            for move in move_list:
                if 0 <= point[0] + move[0] < N and 0 <= point[1] + move[1] < N:
                    adjacent.add((point[0] + move[0], point[1] + move[1]))

estimated = [[-1] * N for _ in range(N)]
x_test = []
for i in range(N):
    for j in range(N):
        point = (i, j)
        x_test.append(point)
x_train = []
y_train = []
for i in range(N):
    for j in range(N):
        if drilled_dict[i][j] != -1:
            x_train.append((i, j))
            y_train.append(drilled_dict[i][j])
            estimated[i][j] = drilled_dict[i][j]
for i in range(N):
    for j in range(N):
        point = (i, j)
        if drilled_dict[i][j] != -1:
            continue
        if (i + j) % 2 == 1:
            continue

        points = [point]
        points.extend(zero_cell_set)
        resp = int(operate("q", points))

        x_train.append(point)
        y_train.append(max(0., resp - eps * (zero_num + 1)))
        estimated[i][j] = max(0., resp - eps * (zero_num + 1))
# regressor.fit(x_train, y_train)
# y_pred, y_std = regressor.predict(x_test, return_std=True)
# for i in range(N):
#     print(f"# {y_pred[i * N:(i + 1) * N]}")
for i in range(N):
    for j in range(N):
        if estimated[i][j] == -1:
            values = []
            for move in move_list:
                if 0 <= i + move[0] < N and 0 <= j + move[1] < N:
                    values.append(estimated[i + move[0]][j + move[1]])
            estimated[i][j] = sum(values) / len(values)

predict_with_pos = []
for i in range(N):
    for j in range(N):
        p = i * N + j
        predict_with_pos.append(((i, j), estimated[i][j]))
predict_with_pos.sort(key=lambda x: x[1])

# Mが小さいとき、全パターン作る
if M == 2:
    grid_candidates = dict()

    move_xy_list = []
    for i in range(M):
        field = fields[i]
        max_x = 0
        max_y = 0
        for cell in field:
            max_y = max(max_y, cell[0])
            max_x = max(max_x, cell[1])
        move_y = N - 1 - max_y
        move_x = N - 1 - max_x
        move_xy_list.append((move_y, move_x))

    for i1 in range(move_xy_list[0][0] + 1):
        for i2 in range(move_xy_list[0][1] + 1):
            for j1 in range(move_xy_list[1][0] + 1):
                for j2 in range(move_xy_list[1][1] + 1):
                    grid = [[0] * N for _ in range(N)]
                    for cell in fields[0]:
                        grid[cell[0] + i1][cell[1] + i2] += 1
                    for cell in fields[1]:
                        grid[cell[0] + j1][cell[1] + j2] += 1
                    grid_candidates[(i1, i2, j1, j2)] = grid

    loss_list = []
    for key in grid_candidates:
        loss = 0
        value = grid_candidates[key]
        for i in range(N):
            for j in range(N):
                p = i * N + j
                loss += (estimated[i][j] - value[i][j]) ** 2
        loss_list.append((key, loss))
    loss_list.sort(key=lambda x: x[1])

    for i in range(len(loss_list)):
        loss = loss_list[i]
        points = set()
        for j in range(len(fields)):
            field = fields[j]
            for cell in field:
                points.add((cell[0] + loss[0][j * 2], cell[1] + loss[0][j * 2 + 1]))
        resp = operate("a", points)
        if resp == "1":
            sys.exit()

# 埋まってそうなところから掘る
# 埋まっているところを見つけたら周りを掘る
while len(predict_with_pos) > 0:
    if found_oil >= all_oil:
        break

    while len(adjacent) > 0:
        if found_oil >= all_oil:
            break

        # 隣接で掘ってないところがあれば掘る
        point = adjacent.pop()
        if drilled_dict[point[0]][point[1]] != -1:
            continue

        points = [point]
        resp = int(operate("q", points))
        found_oil += resp
        drilled_dict[point[0]][point[1]] = resp

        if resp != 0:
            has_oil.append(point)
            for move in move_list:
                if 0 <= point[0] + move[0] < N and 0 <= point[1] + move[1] < N:
                    adjacent.add((point[0] + move[0], point[1] + move[1]))
    if found_oil >= all_oil:
        break

    predict = predict_with_pos.pop()
    point = predict[0]
    if drilled_dict[point[0]][point[1]] != -1:
        continue
    points = [point]
    resp = int(operate("q", points))
    found_oil += resp
    drilled_dict[point[0]][point[1]] = resp
    if resp != 0:
        has_oil.append(point)
        for move in move_list:
            if 0 <= point[0] + move[0] < N and 0 <= point[1] + move[1] < N:
                adjacent.add((point[0] + move[0], point[1] + move[1]))

resp = operate("a", has_oil)
assert found_oil == all_oil
assert resp == "1"

print(f"# cost: {cost}")
