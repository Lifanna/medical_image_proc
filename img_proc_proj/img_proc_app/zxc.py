import math

x = [
0,	
0.5	,
0.3	,
0.7	,
0.9	,
1	,
1.2	,
1.5	,
1.57,
1.7	,
1.9	,
2.2	,
2.5	,
2.8	,
3	,
3.14,
]

res = []

for i in range(0, len(x)):
    res.append(round((0.5*math.pow(x[i], 2) - 4.8*x[i] + 3.5)*math.exp(-2*x[i]), 3))

print(res)
