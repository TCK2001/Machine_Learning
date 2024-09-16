# Visualization_Point.py
```python
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'o', label = 'versicolor')
```
### Why X[:50] and X[50:100]? 
* tips : Using `pd.set_option('display.max_rows', None)` to print all of the data results.
```
       0    1    2    3                4
0    5.1  3.5  1.4  0.2      Iris-setosa
1    4.9  3.0  1.4  0.2      Iris-setosa
2    4.7  3.2  1.3  0.2      Iris-setosa
3    4.6  3.1  1.5  0.2      Iris-setosa
4    5.0  3.6  1.4  0.2      Iris-setosa
5    5.4  3.9  1.7  0.4      Iris-setosa
6    4.6  3.4  1.4  0.3      Iris-setosa
7    5.0  3.4  1.5  0.2      Iris-setosa
8    4.4  2.9  1.4  0.2      Iris-setosa
9    4.9  3.1  1.5  0.1      Iris-setosa
10   5.4  3.7  1.5  0.2      Iris-setosa
11   4.8  3.4  1.6  0.2      Iris-setosa
12   4.8  3.0  1.4  0.1      Iris-setosa
13   4.3  3.0  1.1  0.1      Iris-setosa
14   5.8  4.0  1.2  0.2      Iris-setosa
15   5.7  4.4  1.5  0.4      Iris-setosa
16   5.4  3.9  1.3  0.4      Iris-setosa
17   5.1  3.5  1.4  0.3      Iris-setosa
18   5.7  3.8  1.7  0.3      Iris-setosa
19   5.1  3.8  1.5  0.3      Iris-setosa
20   5.4  3.4  1.7  0.2      Iris-setosa
21   5.1  3.7  1.5  0.4      Iris-setosa
22   4.6  3.6  1.0  0.2      Iris-setosa
23   5.1  3.3  1.7  0.5      Iris-setosa
24   4.8  3.4  1.9  0.2      Iris-setosa
25   5.0  3.0  1.6  0.2      Iris-setosa
26   5.0  3.4  1.6  0.4      Iris-setosa
27   5.2  3.5  1.5  0.2      Iris-setosa
28   5.2  3.4  1.4  0.2      Iris-setosa
29   4.7  3.2  1.6  0.2      Iris-setosa
30   4.8  3.1  1.6  0.2      Iris-setosa
31   5.4  3.4  1.5  0.4      Iris-setosa
32   5.2  4.1  1.5  0.1      Iris-setosa
33   5.5  4.2  1.4  0.2      Iris-setosa
34   4.9  3.1  1.5  0.1      Iris-setosa
35   5.0  3.2  1.2  0.2      Iris-setosa
36   5.5  3.5  1.3  0.2      Iris-setosa
37   4.9  3.1  1.5  0.1      Iris-setosa
38   4.4  3.0  1.3  0.2      Iris-setosa
39   5.1  3.4  1.5  0.2      Iris-setosa
40   5.0  3.5  1.3  0.3      Iris-setosa
41   4.5  2.3  1.3  0.3      Iris-setosa
42   4.4  3.2  1.3  0.2      Iris-setosa
43   5.0  3.5  1.6  0.6      Iris-setosa
44   5.1  3.8  1.9  0.4      Iris-setosa
45   4.8  3.0  1.4  0.3      Iris-setosa
46   5.1  3.8  1.6  0.2      Iris-setosa
47   4.6  3.2  1.4  0.2      Iris-setosa
48   5.3  3.7  1.5  0.2      Iris-setosa
49   5.0  3.3  1.4  0.2      Iris-setosa
50   7.0  3.2  4.7  1.4  Iris-versicolor
51   6.4  3.2  4.5  1.5  Iris-versicolor
52   6.9  3.1  4.9  1.5  Iris-versicolor
53   5.5  2.3  4.0  1.3  Iris-versicolor
54   6.5  2.8  4.6  1.5  Iris-versicolor
55   5.7  2.8  4.5  1.3  Iris-versicolor
56   6.3  3.3  4.7  1.6  Iris-versicolor
57   4.9  2.4  3.3  1.0  Iris-versicolor
58   6.6  2.9  4.6  1.3  Iris-versicolor
59   5.2  2.7  3.9  1.4  Iris-versicolor
60   5.0  2.0  3.5  1.0  Iris-versicolor
61   5.9  3.0  4.2  1.5  Iris-versicolor
62   6.0  2.2  4.0  1.0  Iris-versicolor
63   6.1  2.9  4.7  1.4  Iris-versicolor
64   5.6  2.9  3.6  1.3  Iris-versicolor
65   6.7  3.1  4.4  1.4  Iris-versicolor
66   5.6  3.0  4.5  1.5  Iris-versicolor
67   5.8  2.7  4.1  1.0  Iris-versicolor
68   6.2  2.2  4.5  1.5  Iris-versicolor
69   5.6  2.5  3.9  1.1  Iris-versicolor
70   5.9  3.2  4.8  1.8  Iris-versicolor
71   6.1  2.8  4.0  1.3  Iris-versicolor
72   6.3  2.5  4.9  1.5  Iris-versicolor
73   6.1  2.8  4.7  1.2  Iris-versicolor
74   6.4  2.9  4.3  1.3  Iris-versicolor
75   6.6  3.0  4.4  1.4  Iris-versicolor
76   6.8  2.8  4.8  1.4  Iris-versicolor
77   6.7  3.0  5.0  1.7  Iris-versicolor
78   6.0  2.9  4.5  1.5  Iris-versicolor
79   5.7  2.6  3.5  1.0  Iris-versicolor
80   5.5  2.4  3.8  1.1  Iris-versicolor
81   5.5  2.4  3.7  1.0  Iris-versicolor
82   5.8  2.7  3.9  1.2  Iris-versicolor
83   6.0  2.7  5.1  1.6  Iris-versicolor
84   5.4  3.0  4.5  1.5  Iris-versicolor
85   6.0  3.4  4.5  1.6  Iris-versicolor
86   6.7  3.1  4.7  1.5  Iris-versicolor
87   6.3  2.3  4.4  1.3  Iris-versicolor
88   5.6  3.0  4.1  1.3  Iris-versicolor
89   5.5  2.5  4.0  1.3  Iris-versicolor
90   5.5  2.6  4.4  1.2  Iris-versicolor
91   6.1  3.0  4.6  1.4  Iris-versicolor
92   5.8  2.6  4.0  1.2  Iris-versicolor
93   5.0  2.3  3.3  1.0  Iris-versicolor
94   5.6  2.7  4.2  1.3  Iris-versicolor
95   5.7  3.0  4.2  1.2  Iris-versicolor
96   5.7  2.9  4.2  1.3  Iris-versicolor
97   6.2  2.9  4.3  1.3  Iris-versicolor
98   5.1  2.5  3.0  1.1  Iris-versicolor
99   5.7  2.8  4.1  1.3  Iris-versicolor
...
```
You can see in the above data that the indicies 0 ~ 49 are setosa data, and 50 ~ 99 are versicolor data.

---

```python
plt.xlim(X[:50, 0].min() - 1, X[50:100, 0].max() + 1)
plt.ylim(X[0:50, 1].min() - 1, X[50:100, 1].max() + 1)
```
Set the range of the x and y axes.