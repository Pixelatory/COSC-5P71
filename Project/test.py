from functools import partial

def multiply(a, b):
    return a * b

x = 2
f = partial(multiply, x)

print(f)
print(f(10))
print(f.func, f.func == multiply)